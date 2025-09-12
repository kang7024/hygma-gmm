# utils/dynamic_clustering.py
# GMM 기반 동적(소프트) 클러스터링 + 하이퍼그래프 생성
# 기존 DynamicSpectralClustering와 동일한 public API (cluster/update_groups 등)를 유지해
# 기존 호출부 수정 최소화

from typing import List, Tuple, Optional
import numpy as np
import torch
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score

class DynamicGMMClustering:
    """
    GMM 기반의 동적 클러스터러.
    - 스펙트럴 클러스터링 대신 GMM으로 '에이전트 상태'를 군집화.
    - 클러스터 개수는 [min_k, max_k] 범위에서 BIC 최적 모델을 선택(기본).
    - 안정성(stability_threshold)로 군집 이동이 너무 잦을 때 갱신을 억제.
    """
    def __init__(
        self,
        min_k: int, # 최소 클러스터 개수
        max_k: int, # 최대 클러스터 개수
        n_agents: int, # 전체 에이전트 수
        cov_type: str = "full", # GMM 공분산 형태 : full", "tied", "diag", "spherical"
        reg_covar: float = 1e-6, # 공분산 행렬의 수치적 안정성 위해 추가하는 작은 값
        random_state: int = 1, # 난수 시드 고정
        use_bic_select: bool = True, # Tue면 BIC로 k 선택, False면 max_k 고정
    ):
        self.min_k = min_k
        self.max_k = max_k
        self.n_agents = n_agents
        self.cov_type = cov_type
        self.reg_covar = reg_covar
        self.random_state = random_state
        self.use_bic_select = use_bic_select

        # 이전 스텝의 군집(그룹) 유지용
        self.prev_assign = None  # 이전 군집 라벨 저장, shape: [n_agents]
        self.prev_groups = [list(range(n_agents))] # 이전 그룹 구조, 초기에는 모두 한 그룹
        self.prev_probs = None  # 이전 군집 확률 분포 저장

    @staticmethod
    def _assign_to_groups(labels, n_agents):
        """라벨(길이 n_agents)을 그룹(인덱스 리스트들의 리스트)으로 변환"""
        groups = {}
        for i, lab in enumerate(labels):
            groups.setdefault(int(lab), []).append(i)
        # label index 순으로 정렬
        groups_list = [sorted(v) for k, v in sorted(groups.items(), key=lambda kv: kv[0])]
        # 비어있는 경우 fallback
        if not groups_list:
            groups_list = [list(range(n_agents))]
        return groups_list

    def _select_k_by_bic(self, X, n_init=5, max_iter=200, tol=1e-3, force_covariance=None):
        """[min_k, max_k] 범위에서 BIC가 최소인 k를 선택"""
        best_k, best_bic, best_model = None, float("inf"), None
        cov_type = force_covariance if force_covariance is not None else self.cov_type # add 250911

                # 방어: 샘플 수가 너무 작으면 최대 k 제한
        n_samples, d = X.shape
        practical_max_k = min(self.max_k, max(1, n_samples - 1))  # k <= n_samples-1

        # for k in range(self.min_k, self.max_k + 1):
        #     gm = GaussianMixture(
        #         n_components=k,
        #         covariance_type=self.cov_type,
        #         reg_covar=self.reg_covar,
        #         random_state=self.random_state,
        #     )
        #     gm.fit(X)
        #     bic = gm.bic(X)
        #     if bic < best_bic:
        #         best_k, best_bic, best_model = k, bic, gm
        # return best_k, best_model

        for k in range(self.min_k, practical_max_k + 1):
            gm = GaussianMixture(
                n_components=k,
                covariance_type=cov_type,
                reg_covar=self.reg_covar,
                random_state=self.random_state,
                n_init=n_init,
                max_iter=max_iter,
                tol=tol,
                init_params='kmeans'
            )
            try:
                gm.fit(X)
            except Exception as e:
                # 만약 수치적 문제가 발생하면 건너뛰기
                continue

            bic = gm.bic(X)
            if bic < best_bic:
                best_k, best_bic, best_model = k, bic, gm

        # 만약 아무것도 선택되지 않았다면 최소값 반환
        if best_model is None:
            # fallback: k = min_k로 학습
            gm = GaussianMixture(
                n_components=max(self.min_k, 1),
                covariance_type=cov_type,
                reg_covar=self.reg_covar,
                random_state=self.random_state,
                n_init=n_init,
                max_iter=max_iter
            )
            gm.fit(X)
            return max(self.min_k, 1), gm

        return best_k, best_model

    def _fit_gmm_and_predict(self, X, force_k=None):
        """k가 주어지면 고정, 아니면 BIC로 k 선택 후 예측 라벨 반환."""
        if force_k is not None:
            gm = GaussianMixture(
                n_components=force_k,
                covariance_type=self.cov_type,
                reg_covar=self.reg_covar,
                random_state=self.random_state,
            )
            labels = gm.fit_predict(X)
            return labels, gm
        # BIC로 k 선택
        k, gm = self._select_k_by_bic(X) if self.use_bic_select else (self.max_k, None)
        if gm is None:  # 방어
            gm = GaussianMixture(
                n_components=max(self.min_k, 1),
                covariance_type=self.cov_type,
                reg_covar=self.reg_covar,
                random_state=self.random_state,
            )
            gm.fit(X)
        labels = gm.predict(X)
        return labels, gm

    def update_groups(self, state_history, stability_threshold=0.0):
        # ---- 입력 전처리: 에이전트별 특징 벡터 생성
        sh = torch.as_tensor(state_history) if not isinstance(state_history, torch.Tensor) else state_history
        
        # 마지막 타임스텝의 에이전트별 상태 추출 (SMAC 환경 가정: [B, T, n_agents, feat])
        X_last = sh[:, -1]  # [B, n_agents, feat]
        if X_last.dim() == 3:
            F = X_last.mean(dim=0)  # [n_agents, feat] - 배치 평균
        else:
            raise ValueError("state_history shape mismatch: expected [B, T, n_agents, feat]")

        F_np = F.detach().cpu().numpy()

        # ---- GMM 피팅 & 라벨 예측
        labels, gm = self._fit_gmm_and_predict(F_np)
        probs = gm.predict_proba(F_np)  # [n_agents, k]

        # ---- 안정성 체크: 이전 할당 대비 이동 비율
        if self.prev_assign is None:
            moved = self.n_agents
            prev_probs = probs  # For first time, use current
        else:
            moved = int((labels != self.prev_assign).sum())
            prev_probs = self.prev_probs if self.prev_probs is not None else probs

        move_ratio = moved / float(self.n_agents)
        if stability_threshold > 0.0 and move_ratio < stability_threshold:
            # 변경은 있었지만 임계치 미만이면 '유지'
            return False, self.prev_groups, moved, prev_probs  # Return previous probs

        # ---- 그룹 생성 및 저장
        new_groups = self._assign_to_groups(labels, self.n_agents)
        self.prev_assign = labels
        self.prev_groups = new_groups
        self.prev_probs = probs
        updated = True

        return updated, new_groups, moved, probs  # Return new probs