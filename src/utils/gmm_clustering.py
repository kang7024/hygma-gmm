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
        min_k: int,
        max_k: int,
        n_agents: int,
        cov_type: str = "full",
        reg_covar: float = 1e-6,
        random_state: int = 1,
        use_bic_select: bool = True,
    ):
        self.min_k = min_k
        self.max_k = max_k
        self.n_agents = n_agents
        self.cov_type = cov_type
        self.reg_covar = reg_covar
        self.random_state = random_state
        self.use_bic_select = use_bic_select

        # 이전 스텝의 군집(그룹) 유지용
        self.prev_assign = None  # shape: [n_agents]
        self.prev_groups = [list(range(n_agents))]

    @staticmethod
    def _assign_to_groups(labels, n_agents):
        """라벨(길이 n_agents)을 그룹(인덱스 리스트들의 리스트)으로 변환."""
        groups = {}
        for i, lab in enumerate(labels):
            groups.setdefault(int(lab), []).append(i)
        # label index 순으로 정렬
        groups_list = [sorted(v) for k, v in sorted(groups.items(), key=lambda kv: kv[0])]
        # 비어있는 경우 fallback
        if not groups_list:
            groups_list = [list(range(n_agents))]
        return groups_list

    def _select_k_by_bic(self, X):
        """[min_k, max_k] 범위에서 BIC가 최소인 k를 선택."""
        best_k, best_bic, best_model = None, float("inf"), None
        for k in range(self.min_k, self.max_k + 1):
            gm = GaussianMixture(
                n_components=k,
                covariance_type=self.cov_type,
                reg_covar=self.reg_covar,
                random_state=self.random_state,
            )
            gm.fit(X)
            bic = gm.bic(X)
            if bic < best_bic:
                best_k, best_bic, best_model = k, bic, gm
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
        """
        Args:
          state_history: Tensor [B, T, state_dim] 혹은 [B, T, ...]
                         여기서는 '현재 시점에서 각 에이전트의 상태'를 만들기 위해
                         마지막 시점의 상태에서 에이전트 차원을 풀어 만드는 방식 사용.
          stability_threshold: (0~1) 비율. 라벨 변경 비율이 이보다 작으면 '갱신하지 않음'.

        Returns:
          updated(bool), new_groups(List[List[int]]), num_moved(int)
        """
        # ---- 입력 전처리: 마지막 타임스텝 기준의 (batch 평균) 상태 벡터 생성
        # state_history: [B, T, state_dim] or [B, T, n_agents, ...]
        sh = state_history
        if isinstance(sh, torch.Tensor):
            X = sh
        else:
            X = torch.as_tensor(sh)

        # 마지막 시점만 선택
        X_last = X[:, -1]

        # 에이전트 단위의 특징 벡터 만들기:
        # 상태에 에이전트 차원이 이미 있다면(예: [B, n_agents, feat]), 그걸 평균(B)으로 축소
        # 아니라면(전역상태) 에이전트별 관측/특징으로 바꾸는 별도 로직이 필요할 수 있음.
        # 여기서는 HYGMA 원 코드를 최대한 보존하기 위해 '전역 상태를 에이전트별로 동일하게 배정'하는 보수적 처리:
        if X_last.dim() == 3:        # [B, n_agents, feat]
            F = X_last.mean(dim=0)   # [n_agents, feat]
        elif X_last.dim() == 2:      # [B, feat] (전역)
            F = X_last.mean(dim=0, keepdim=True).repeat(self.n_agents, 1)  # [n_agents, feat]
        else:
            F = X_last.reshape(X_last.size(0), -1).mean(dim=0, keepdim=True).repeat(self.n_agents, 1)

        F_np = F.detach().cpu().numpy()

        # ---- GMM 피팅 & 라벨 예측
        labels, _ = self._fit_gmm_and_predict(F_np)

        # ---- 안정성 체크: 이전 할당 대비 이동 비율
        if self.prev_assign is None:
            moved = self.n_agents  # 첫 갱신으로 간주
        else:
            moved = int((labels != self.prev_assign).sum())

        move_ratio = moved / float(self.n_agents)
        if stability_threshold > 0.0 and move_ratio < stability_threshold:
            # 변경은 있었지만 임계치 미만이면 '유지'
            return False, self.prev_groups, moved

        # ---- 그룹 생성 및 저장
        new_groups = self._assign_to_groups(labels, self.n_agents)
        self.prev_assign = labels
        self.prev_groups = new_groups
        updated = True

        return updated, new_groups, moved