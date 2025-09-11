import time
import torch
import torch as th
import torch.nn as nn
from sklearn.mixture import GaussianMixture

from modules.agents import REGISTRY as agent_REGISTRY
from components.action_selectors import REGISTRY as action_REGISTRY

# HGCN/하이퍼그래프는 그대로 사용 (HYGMA의 표현력 유지)
from utils.HGCN import HGCN
from utils.gmm_clustering import DynamicGMMClustering  # ← 신규 위치에서 임포트

class GMMHYGMA(nn.Module):
    """
    HYGMA MAC의 클러스터링 백엔드를 GMM으로 교체한 버전.
    - 하이퍼그래프 + HGCN은 유지
    - 'DynamicSpectralClustering' → 'GMMDynamicClustering'
    - 나머지 인터페이스는 기존 HYGMA와 동일하게 유지
    """
    def __init__(self, scheme, groups, args):
        super().__init__()
        self.t_env = 0
        self.args = args
        self.n_agents = args.n_agents
        self.input_shape = self._get_input_shape(scheme)
        self.agent_output_type = args.agent_output_type
        self.action_selector = action_REGISTRY[args.action_selector](args)
        self.hidden_states = None

        # GMM 클러스터러 초기화
        self.clustering = DynamicGMMClustering(
            min_k=args.min_clusters,
            max_k=args.max_clusters,
            n_agents=args.n_agents,
            cov_type=getattr(args, "gmm_cov_type", "full"),
            reg_covar=float(getattr(args, "gmm_reg_covar", 1e-6)),
            random_state=getattr(args, "gmm_random_state", 1),
            use_bic_select=getattr(args, "gmm_use_bic_select", True),
        )
        self.last_clustering_step = 0
        self.clustering_interval = args.clustering_interval
        self.stability_threshold = args.stability_threshold

        # 초기 그룹: 모든 에이전트 한 그룹
        self.agent_groups = [list(range(self.n_agents))]
        self.current_probs = None  # 초기 probs를 None으로 설정

        # Agent & HGCN 구성
        self._build_agents(self.input_shape)
        self.hgcn_in_dim = self.input_shape
        self.hgcn_hidden_dim = args.hgcn_hidden_dim
        self.hgcn_num_layers = args.hgcn_num_layers
        self.hgcn_out_dim = args.hgcn_out_dim

        self.hgcn = HGCN(
            in_dim=self.hgcn_in_dim,
            hidden_dim=self.hgcn_hidden_dim,
            out_dim=self.hgcn_out_dim,
            num_agents=args.n_agents,
            num_groups=len(self.agent_groups),
            num_layers=self.hgcn_num_layers,
        )

        # 고정 스텝 파라미터(원 코드 호환)
        self.training_steps = 0
        self.fix_hgcn_steps = args.fix_hgcn_steps
        self.fix_grouping_steps = args.fix_grouping_steps
        

    # -------- 핵심 실행 경로 ----------
    def select_actions(self, ep_batch, t_ep, t_env, bs=slice(None), test_mode=False):
        avail_actions = ep_batch["avail_actions"][:, t_ep]
        agent_outputs = self.forward(ep_batch, t_ep, t_env, test_mode=test_mode)
        chosen_actions = self.action_selector.select_action(
            agent_outputs[bs], avail_actions[bs], t_env, test_mode=test_mode
        )
        return chosen_actions

    def forward(self, ep_batch, t, t_env, test_mode=False):
        self.t_env = t_env
        agent_inputs = self._build_inputs(ep_batch, t)
        avail_actions = ep_batch["avail_actions"][:, t]

        # 초기 probs는 self.current_probs 사용, 갱신 전까지 유지
        probs = self.current_probs

        # 학습 초반 그룹 고정 기간이 끝나면 일정 주기로 GMM 재군집
        if (not test_mode) and (self.training_steps < self.fix_grouping_steps):
            if t_env - self.last_clustering_step >= self.clustering_interval:
                start_time = time.time()
                state_history = self._get_state_history(ep_batch, t)
                updated, new_groups, num_moved, new_probs = self.clustering.update_groups(
                    state_history, self.stability_threshold
                )
                if updated:
                    self.agent_groups = new_groups
                    self.last_clustering_step = t_env
                    # HGCN의 그룹 수만 갱신(가중치 보존)
                    self.hgcn.update_groups(len(new_groups))
                    self.current_probs = new_probs  # probs 업데이트
                else:
                    # 업데이트가 없을 때도 probs를 유지
                    new_probs = self.current_probs
                probs = new_probs  # 현재 forward에서 사용

        # [B, n_agents, feat]
        agent_inputs = agent_inputs.view(ep_batch.batch_size, self.n_agents, -1)

        # 하이퍼그래프 구성 (probs가 None일 경우 하드 할당 사용)
        hypergraph = self._create_hypergraph(self.agent_groups, ep_batch.batch_size, probs)

        # HGCN 통과 (그룹 특성 추출)
        hgcn_features = self.hgcn(agent_inputs, hypergraph)

        # 원 입력 + HGCN 특성 결합 → 에이전트 네트워크로
        combined_inputs = th.cat([agent_inputs, hgcn_features], dim=-1)
        combined_inputs = combined_inputs.view(ep_batch.batch_size * self.n_agents, -1)

        agent_outs, self.hidden_states = self.agent(combined_inputs, self.hidden_states)

        # 정책 로짓 후처리(원 코드와 동일)
        if self.agent_output_type == "pi_logits":
            if getattr(self.args, "mask_before_softmax", True):
                reshaped_avail_actions = avail_actions.reshape(ep_batch.batch_size * self.n_agents, -1)
                agent_outs[reshaped_avail_actions == 0] = -1e10
            agent_outs = th.nn.functional.softmax(agent_outs, dim=-1)
            if not test_mode:
                epsilon_action_num = agent_outs.size(-1)
                if getattr(self.args, "mask_before_softmax", True):
                    epsilon_action_num = reshaped_avail_actions.sum(dim=1, keepdim=True).float()
                agent_outs = ((1 - self.action_selector.epsilon) * agent_outs
                             + th.ones_like(agent_outs) * self.action_selector.epsilon / epsilon_action_num)
                if getattr(self.args, "mask_before_softmax", True):
                    agent_outs[reshaped_avail_actions == 0] = 0.0

        if not test_mode:
            self.training_steps += 1

        return agent_outs.view(ep_batch.batch_size, self.n_agents, -1)
    
    def _create_hypergraph(self, groups, batch_size, probs=None):
        """그룹-노드 연결을 나타내는 초그래프(하이퍼그래프) 인시던스 텐서 [B, n_groups, n_agents]."""
        n_groups = len(groups)
        device = next(self.parameters()).device
        H = torch.zeros(batch_size, n_groups, self.n_agents, device=device)
        if probs is not None:
            probs_t = torch.tensor(probs, device=device)  # [n_agents, n_groups]
            if probs_t.shape[0] != self.n_agents or probs_t.shape[1] != n_groups:
                raise ValueError(f"Probs shape {probs_t.shape} mismatch with n_agents={self.n_agents}, n_groups={n_groups}")
            # [n_agents, n_groups] → [n_groups, n_agents]로 transpose
            probs_t = probs_t.T  # [n_groups, n_agents]
            # [n_groups, n_agents] → [batch_size, n_groups, n_agents]로 확장
            H = probs_t.unsqueeze(0).expand(batch_size, -1, -1)  # [B, n_groups, n_agents]
        else:
            # 초기 None 시 uniform 소프트 probs로 대체
            uniform_probs = torch.ones(n_groups, self.n_agents, device=device) / n_groups
            H = uniform_probs.unsqueeze(0).expand(batch_size, -1, -1)  # [B, n_groups, n_agents]
            # 또는 기존 하드: for gi, g in enumerate(groups): H[:, gi, g] = 1
        return H

    def _get_state_history(self, ep_batch, t):
        history_length = min(self.args.state_history_length, t + 1)
        start = max(0, t - history_length + 1)
        return ep_batch["obs"][:, start:t + 1]  # [B, T, n_agents, obs_dim]

    def get_attention_weights(self):
        return self.hgcn.get_attention_weights() if hasattr(self.hgcn, "get_attention_weights") else None

    def init_hidden(self, batch_size):
        self.hidden_states = self.agent.init_hidden().unsqueeze(0).expand(batch_size, self.n_agents, -1)

    def parameters(self):
        return self.agent.parameters()

    def clone(self, scheme, groups, args):
        new_mac = type(self)(scheme, groups, args)
        new_mac.load_state(self)
        return new_mac

    def load_state(self, other_mac):
        self.agent.load_state_dict(other_mac.agent.state_dict())
        if hasattr(self, "hgcn") and hasattr(other_mac, "hgcn"):
            self.hgcn.load_state_dict(other_mac.hgcn.state_dict())
        self.agent_groups = [g[:] for g in other_mac.agent_groups]

    def cuda(self):
        self.agent.cuda()

    def save_models(self, path):
        th.save(self.agent.state_dict(), f"{path}/agent.th")

    def load_models(self, path):
        self.agent.load_state_dict(th.load(f"{path}/agent.th", map_location=lambda s, l: s))

    def _build_agents(self, input_shape):
        """HGCN 출력 차원을 포함해 에이전트 입력을 재구성."""
        self.agent = agent_REGISTRY[self.args.agent](input_shape + self.args.hgcn_out_dim, self.args)



    def _build_inputs(self, batch, t):
            """
            Make per-agent inputs at time t.
            Returns: (bs * n_agents, input_dim)
            """
            bs = batch.batch_size
            device = batch.device
            n_agents = self.args.n_agents

            # 1) 관측 (bs, n_agents, obs_dim)
            inputs = [batch["obs"][:, t]]  # (B, A, O)

            # 2) 직전 one-hot action (선택)
            if getattr(self.args, "obs_last_action", False):
                if t == 0:
                    last_actions = th.zeros_like(batch["actions_onehot"][:, t])  # (B, A, n_actions)
                else:
                    last_actions = batch["actions_onehot"][:, t-1]
                inputs.append(last_actions)

            # 3) agent id one-hot (선택)
            if getattr(self.args, "obs_agent_id", False):
                agent_ids = th.eye(n_agents, device=device).unsqueeze(0).expand(bs, -1, -1)  # (B, A, A)
                inputs.append(agent_ids)

            # (B, A, O+...) -> (B*A, O+...)
            inputs = th.cat(inputs, dim=-1)                         # (B, A, D)
            inputs = inputs.reshape(bs * n_agents, -1)              # (B*A, D)
            return inputs
    
    def _get_input_shape(self, scheme):
        input_shape = scheme["obs"]["vshape"]
        if self.args.obs_last_action:
            input_shape += scheme["actions_onehot"]["vshape"][0]
        if self.args.obs_agent_id:
            input_shape += self.n_agents
        return input_shape