import time
import torch
import torch as th
import torch.nn as nn

from modules.agents import REGISTRY as agent_REGISTRY
from components.action_selectors import REGISTRY as action_REGISTRY

from utils.HGCN import HGCN
from utils.dynamic_clustering import DynamicSpectralClustering
from utils.gmm_clustering import DynamicGMMClustering

# AERIAL-style: (B, N, H) -> rect (B, R)
from modules.transformers import HiddenStateTransformer


class HYGMA(nn.Module):
    def __init__(self, scheme, groups, args):
        super(HYGMA, self).__init__()
        self.t_env = 0
        self.args = args
        self.n_agents = args.n_agents
        self.input_shape = self._get_input_shape(scheme)
        self.agent_output_type = args.agent_output_type
        self.action_selector = action_REGISTRY[args.action_selector](args)
        self.hidden_states = None

        # 동적 클러스터링 초기화 (스펙트럼 or GMM)
        # self.clustering = DynamicSpectralClustering(args.min_clusters, args.max_clusters, args.n_agents)
        self.clustering = DynamicGMMClustering(args.min_clusters, args.max_clusters, args.n_agents)
        self.last_clustering_step = 0
        self.clustering_interval = args.clustering_interval
        self.stability_threshold = args.stability_threshold

        # 에이전트 그룹 초기화
        # 기본: 모든 에이전트를 하나의 그룹으로 시작
        self.agent_groups = [list(range(self.n_agents))]
        # 개별 그룹 시작하려면 아래 주석 해제
        # self.agent_groups = [[i] for i in range(self.n_agents)]

        # 에이전트 네트워크 구성 (HGCN 출력 포함)
        self._build_agents(self.input_shape)

        # HGCN 초기화
        self.hgcn_in_dim = self.input_shape
        self.hgcn_hidden_dim = args.hgcn_hidden_dim
        self.hgcn_num_layers = args.hgcn_num_layers
        self.hgcn_out_dim = self.args.hgcn_out_dim

        self.hgcn = HGCN(
            in_dim=self.hgcn_in_dim,
            hidden_dim=self.hgcn_hidden_dim,
            out_dim=self.hgcn_out_dim,
            num_agents=args.n_agents,
            num_groups=len(self.agent_groups),
            num_layers=self.hgcn_num_layers
        )

        # AERIAL: RNN hidden -> Transformer -> rect(B, R)
        self.use_hidden_state_transformer = bool(getattr(self.args, "use_hidden_state_transformer", False))
        self.hidden_transformer = None
        self._last_transformer_out = None  # rect 캐시

        if self.use_hidden_state_transformer:
            # 에이전트 RNN 히든 차원(H)과 rect 차원(R)
            rnn_hidden_dim = getattr(self.args, "hidden_dim", None)
            if rnn_hidden_dim is None:
                raise ValueError("args.hidden_dim (에이전트 RNN hidden 크기)가 필요합니다.")
            rect_dim = int(getattr(self.args, "rect_dim", rnn_hidden_dim))

            self.hidden_transformer = HiddenStateTransformer(
                input_dim=rnn_hidden_dim,
                rect_dim=rect_dim,
                n_heads=getattr(self.args, "hidden_state_transformer_heads", 4),
                n_layers=getattr(self.args, "hidden_state_transformer_layers", 1),
                attn_dim=getattr(self.args, "hidden_state_transformer_attn_dim", 64),
                ff_multiplier=getattr(self.args, "hidden_state_transformer_ff_multiplier", 2),
                dropout=getattr(self.args, "hidden_state_transformer_dropout", 0.0),
            )

        # 학습 단계/고정 파라미터
        self.training_steps = 0
        self.fix_hgcn_steps = args.fix_hgcn_steps
        self.fix_grouping_steps = args.fix_grouping_steps


    def select_actions(self, ep_batch, t_ep, t_env, bs=slice(None), test_mode=False):
        # 기존 선택 정책 유지
        avail_actions = ep_batch["avail_actions"][:, t_ep]
        agent_outputs = self.forward(ep_batch, t_ep, t_env, test_mode=test_mode)
        chosen_actions = self.action_selector.select_action(
            agent_outputs[bs], avail_actions[bs], t_env, test_mode=test_mode
        )
        return chosen_actions

    def forward(self, ep_batch, t, t_env, test_mode=False):
        self.t_env = t_env
        bs = ep_batch.batch_size

        agent_inputs = self._build_inputs(ep_batch, t)  # (B*A, D_in)
        avail_actions = ep_batch["avail_actions"][:, t]

        # (학습 초반) 그룹 고정 기간 동안 주기적으로 재군집
        if not test_mode and self.training_steps < self.fix_grouping_steps:
            if t_env - self.last_clustering_step >= self.clustering_interval:
                start_time = time.time()
                print(
                    f"Clustering check @ t_env={t_env} (interval={self.clustering_interval}) "
                    f"last={self.last_clustering_step}, elapsed={t_env - self.last_clustering_step}"
                )
                print(f"current groups: {self.agent_groups}")

                state_history = self._get_state_history(ep_batch, t)
                groups_updated, new_groups, num_moved = self.clustering.update_groups(
                    state_history, self.stability_threshold
                )

                if groups_updated:
                    self.agent_groups = new_groups
                    self.last_clustering_step = t_env
                    # HGCN 그룹 수 갱신(가중치는 유지)
                    self.hgcn.update_groups(len(new_groups))
                    print(
                        f"Groups updated @ t_env={t_env}. moved={num_moved}/{self.n_agents}, "
                        f"new groups: {self.agent_groups}"
                    )
                else:
                    if num_moved > 0:
                        print(
                            f"Groups unchanged @ t_env={t_env} (stability threshold). "
                            f"potential moves: {num_moved}/{self.n_agents}"
                        )
                    else:
                        print(f"Groups unchanged @ t_env={t_env}. no potential moves.")
                print(f"Clustering elapsed: {time.time() - start_time:.4f}s")

        # HGCN 입력 차원으로 재배치: (B, N, D_in)
        agent_inputs = agent_inputs.view(bs, self.n_agents, -1)

        # 하이퍼그래프(그룹-에이전트 연결) 생성
        hypergraph = self._create_hypergraph(self.agent_groups, bs)

        # HGCN 처리: 그룹 공유 특성 추출 (B, N, hgcn_out_dim)
        hgcn_features = self.hgcn(agent_inputs, hypergraph)

        # 원 입력 + HGCN 특성 결합 → 에이전트 네트워크 입력으로 사용
        combined_inputs = th.cat([agent_inputs, hgcn_features], dim=-1)   # (B, N, D_in + hgcn_out)
        combined_inputs = combined_inputs.view(bs * self.n_agents, -1)    # (B*N, D_cat)

        # 에이전트 RNN 통과
        agent_outs, self.hidden_states = self.agent(combined_inputs, self.hidden_states)
        # agent_outs: (B*N, n_actions)  /  hidden_states: 구현체에 따라 (B, N, H) 또는 (B*N, H)

        # === AERIAL rect 생성: RNN 은닉 상태 -> Transformer -> rect(B, R) ===
        if self.use_hidden_state_transformer:
            # 컨트롤러 내부 은닉 상태를 (B, N, H)로 정리
            # init_hidden에서 (B, N, H)로 관리하므로 동일 형태로 view
            hidden_bnH = self.hidden_states.view(bs, self.n_agents, -1)  # (B, N, H)
            rect = self.hidden_transformer(hidden_bnH)                   # (B, R)
            self._last_transformer_out = rect.detach()                   # learner에서 읽도록 캐시
        else:
            self._last_transformer_out = None

        # 에이전트 출력 후처리 (정책 로짓/마스킹/epsilon-greedy)
        if self.agent_output_type == "pi_logits":
            if getattr(self.args, "mask_before_softmax", True):
                reshaped_avail_actions = avail_actions.reshape(bs * self.n_agents, -1)
                agent_outs[reshaped_avail_actions == 0] = -1e10

            agent_outs = th.nn.functional.softmax(agent_outs, dim=-1)

            if not test_mode:
                epsilon_action_num = agent_outs.size(-1)
                if getattr(self.args, "mask_before_softmax", True):
                    epsilon_action_num = reshaped_avail_actions.sum(dim=1, keepdim=True).float()
                agent_outs = (
                    (1 - self.action_selector.epsilon) * agent_outs
                    + th.ones_like(agent_outs) * self.action_selector.epsilon / epsilon_action_num
                )
                if getattr(self.args, "mask_before_softmax", True):
                    agent_outs[reshaped_avail_actions == 0] = 0.0

        if not test_mode:
            self.training_steps += 1

        # (B, N, n_actions)로 반환
        return agent_outs.view(bs, self.n_agents, -1)

    def _create_hypergraph(self, groups, batch_size):
        """
        하이퍼그래프 인접 행렬 생성 (그룹-에이전트 연결을 표현).
        HGCN 입력으로 사용되는 인시던스 텐서: (B, n_groups, n_agents)
        """
        n_groups = len(groups)
        # agent 파라미터 기준으로 디바이스 통일
        device = next(self.agent.parameters()).device
        H = torch.zeros(batch_size, n_groups, self.n_agents, device=device)
        for gi, g in enumerate(groups):
            H[:, gi, g] = 1
        return H

    def _get_state_history(self, ep_batch, t):
        """
        상태 히스토리 (군집 업데이트용 윈도우) 반환: (B, L_hist, state_dim)
        """
        history_length = min(self.args.state_history_length, t + 1)
        start = max(0, t - history_length + 1)
        return ep_batch["state"][:, start:t + 1]

    def get_attention_weights(self):
        """
        HGCN 내부 어텐션 가중치가 노출되는 경우(디버깅/해석) 반환
        """
        if hasattr(self.hgcn, 'get_attention_weights'):
            return self.hgcn.get_attention_weights()
        return None

    def get_last_transformer_out(self):
        """
        최신 rect(B, R) 반환. (use_hidden_state_transformer=True일 때만 유효)
        """
        return self._last_transformer_out

    def init_hidden(self, batch_size):
        """
        에피소드/배치 시작 시 에이전트 RNN 히든 상태 초기화
        """
        self.hidden_states = self.agent.init_hidden().unsqueeze(0).expand(batch_size, self.n_agents, -1)
        self._last_transformer_out = None

    def parameters(self):
        """
        옵티마이저에 전달할 파라미터 집합.
        (에이전트 + Transformer 포함)
        """
        params = list(self.agent.parameters())
        if self.hidden_transformer is not None:
            params += list(self.hidden_transformer.parameters())
        return params

    def clone(self, scheme, groups, args):
        """
        컨트롤러 사본 생성(타깃 네트워크 구성 등에 사용)
        """
        new_mac = type(self)(scheme, groups, args)
        new_mac.load_state(self)
        return new_mac

    def load_state(self, other_mac):
        """
        파라미터/구성 동기화 (타깃 네트워크 갱신 등)
        """
        self.agent.load_state_dict(other_mac.agent.state_dict())
        if hasattr(self, 'hgcn') and hasattr(other_mac, 'hgcn'):
            self.hgcn.load_state_dict(other_mac.hgcn.state_dict())
        self.agent_groups = [group[:] for group in other_mac.agent_groups]

        # Transformer까지 복제
        if self.hidden_transformer is not None and getattr(other_mac, "hidden_transformer", None) is not None:
            self.hidden_transformer.load_state_dict(other_mac.hidden_transformer.state_dict())

    def cuda(self):
        """
        GPU로 이동
        """
        self.agent.cuda()
        self.hgcn.cuda()
        if self.hidden_transformer is not None:
            self.hidden_transformer.cuda()

    def save_models(self, path):
        """
        체크포인트 저장
        """
        th.save(self.agent.state_dict(), f"{path}/agent.th")
        th.save(self.hgcn.state_dict(), f"{path}/hgcn.th")
        if self.hidden_transformer is not None:
            th.save(self.hidden_transformer.state_dict(), f"{path}/hidden_transformer.th")

    def load_models(self, path):
        """
        체크포인트 로드
        """
        self.agent.load_state_dict(th.load(f"{path}/agent.th", map_location=lambda s, l: s))
        self.hgcn.load_state_dict(th.load(f"{path}/hgcn.th", map_location=lambda s, l: s))
        if self.hidden_transformer is not None:
            self.hidden_transformer.load_state_dict(
                th.load(f"{path}/hidden_transformer.th", map_location=lambda s, l: s)
            )

    def _build_agents(self, input_shape):
        """
        HGCN이 만들어낸 추가 특징을 포함하도록 에이전트 네트워크를 구성.
        최종 입력: (원 입력 + HGCN 출력)
        """
        self.agent = agent_REGISTRY[self.args.agent](input_shape + self.args.hgcn_out_dim, self.args)

    def _build_inputs(self, batch, t):
        """
        t 시점의 에이전트별 입력 구성.
        반환: (B*N, D_in)
        """
        bs = batch.batch_size
        inputs = []

        # 관측값 (B, N, O)
        inputs.append(batch["obs"][:, t])

        # 직전 행동(one-hot) 포함 옵션
        if self.args.obs_last_action:
            if t == 0:
                inputs.append(th.zeros_like(batch["actions_onehot"][:, t]))
            else:
                inputs.append(batch["actions_onehot"][:, t - 1])

        # 에이전트 ID(one-hot) 포함 옵션
        if self.args.obs_agent_id:
            inputs.append(th.eye(self.n_agents, device=batch.device).unsqueeze(0).expand(bs, -1, -1))

        # (B, N, ·) -> (B*N, ·)
        inputs = th.cat([x.reshape(bs * self.n_agents, -1) for x in inputs], dim=1)
        return inputs

    def _get_input_shape(self, scheme):
        """
        에이전트 입력 차원 계산(옵션 포함)
        """
        input_shape = scheme["obs"]["vshape"]
        if self.args.obs_last_action:
            input_shape += scheme["actions_onehot"]["vshape"][0]
        if self.args.obs_agent_id:
            input_shape += self.n_agents
        return input_shape
