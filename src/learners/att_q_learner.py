# src/learners/att_q_learner.py

import copy
from components.episode_buffer import EpisodeBatch
from modules.mixers.vdn import VDNMixer
from modules.mixers.qmix import QMixer
import torch as th
from torch.optim import RMSprop

# rect 기반 믹서 (별도 파일: modules/mixers/qmix_rect.py 에서 제공)
try:
    from modules.mixers.qmix_rect import QMixerRect
except Exception:
    QMixerRect = None


class AttQLearner:
    """
    AERIAL(rect) 경로를 지원하는 Learner.
    - 기존 qmix/vdn은 그대로 state 기반
    - args.mixer == "qmix_rect" 일 때 rect 기반 믹싱
      (rect는 mac.get_last_transformer_out() -> batch["rect"] -> batch["state"] 순으로 선택)
    """

    def __init__(self, mac, scheme, logger, args):
        self.args = args
        self.mac = mac
        self.logger = logger

        self.params = list(mac.parameters())
        self.last_target_update_episode = 0

        self.mixer = None
        if args.mixer is not None:
            if args.mixer == "vdn":
                self.mixer = VDNMixer()
            elif args.mixer == "qmix":
                self.mixer = QMixer(args)
            elif args.mixer == "qmix_rect":
                assert QMixerRect is not None
                rect_dim = getattr(args, "rect_dim", None) or getattr(args, "hidden_state_transformer_dim", None)
                assert rect_dim is not None, "rect_dim 또는 hidden_state_transformer_dim 필요"
                self.mixer = QMixerRect(args, rect_dim=int(rect_dim))
            else:
                raise ValueError(f"Mixer {args.mixer} not recognised.")
            self.params += list(self.mixer.parameters())
            self.target_mixer = copy.deepcopy(self.mixer)

        # 일부 설정이 문자열로 들어오는 환경(예: sacred/yaml 로딩) 대비: 안전 캐스팅
        def _as_float(x):
            return float(x) if isinstance(x, str) else x

        lr = _as_float(args.lr)
        alpha = _as_float(getattr(args, "optim_alpha", 0.99))
        eps = _as_float(getattr(args, "optim_eps", 1e-8))
        self.optimiser = RMSprop(params=self.params,
                                 lr=lr,
                                 alpha=alpha,
                                 eps=eps)

        # target MAC
        self.target_mac = copy.deepcopy(mac)

        self.log_stats_t = -self.args.learner_log_interval - 1

    # ---- 내부 유틸: MAC 시퀀스 수집(+rect 시퀀스) ----
    def _collect_mac_seq(self, mac, batch, t_env, test_mode=False):
        mac_out = []
        rect_seq = []
        use_rect_runtime = hasattr(mac, "get_last_transformer_out")
        mac.init_hidden(batch.batch_size)

        for t in range(batch.max_seq_length):
            out = mac.forward(batch, t=t, t_env=t_env, test_mode=test_mode)
            mac_out.append(out)

            if use_rect_runtime:
                # 컨트롤러가 rect를 바로 뱉는 경우
                r = mac.get_last_transformer_out()
                if r is None:
                    # rect를 실시간으로 사용하려는 설정인데 rect가 없다면 에러
                    raise RuntimeError("Hidden transformer enabled but no rect output was produced.")
                rect_seq.append(r)

        mac_out = th.stack(mac_out, dim=1)  # [B, T, N, A] (또는 Q)
        rect_seq = th.stack(rect_seq, dim=1) if len(rect_seq) > 0 else None  # [B, T, R]
        return mac_out, rect_seq

    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int):
        # 기본 텐서들
        rewards = batch["reward"][:, :-1]
        actions = batch["actions"][:, :-1]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        avail_actions = batch["avail_actions"]

        # Q 추정
        mac_out, mac_rect_out = self._collect_mac_seq(self.mac, batch, t_env)
        # 선택한 액션의 Q
        chosen_action_qvals = th.gather(mac_out[:, :-1], dim=3, index=actions).squeeze(3)  # [B, T-1, N]

        # 타깃 Q
        target_mac_out, target_rect_out = self._collect_mac_seq(self.target_mac, batch)
        target_mac_out = target_mac_out[:, 1:]  # [B, T-1, N, A]

        # 사용 불가 액션 마스킹
        target_mac_out[avail_actions[:, 1:] == 0] = -9999999

        # Double-Q
        if getattr(self.args, "double_q", False):
            mac_out_detach = mac_out.clone().detach()
            mac_out_detach[avail_actions == 0] = -9999999
            cur_max_actions = mac_out_detach[:, 1:].max(dim=3, keepdim=True)[1]
            target_max_qvals = th.gather(target_mac_out, 3, cur_max_actions).squeeze(3)  # [B, T-1, N]
        else:
            target_max_qvals = target_mac_out.max(dim=3)[0]  # [B, T-1, N]

        # ---- Mix 단계: mixer 종류에 따라 state/rect 선택 ----
        if self.mixer is not None:
            # qmix_rect면 rect 우선, 아니면 state
            if getattr(self.args, "mixer", None) == "qmix_rect":
                # 1) MAC가 즉시 내놓은 rect (우선)
                if mac_rect_out is not None:
                    mixing_states = mac_rect_out[:, :-1]                 # [B, T-1, R]
                    target_mixing_states = target_rect_out[:, 1:] if target_rect_out is not None else None
                    if target_mixing_states is None:
                        raise RuntimeError("Target MAC did not produce rect output for qmix_rect.")
                # 2) 배치에 rect 저장된 경우
                elif "rect" in batch.scheme:
                    mixing_states = batch["rect"][:, :-1]               # [B, T-1, R]
                    target_mixing_states = batch["rect"][:, 1:]         # [B, T-1, R]
                # 3) 최후: state로 폴백 (호환)
                else:
                    mixing_states = batch["state"][:, :-1]
                    target_mixing_states = batch["state"][:, 1:]

                # 모양 보정(필요 시)
                if mixing_states.dim() == 2:
                    mixing_states = mixing_states.unsqueeze(1)          # [B, 1, R]
                if target_mixing_states.dim() == 2:
                    target_mixing_states = target_mixing_states.unsqueeze(1)

                chosen_action_qvals = self.mixer(chosen_action_qvals, mixing_states)
                target_max_qvals   = self.target_mixer(target_max_qvals, target_mixing_states)

            else:
                # 기존 qmix/vdn: state 기반
                chosen_action_qvals = self.mixer(chosen_action_qvals, batch["state"][:, :-1])
                target_max_qvals   = self.target_mixer(target_max_qvals, batch["state"][:, 1:])

        # ---- N-step / 1-step 타깃 ----
        N = getattr(self.args, "n_step", 1)
        if N == 1:
            targets = rewards + self.args.gamma * (1 - terminated) * target_max_qvals
        else:
            # N-step targets (원본 q_learner의 로직 유지)
            n_rewards = th.zeros_like(rewards)
            gamma_tensor = th.tensor([self.args.gamma**i for i in range(N)],
                                     dtype=th.float, device=n_rewards.device)
            steps = mask.flip(1).cumsum(dim=1).flip(1).clamp_max(N).long()
            for i in range(batch.max_seq_length - 1):
                n_rewards[:, i, 0] = ((rewards * mask)[:, i:i + N, 0] *
                                      gamma_tensor[:(batch.max_seq_length - 1 - i)]).sum(dim=1)
            indices = th.linspace(0, batch.max_seq_length - 2,
                                  steps=batch.max_seq_length - 1,
                                  device=steps.device).unsqueeze(1).long()
            n_targets_terminated = th.gather(target_max_qvals * (1 - terminated),
                                             dim=1, index=steps.long() + indices - 1)
            targets = n_rewards + th.pow(self.args.gamma, steps.float()) * n_targets_terminated

        # TD error & 손실
        td_error = (chosen_action_qvals - targets.detach())
        mask = mask.expand_as(td_error)
        masked_td_error = td_error * mask
        loss = (masked_td_error ** 2).sum() / mask.sum()

        # Optimize
        self.optimiser.zero_grad()
        loss.backward()
        grad_norm = th.nn.utils.clip_grad_norm_(self.params, self.args.grad_norm_clip)
        self.optimiser.step()

        # 타깃 업데이트(episode 기반)
        if (episode_num - self.last_target_update_episode) / self.args.target_update_interval >= 1.0:
            self._update_targets()
            self.last_target_update_episode = episode_num

        # 로그
        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            self.logger.log_stat("loss", loss.item(), t_env)
            self.logger.log_stat("grad_norm", grad_norm, t_env)
            mask_elems = mask.sum().item()
            self.logger.log_stat("td_error_abs", (masked_td_error.abs().sum().item()/mask_elems), t_env)
            self.logger.log_stat("q_taken_mean",
                                 (chosen_action_qvals * mask).sum().item()/(mask_elems * self.args.n_agents), t_env)
            self.logger.log_stat("target_mean",
                                 (targets * mask).sum().item()/(mask_elems * self.args.n_agents), t_env)
            agent_utils = (th.gather(mac_out[:, :-1], dim=3, index=actions).squeeze(3) * mask).sum().item() / (mask_elems * self.args.n_agents)
            self.logger.log_stat("agent_utils", agent_utils, t_env)
            self.log_stats_t = t_env

    def _update_targets(self):
        self.target_mac.load_state(self.mac)
        if self.mixer is not None:
            self.target_mixer.load_state_dict(self.mixer.state_dict())
        self.logger.console_logger.info("Updated target network")

    def cuda(self):
        self.mac.cuda()
        self.target_mac.cuda()
        if self.mixer is not None:
            self.mixer.cuda()
            self.target_mixer.cuda()

    def save_models(self, path):
        self.mac.save_models(path)
        if self.mixer is not None:
            th.save(self.mixer.state_dict(), f"{path}/mixer.th")
        th.save(self.optimiser.state_dict(), f"{path}/opt.th")

    def load_models(self, path):
        self.mac.load_models(path)
        self.target_mac.load_models(path)
        if self.mixer is not None:
            self.mixer.load_state_dict(th.load(f"{path}/mixer.th", map_location=lambda s, l: s))
        self.optimiser.load_state_dict(th.load(f"{path}/opt.th", map_location=lambda s, l: s))
