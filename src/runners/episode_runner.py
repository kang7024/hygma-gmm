from functools import partial

import numpy as np
import torch as th
import math

from components.episode_buffer import EpisodeBatch
from envs import REGISTRY as env_REGISTRY
from envs import register_smac, register_smacv2


class EpisodeRunner:
    def __init__(self, args, logger):
        self.args = args
        self.logger = logger
        self.batch_size = self.args.batch_size_run
        assert self.batch_size == 1

        # registering both smac and smacv2 causes a pysc2 error
        # --> dynamically register the needed env
        if self.args.env == "sc2":
            register_smac()
        elif self.args.env == "sc2v2":
            register_smacv2()

        self.env = env_REGISTRY[self.args.env](
            **self.args.env_args,
            common_reward=self.args.common_reward,
            reward_scalarisation=self.args.reward_scalarisation,
        )
        self.episode_limit = self.env.episode_limit
        self.t = 0

        self.t_env = 0

        self.train_returns = []
        self.test_returns = []
        self.train_stats = {}
        self.test_stats = {}

        # Log the first run
        self.log_train_stats_t = -1000000

    def setup(self, scheme, groups, preprocess, mac):
        # ▼ AERIAL/rect 활성 시 rect_dim을 환경/스키마/args에서 안전하게 계산
        if getattr(self.args, "use_hidden_state_transformer", False):
            rect_dim = None
    
            # 1) env에서 가져오기
            try:
                env_info = self.env.get_env_info()
                state_shape = env_info.get("state_shape", None)
                if state_shape is not None:
                    if isinstance(state_shape, int):
                        rect_dim = int(state_shape)
                    else:
                        rect_dim = int(math.prod(state_shape))
            except Exception:
                pass
    
            # 2) 스키마에 state가 있으면 거기서 계산
            if rect_dim is None and "state" in scheme:
                vshape = scheme["state"].get("vshape", None)
                if vshape is not None:
                    if isinstance(vshape, int):
                        rect_dim = int(vshape)
                    else:
                        rect_dim = int(math.prod(vshape))
    
            # 3) 마지막 대안: args에 이미 넣어둔 rect_dim 사용
            if rect_dim is None:
                rect_dim = getattr(self.args, "rect_dim", None)
    
            # 4) 그래도 못 찾으면 명확히 에러
            if rect_dim is None:
                raise RuntimeError(
                    "use_hidden_state_transformer=True 인데 rect_dim을 계산할 수 없습니다. "
                    "env.get_env_info()['state_shape']가 없고, scheme['state']['vshape']도 없으며, "
                    "args.rect_dim도 비어있습니다. 셋 중 하나를 제공해 주세요."
                )
    
            # args에 주입 (다른 모듈들이 참조할 수 있도록)
            setattr(self.args, "rect_dim", int(rect_dim))
    
            # 스키마에 rect 등록(없을 때만)
            if "rect" not in scheme:
                scheme["rect"] = {"vshape": (int(rect_dim),), "dtype": th.float32}
            
        self.new_batch = partial(
            EpisodeBatch,
            scheme,
            groups,
            self.batch_size,
            self.episode_limit + 1,
            preprocess=preprocess,
            device=self.args.device,
        )
        self.mac = mac

    def get_env_info(self):
        return self.env.get_env_info()

    def save_replay(self):
        self.env.save_replay()

    def close_env(self):
        self.env.close()

    def reset(self):
        self.batch = self.new_batch()
        self.env.reset()
        self.t = 0

    def run(self, test_mode=False):
        self.reset()

        terminated = False
        if self.args.common_reward:
            episode_return = 0
        else:
            episode_return = np.zeros(self.args.n_agents)
        self.mac.init_hidden(batch_size=self.batch_size)

        while not terminated:
            pre_transition_data = {
                "state": [self.env.get_state()],
                "avail_actions": [self.env.get_avail_actions()],
                "obs": [self.env.get_obs()],
            }

            self.batch.update(pre_transition_data, ts=self.t)

            # Pass the entire batch of experiences up till now to the agents
            # Receive the actions for each agent at this timestep in a batch of size 1
            actions = self.mac.select_actions(
                self.batch, t_ep=self.t, t_env=self.t_env, test_mode=test_mode
            )

            # ▼ rect 가져오기 (B=1 가정)
            rect = None
            if getattr(self.args, "use_hidden_state_transformer", False):
                rect = self.mac.get_last_transformer_out()  # shape: (1, R) or (B, R)

            _, reward, terminated, truncated, env_info = self.env.step(actions[0])
            terminated = terminated or truncated
            if test_mode and self.args.render:
                self.env.render()
            episode_return += reward

            post_transition_data = {
                "actions": actions,
                "terminated": [(terminated != env_info.get("episode_limit", False),)],
            }
            if self.args.common_reward:
                post_transition_data["reward"] = [(reward,)]
            else:
                post_transition_data["reward"] = [tuple(reward)]

            # ▼ rect도 같은 ts에 같이 저장 (filled 타이밍 어긋나지 않게)
            if rect is not None:
                # EpisodeBatch.update는 list → tensor 변환을 스스로 처리
                post_transition_data["rect"] = [rect.squeeze(0)]

            self.batch.update(post_transition_data, ts=self.t)

            self.t += 1

        last_data = {
            "state": [self.env.get_state()],
            "avail_actions": [self.env.get_avail_actions()],
            "obs": [self.env.get_obs()],
        }
        if test_mode and self.args.render:
            print(f"Episode return: {episode_return}")
        self.batch.update(last_data, ts=self.t)

        # Select actions in the last stored state
        actions = self.mac.select_actions(
            self.batch, t_ep=self.t, t_env=self.t_env, test_mode=test_mode
        )
        last_update = {"actions": actions}

         # --- 마지막 rect도 기록 (타깃용 rect[:, 1:] 맞추기) ---
        if getattr(self.args, "use_hidden_state_transformer", False):
            rect_last = self.mac.get_last_transformer_out()
            if rect_last is not None:
                last_update["rect"] = [rect_last.squeeze(0)]
 
        self.batch.update(last_update, ts=self.t)

        cur_stats = self.test_stats if test_mode else self.train_stats
        cur_returns = self.test_returns if test_mode else self.train_returns
        log_prefix = "test_" if test_mode else ""
        cur_stats.update(
            {
                k: cur_stats.get(k, 0) + env_info.get(k, 0)
                for k in set(cur_stats) | set(env_info)
            }
        )
        cur_stats["n_episodes"] = 1 + cur_stats.get("n_episodes", 0)
        cur_stats["ep_length"] = self.t + cur_stats.get("ep_length", 0)

        if not test_mode:
            self.t_env += self.t

        cur_returns.append(episode_return)

        if test_mode and (len(self.test_returns) == self.args.test_nepisode):
            self._log(cur_returns, cur_stats, log_prefix)
        elif self.t_env - self.log_train_stats_t >= self.args.runner_log_interval:
            self._log(cur_returns, cur_stats, log_prefix)
            if hasattr(self.mac.action_selector, "epsilon"):
                self.logger.log_stat(
                    "epsilon", self.mac.action_selector.epsilon, self.t_env
                )
            self.log_train_stats_t = self.t_env

        return self.batch

    def _log(self, returns, stats, prefix):
        if self.args.common_reward:
            self.logger.log_stat(prefix + "return_mean", np.mean(returns), self.t_env)
            self.logger.log_stat(prefix + "return_std", np.std(returns), self.t_env)
        else:
            for i in range(self.args.n_agents):
                self.logger.log_stat(
                    prefix + f"agent_{i}_return_mean",
                    np.array(returns)[:, i].mean(),
                    self.t_env,
                )
                self.logger.log_stat(
                    prefix + f"agent_{i}_return_std",
                    np.array(returns)[:, i].std(),
                    self.t_env,
                )
            total_returns = np.array(returns).sum(axis=-1)
            self.logger.log_stat(
                prefix + "total_return_mean", total_returns.mean(), self.t_env
            )
            self.logger.log_stat(
                prefix + "total_return_std", total_returns.std(), self.t_env
            )
        returns.clear()

        for k, v in stats.items():
            if k != "n_episodes":
                self.logger.log_stat(
                    prefix + k + "_mean", v / stats["n_episodes"], self.t_env
                )
        stats.clear()
