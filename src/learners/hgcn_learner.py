import copy
import json
import time

import torch
from components.episode_buffer import EpisodeBatch
from modules.mixers.qmix import QMixer
import torch as th
from torch.optim import RMSprop
import torch.nn.functional as F
import torch.optim as optim
from utils.attention_collector import AttentionDataCollector
from utils.model_stats_collector import ModelStatsCollector
import os


class HGCNQLearner:
    def __init__(self, mac, scheme, logger, args):
        self.t_env = 0
        self.args = args
        self.mac = mac
        self.logger = logger

        self.params = list(mac.parameters())
        self.last_target_update_episode = 0
        self.last_target_update_t = 0
        self.group_change_history = []
        self.last_save_step = -self.args.save_model_interval

        self.mixer = None
        if args.mixer is not None:
            if args.mixer == "qmix":
                self.mixer = QMixer(args)
            else:
                raise ValueError(f"Mixer {args.mixer} not recognised.")
            self.params += list(self.mixer.parameters())
            self.target_mixer = copy.deepcopy(self.mixer)

        self.model_stats = self.collect_and_log_model_stats()
        self.model_stats_collector = ModelStatsCollector()
        # Collect and record model statistics
        # initial_stats = self.collect_and_log_model_stats()
        # self.logger.log_stat("total_params", initial_stats['total_params'], 0)  # Record initial parameter count
        # self.logger.log_stat("trainable_params", initial_stats['trainable_params'], 0)


        # self.optimiser = RMSprop(params=self.params, lr=args.lr, alpha=args.optim_alpha, eps=args.optim_eps)
        # Use AdamW optimizer
        eps = float(args.optim_eps)
        self.optimiser = optim.AdamW(params=self.params,
                                         lr=args.lr,
                                         betas=(args.optim_beta1, args.optim_beta2),
                                         eps=eps,
                                         weight_decay=args.weight_decay)

        # a little wasteful to deepcopy (e.g. duplicates action selector), but should work for any MAC
        self.target_mac = copy.deepcopy(mac)

        self.log_stats_t = -self.args.learner_log_interval - 1

    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int):
        # Record overall training start time
        start_time = time.time()

        self.t_env = t_env
        # Check and update grouping
        clustering_time = 0
        hgcn_time = 0

        clustering_start = time.time()
        # Check and update grouping
        current_num_groups = len(self.mac.agent_groups)
        if hasattr(self.target_mac, 'hgcn') and hasattr(self.mac, 'hgcn'):
            if current_num_groups != self.target_mac.hgcn.num_groups:
                old_num_groups = self.target_mac.hgcn.num_groups

                # Save current weights
                old_weights = self.target_mac.hgcn.state_dict()

                # Update HGCN structure
                self.target_mac.hgcn.update_groups(current_num_groups)

                # Try loading old weights
                new_weights = self.target_mac.hgcn.state_dict()
                for name, param in old_weights.items():
                    if name in new_weights:
                        if new_weights[name].shape == param.shape:
                            new_weights[name].copy_(param)
                        else:
                            # If shape is different, try partial copy
                            min_shape = tuple(min(s1, s2) for s1, s2 in zip(new_weights[name].shape, param.shape))
                            new_weights[name][tuple(slice(0, s) for s in min_shape)].copy_(
                                param[tuple(slice(0, s) for s in min_shape)]
                            )

                # Load adjusted weights
                self.target_mac.hgcn.load_state_dict(new_weights)

                self.group_change_history.append((t_env, old_num_groups, current_num_groups))
                self.logger.console_logger.info(
                    f"Updated target network HGCN from {old_num_groups} to {current_num_groups} groups at t_env {t_env}")

                # No need to reinitialize hidden states, since this is automatically done at each sequence start

                # If grouping changes, we may need to update the target network more frequently
                if t_env - self.last_target_update_t > self.args.target_update_interval // 2:
                    self._update_targets()
                    self.last_target_update_t = t_env

        clustering_time = time.time() - clustering_start

        # Retrieve relevant Q-values and other variables
        rewards = batch["reward"][:, :-1]
        actions = batch["actions"][:, :-1]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        avail_actions = batch["avail_actions"]

        hgcn_forward_start = time.time()
        # Compute estimated Q-values
        mac_out = []
        self.mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            agent_outs = self.mac.forward(batch, t=t, t_env=self.t_env)
            mac_out.append(agent_outs)
        mac_out = th.stack(mac_out, dim=1)  # Concat over time

        hgcn_time = time.time() - hgcn_forward_start

        # Select Q-values corresponding to each agent's actions
        chosen_action_qvals = th.gather(mac_out[:, :-1], dim=3, index=actions).squeeze(3)

        # Compute target Q-values
        target_mac_out = []
        self.target_mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            target_agent_outs = self.target_mac.forward(batch, t=t, t_env=self.t_env)
            target_mac_out.append(target_agent_outs)

        # We don't need the first timesteps Q-Value estimate for calculating targets
        target_mac_out = th.stack(target_mac_out[1:], dim=1)  # Concat across time

        # Mask out unavailable actions
        target_mac_out[avail_actions[:, 1:] == 0] = -9999999

        # Max over target Q-Values
        if self.args.double_q:
            # Get actions that maximise live Q (for double q-learning)
            mac_out_detach = mac_out.clone().detach()
            mac_out_detach[avail_actions == 0] = -9999999
            cur_max_actions = mac_out_detach[:, 1:].max(dim=3, keepdim=True)[1]
            target_max_qvals = th.gather(target_mac_out, 3, cur_max_actions).squeeze(3)
        else:
            target_max_qvals = target_mac_out.max(dim=3)[0]

        # Mix
        if self.mixer is not None:
            chosen_action_qvals = self.mixer(chosen_action_qvals, batch["state"][:, :-1])
            target_max_qvals = self.target_mixer(target_max_qvals, batch["state"][:, 1:])

        # Calculate 1-step Q-Learning targets
        targets = rewards + self.args.gamma * (1 - terminated) * target_max_qvals

        # TD-error
        td_error = (chosen_action_qvals - targets.detach())

        mask = mask.expand_as(td_error)

        # 0-out the targets that came from padded data
        masked_td_error = td_error * mask

        # Normal L2 loss, take mean over actual data
        td_loss = (masked_td_error ** 2).sum() / mask.sum()

        # ------------------- New section: Shared consistency loss and attention regularization loss -------------------
        # Obtain current HGCN feature outputs and compute consistency loss
        hgcn_features = mac_out[:, :-1, :, :]  # Assume HGCN outputs are included in MAC (adjust based on actual implementation)
        consistency_loss = 0.0
        for group in self.mac.agent_groups:
            if len(group) > 1:
                group_features = hgcn_features[:, :, group, :]
                group_mean = group_features.mean(dim=2, keepdim=True)
                consistency_loss += ((group_features - group_mean) ** 2).mean()

        # Attention regularization loss
        # Default attention_regularization set to 0.0
        attention_regularization = th.tensor(0.0, device=batch.device)
        # Get attention weights
        attention_weights = self.mac.get_attention_weights()
        if attention_weights is not None:
            for weights in attention_weights:
                weights = weights + 1e-8
                weights = F.softmax(weights, dim=-1)
                attention_regularization += F.kl_div(weights, th.ones_like(weights) / weights.size(-1),
                                                             reduction='batchmean')


        # Total loss function
        total_loss = td_loss + self.args.lambda_consistency * consistency_loss + self.args.lambda_attention * attention_regularization

        # ------------------------------------------------------------------------

        # Optimization step
        self.optimiser.zero_grad()
        total_loss.backward()
        grad_norm = th.nn.utils.clip_grad_norm_(self.params, self.args.grad_norm_clip)
        self.optimiser.step()

        if (episode_num - self.last_target_update_episode) / self.args.target_update_interval >= 1.0:
            self._update_targets()
            self.last_target_update_episode = episode_num

        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            self.logger.log_stat("loss", total_loss.item(), t_env)
            self.logger.log_stat("td_loss", td_loss.item(), t_env)
            if isinstance(consistency_loss, torch.Tensor):
                consistency_loss_value = consistency_loss.item()
            else:
                consistency_loss_value = consistency_loss
            self.logger.log_stat("consistency_loss", consistency_loss_value, t_env)
            self.logger.log_stat("attention_regularization", attention_regularization.item(), t_env)
            self.logger.log_stat("grad_norm", grad_norm, t_env)
            mask_elems = mask.sum().item()
            self.logger.log_stat("td_error_abs", (masked_td_error.abs().sum().item() / mask_elems), t_env)
            self.logger.log_stat("q_taken_mean", (chosen_action_qvals * mask).sum().item() / (mask_elems * self.args.n_agents), t_env)
            self.logger.log_stat("target_mean", (targets * mask).sum().item() / (mask_elems * self.args.n_agents), t_env)
            self.log_stats_t = t_env

        # Attention regularization loss
        # Default attention_regularization set to 0.0
        if t_env > 0 and abs(t_env % self.args.save_model_interval) <= 10 and (t_env - self.last_save_step >= 100000):
            self.last_save_step = t_env
            self.save_attention_weights(t_env)

        end_time = time.time()
        # Compute and record computational overhead
        overhead_stats = self.model_stats_collector.measure_computational_overhead(
            start_time, end_time, clustering_time, hgcn_time
        )

        # Log computational overhead statistics
        self.logger.log_stat("training_time", overhead_stats['total_time'], t_env)
        self.logger.log_stat("clustering_time", overhead_stats['clustering_time'], t_env)
        self.logger.log_stat("clustering_percentage", overhead_stats['clustering_percentage'], t_env)
        self.logger.log_stat("hgcn_time", overhead_stats['hgcn_time'], t_env)
        self.logger.log_stat("hgcn_percentage", overhead_stats['hgcn_percentage'], t_env)

        # Print computational overhead info periodically
        if t_env - self.log_stats_t + 1 >= self.args.learner_log_interval:
            self.logger.console_logger.info(
                f"Overhead - Total training time: {overhead_stats['total_time']:.4f}s, "
                f"Spectral clustering time: {overhead_stats['clustering_time']:.4f}s ({overhead_stats['clustering_percentage']:.2f}%), "
                f"HGCN computation time: {overhead_stats['hgcn_time']:.4f}s ({overhead_stats['hgcn_percentage']:.2f}%)"
            )

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
            th.save(self.mixer.state_dict(), "{}/mixer.th".format(path))
        th.save(self.optimiser.state_dict(), "{}/opt.th".format(path))

    def load_models(self, path):
        self.mac.load_models(path)
        # Not quite right but I don't want to save target networks
        self.target_mac.load_models(path)
        if self.mixer is not None:
            self.mixer.load_state_dict(th.load("{}/mixer.th".format(path), map_location=lambda storage, loc: storage))
        self.optimiser.load_state_dict(th.load("{}/opt.th".format(path), map_location=lambda storage, loc: storage))

    def collect_and_log_model_stats(self):
        """Collect and record model statistics"""
        # Directly compute parameter counts for each component
        hgcn_params = sum(p.numel() for p in self.mac.hgcn.parameters()) if hasattr(self.mac, 'hgcn') else 0
        agent_params = sum(p.numel() for p in self.mac.agent.parameters())
        mixer_params = sum(p.numel() for p in self.mixer.parameters()) if self.mixer is not None else 0
        total_params = hgcn_params + agent_params + mixer_params

        # Compute trainable parameters
        hgcn_trainable = sum(p.numel() for p in self.mac.hgcn.parameters() if p.requires_grad) if hasattr(self.mac,
                                                                                                          'hgcn') else 0
        agent_trainable = sum(p.numel() for p in self.mac.agent.parameters() if p.requires_grad)
        mixer_trainable = sum(
            p.numel() for p in self.mixer.parameters() if p.requires_grad) if self.mixer is not None else 0
        trainable_params = hgcn_trainable + agent_trainable + mixer_trainable

        # Print parameter statistics
        print("\n===== PARAMETER STATISTICS =====")
        print(f"HGCN parameters: {hgcn_params:,}")
        print(f"Agent parameters: {agent_params:,}")
        print(f"Mixer parameters: {mixer_params:,}")
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        print("===============================\n")

        # Log into the logger (keep this functionality)
        self.logger.log_stat("hgcn_params", hgcn_params, 0)
        self.logger.log_stat("agent_params", agent_params, 0)
        self.logger.log_stat("mixer_params", mixer_params, 0)
        self.logger.log_stat("total_params", total_params, 0)
        self.logger.log_stat("trainable_params", trainable_params, 0)

        # Return statistics dictionary to keep interface consistent
        return {
            'hgcn_params': hgcn_params,
            'agent_params': agent_params,
            'mixer_params': mixer_params,
            'total_params': total_params,
            'trainable_params': trainable_params
        }

    def save_attention_weights(self, t_env):
        attention_weights = self.mac.get_attention_weights()
        if attention_weights is not None:
            weights_to_save = []
            for layer_weights in attention_weights:
                weights_to_save.append(layer_weights.cpu().tolist())

            # Create a new file to save the attention weights at the current timestep
            with open(f"attention_weights_t{t_env}.json", "w") as f:
                json.dump(weights_to_save, f)

        self.logger.console_logger.info(f"Saved attention weights at t_env {t_env}")