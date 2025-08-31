import copy
from collections import defaultdict
import numpy as np
import torch as th
import os
import h5py


class AttentionDataCollector:
    def __init__(self, save_path="attention_data"):
        self.save_path = save_path
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        self.reset_data()

    def reset_data(self):
        """Reset all collected data to initial empty state."""
        self.attention_data = {
            'timesteps': [],
            'layer_data': defaultdict(list),
            'group_info': [],
            'stats': defaultdict(list)
        }

    def update_data(self, attention_weights, t_env, layer_idx, agent_groups):
        if attention_weights is None:
            return False

        try:
            att_weights = attention_weights.detach().cpu().numpy()
            self.attention_data['timesteps'].append(t_env)
            self.attention_data['layer_data'][layer_idx].append(att_weights)
            self.attention_data['group_info'].append(copy.deepcopy(agent_groups))
            return True
        except Exception as e:
            print(f"[Collector] Error updating data: {str(e)}")
            return False

    def save_data(self, filename, extra_data=None):
        filepath = os.path.join(self.save_path, filename)
        try:
            with h5py.File(filepath, 'w') as f:
                # 保存时间步信息
                f.create_dataset('timesteps', data=np.array(self.attention_data['timesteps']))

                # 保存每层的注意力数据
                layers_group = f.create_group('layers')
                for layer_idx, layer_data in self.attention_data['layer_data'].items():
                    layers_group.create_dataset(
                        f'layer_{layer_idx}',
                        data=np.array(layer_data)
                    )

                # 保存分组信息
                group_info_data = []
                for groups in self.attention_data['group_info']:
                    group_str = str(groups)
                    group_info_data.append(group_str.encode('utf-8'))

                dt = h5py.special_dtype(vlen=str)
                f.create_dataset('group_info', data=group_info_data, dtype=dt)

                # 保存统计信息
                stats_group = f.create_group('stats')
                for stat_name, stat_data in self.attention_data['stats'].items():
                    stats_group.create_dataset(stat_name, data=np.array(stat_data))

                # 保存额外模型统计信息
                if extra_data and 'model_stats' in extra_data:
                    model_stats = extra_data['model_stats']
                    stats_group = f.create_group('model_stats')

                    stats_group.attrs['total_params'] = model_stats['total_params']
                    stats_group.attrs['trainable_params'] = model_stats['trainable_params']

                    if model_stats.get('mac_stats'):
                        mac_group = stats_group.create_group('mac')
                        mac_group.attrs['total'] = model_stats['mac_stats']['total_params']
                        mac_group.attrs['trainable'] = model_stats['mac_stats']['trainable_params']

                    if model_stats.get('mixer_stats'):
                        mixer_group = stats_group.create_group('mixer')
                        mixer_group.attrs['total'] = model_stats['mixer_stats']['total_params']
                        mixer_group.attrs['trainable'] = model_stats['mixer_stats']['trainable_params']

                self.reset_data()
                return True
        except Exception as e:
            print(f"Error saving attention data: {e}")
            return False

    def collect_model_stats(self, model):
        # 假设目的是收集模型中的参数数量
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        return {

            'total_params': total_params,
            'trainable_params': trainable_params
        }