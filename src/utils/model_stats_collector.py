import numpy as np
import torch as th
import os


class ModelStatsCollector:
    def __init__(self):
        self.model_stats = {
            'total_params': 0,
            'trainable_params': 0,
            'module_params': {},  # 每个模块的参数量
            'architecture': {}  # 架构信息
        }

    def collect_model_stats(self, mac, mixer=None):
        """收集模型统计信息"""
        stats = {}

        # 统计HGCN参数
        if hasattr(mac, 'hgcn'):
            hgcn_params = sum(p.numel() for p in mac.hgcn.parameters())
            stats['hgcn'] = hgcn_params

        # 统计智能体网络参数 (排除HGCN部分)
        agent_params = sum(p.numel() for n, p in mac.named_parameters()
                           if not n.startswith('hgcn'))
        stats['agent'] = agent_params

        # 统计混合网络参数
        if mixer is not None:
            mixer_params = sum(p.numel() for p in mixer.parameters())
            stats['mixer'] = mixer_params

        return stats

    def collect_complete_stats(self, mac, mixer=None):
        """收集并汇总所有组件的参数信息，避免重复计算"""
        # 初始化参数计数器
        total_params = 0
        trainable_params = 0

        # 1. 收集HGCN部分的统计信息
        hgcn_stats = {'total_params': 0, 'trainable_params': 0}
        if hasattr(mac, 'hgcn'):
            hgcn_stats = self.collect_model_stats(mac.hgcn)
            total_params += hgcn_stats['total_params']
            trainable_params += hgcn_stats['trainable_params']

        # 2. 收集智能体网络的统计信息
        agent_stats = self.collect_model_stats(mac.agent)
        total_params += agent_stats['total_params']
        trainable_params += agent_stats['trainable_params']

        # 3. 对于MAC，我们不直接使用其总参数统计，因为它包含了HGCN和agent
        # 相反，我们记录MAC的原始统计以供参考
        mac_stats = self.collect_model_stats(mac)

        # 4. 收集mixer的统计信息
        mixer_stats = {'total_params': 0, 'trainable_params': 0}
        if mixer is not None:
            mixer_stats = self.collect_model_stats(mixer)
            total_params += mixer_stats['total_params']
            trainable_params += mixer_stats['trainable_params']

        return {
            'total_params': total_params,
            'trainable_params': trainable_params,
            'mac_stats': mac_stats,  # 这里包含了HGCN和agent的参数
            'hgcn_stats': hgcn_stats,
            'agent_stats': agent_stats,
            'mixer_stats': mixer_stats
        }

    def format_complete_stats(self, stats):
        """格式化完整的模型统计信息为可读形式"""
        lines = []
        lines.append("完整模型参数统计:")
        lines.append(f"总参数量: {stats['total_params']:,}")
        lines.append(f"可训练参数量: {stats['trainable_params']:,}")

        # 各组件参数统计
        lines.append("\n各组件参数量:")

        # HGCN部分统计
        lines.append(f"HGCN网络: {stats['hgcn_stats']['total_params']:,} 参数")

        # 智能体网络统计
        lines.append(f"智能体网络: {stats['agent_stats']['total_params']:,} 参数")

        # 混合网络统计
        lines.append(f"混合网络: {stats['mixer_stats']['total_params']:,} 参数")

        # 重新计算的总参数量
        computed_total = (stats['hgcn_stats']['total_params'] +
                          stats['agent_stats']['total_params'] +
                          stats['mixer_stats']['total_params'])
        lines.append(f"汇总参数量: {computed_total:,} 参数")

        # MAC总体统计（仅作参考）
        lines.append("\n注意：以下MAC总体参数包含了HGCN和智能体网络的参数，仅作参考，不计入总参数量:")
        lines.append(f"MAC总体参数: {stats['mac_stats']['total_params']:,}")

        return "\n".join(lines)

    def format_model_stats(self, stats):
        """格式化模型统计信息为可读形式"""
        lines = []
        lines.append("Model Statistics:")
        lines.append(f"Total Parameters: {stats['total_params']:,}")
        lines.append(f"Trainable Parameters: {stats['trainable_params']:,}")
        lines.append("\nParameters by Module:")

        for module_name, module_stats in stats['module_params'].items():
            lines.append(f"\n{module_name}:")
            lines.append(f"  Total: {module_stats['total']:,}")
            lines.append(f"  Trainable: {module_stats['trainable']:,}")
            shape_info = []
            for shape, count in module_stats['shape']:
                shape_info.append(f"{shape} ({count})")
            lines.append(f"  Shape: {shape_info}")

        return "\n".join(lines)

    def measure_computational_overhead(self, start_time, end_time, clustering_time=None, hgcn_time=None):
        """测量和记录计算开销"""
        total_time = end_time - start_time
        result = {
            'total_time': total_time
        }

        if clustering_time is not None:
            result['clustering_time'] = clustering_time
            result['clustering_percentage'] = (clustering_time / total_time) * 100 if total_time > 0 else 0

        if hgcn_time is not None:
            result['hgcn_time'] = hgcn_time
            result['hgcn_percentage'] = (hgcn_time / total_time) * 100 if total_time > 0 else 0

        return result