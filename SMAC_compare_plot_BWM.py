import json
import matplotlib.pyplot as plt

# 파일 경로 지정
#base_metrics_path = "results/sacred/hygma/3s5z_vs_3s6z/2/metrics.json"
#gmm_metrics_path = "results/sacred/gmm_hygma/3s5z_vs_3s6z/16/metrics.json"

base_metrics_path = "results/sacred/hygma/corridor/2/metrics.json"
gmm_metrics_path = "results/sacred/gmm_hygma/corridor/2/metrics.json"

def load_returns(filepath, max_timestep=None):
    with open(filepath, "r") as f:
        data = json.load(f)
    steps = data["battle_won_mean"]["steps"]
    returns = data["battle_won_mean"]["values"]

    # 타임스텝 잘라내기
    if max_timestep is not None:
        steps, returns = zip(*[(s, r) for s, r in zip(steps, returns) if s <= max_timestep])

    return list(steps), list(returns)

# 비교할 최대 timestep 지정 (예: 100000)
max_timestep = 10000000

# 데이터 로드
base_steps, base_returns = load_returns(base_metrics_path, max_timestep=max_timestep)
gmm_steps, gmm_returns = load_returns(gmm_metrics_path, max_timestep=max_timestep)

# 그래프 그리기
plt.figure(figsize=(10, 6))
plt.plot(base_steps, base_returns, marker='o', label='Base(HYGMA)')
plt.plot(gmm_steps, gmm_returns, marker='o', label='Ours(GMM-HYGMA)')
plt.xlabel('Timesteps')
plt.ylabel('Battle Won Mean')
#plt.title(f'Base vs Ours on SMAC 3s5z_vs_3s6z')
plt.title(f'Base vs Ours on SMAC(Corridor MAP)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
