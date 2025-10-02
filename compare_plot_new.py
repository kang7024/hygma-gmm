import json
import matplotlib.pyplot as plt

# 파일 경로 지정
hygma_metrics_path = "results/sacred/hygma/lbforaging-Foraging-7x7-4p-4f-v3/2/metrics.json"
gmm_hygma_metrics_path = "results/sacred/hygma/lbforaging-Foraging-7x7-4p-4f-v3/15/metrics.json"

def load_returns(filepath, max_timestep=None):
    with open(filepath, "r") as f:
        data = json.load(f)
    steps = data["return_mean"]["steps"]
    returns = data["return_mean"]["values"]

    # 타임스텝 잘라내기
    if max_timestep is not None:
        steps, returns = zip(*[(s, r) for s, r in zip(steps, returns) if s <= max_timestep])

    return list(steps), list(returns)

# 비교할 최대 timestep 지정 (예: 100000)
max_timestep = 10000000

# 데이터 로드
hygma_steps, hygma_returns = load_returns(hygma_metrics_path, max_timestep=max_timestep)
gmm_hygma_steps, gmm_hygma_returns = load_returns(gmm_hygma_metrics_path, max_timestep=max_timestep)

# 그래프 그리기
plt.figure(figsize=(10, 6))
plt.plot(hygma_steps, hygma_returns, marker='o', label='HYGMA')
plt.plot(gmm_hygma_steps, gmm_hygma_returns, marker='o', label='GMM-HYGMA')
plt.xlabel('Timesteps')
plt.ylabel('Mean Return')
plt.title(f'Origin-HYGMA vs GMM-HYGMA on LBF-v3 (<= {max_timestep} timesteps)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
