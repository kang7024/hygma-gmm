import json
import matplotlib.pyplot as plt

# 파일 경로 지정
qmix_metrics_path = "results/sacred/hygma/lbforaging-Foraging-7x7-4p-4f-v3/1/metrics.json"
hygma_metrics_path = "results/sacred/hygma/lbforaging-Foraging-7x7-4p-4f-v3/5/metrics.json"

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
qmix_steps, qmix_returns = load_returns(qmix_metrics_path, max_timestep=max_timestep)
hygma_steps, hygma_returns = load_returns(hygma_metrics_path, max_timestep=max_timestep)

# 그래프 그리기
plt.figure(figsize=(10, 6))
plt.plot(qmix_steps, qmix_returns, marker='o', label='QMIX')
plt.plot(hygma_steps, hygma_returns, marker='o', label='HYGMA')
plt.xlabel('Timesteps')
plt.ylabel('Mean Return')
plt.title(f'QMIX vs HYGMA on LBF-v3 (<= {max_timestep} timesteps)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
