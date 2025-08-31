import json
import matplotlib.pyplot as plt
import os
import sys

def plot_mean_return(result_dir):
    """
    지정된 결과 폴더에서 'metrics.json'을 찾아
    평균 리턴(mean return) 값을 시각화합니다.

    Args:
        result_dir (str): 'metrics.json' 파일이 있는 결과 폴더의 경로.
    """
    # metrics.json 파일 경로 구성
    metrics_path = os.path.join(result_dir, 'metrics.json')

    # 파일 존재 여부 확인
    if not os.path.exists(metrics_path):
        print(f"오류: 지정된 경로에 'metrics.json' 파일이 없습니다. -> {metrics_path}")
        return

    # metrics.json 로드
    with open(metrics_path, "r") as f:
        metrics = json.load(f)

    # 필요한 데이터 추출
    try:
        steps = metrics["return_mean"]["steps"]
        returns = metrics["return_mean"]["values"]
    except KeyError:
        print("오류: 'metrics.json' 파일에 'return_mean' 데이터가 없습니다.")
        return

    # 시각화
    plt.figure(figsize=(8, 5))
    plt.plot(steps, returns, marker="o")
    plt.xlabel("Timesteps")
    plt.ylabel("Mean Return")
    plt.title(f"Training Return for {os.path.basename(result_dir)}")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# 스크립트 실행
if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("사용법: python plot_metrics.py <결과_폴더_경로>")
        print("예시: python plot_metrics.py 'results/sacred/hygma/lbforaging-Foraging-5x5-2p-2f-v3/1'")
    else:
        # 첫 번째 매개변수를 결과 폴더 경로로 사용
        target_path = sys.argv[1]
        plot_mean_return(target_path)
