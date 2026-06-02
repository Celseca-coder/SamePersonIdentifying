import subprocess
import time
import os

PYTHON_EXEC = "python"             # 你的 Python 命令 (如 python3 或特定虚拟环境下的 python)
EVAL_SCRIPT = "run_evaluation.py"  # 核心评测脚本名称
DEFAULT_TRIALS = 50                # 默认的评测次数 (建议50，兼顾统计显著性与时间)

DATA_DIR = "/home/user/GSK/heyalan/Reid/data/multi_dataset/PKUSketchRE-ID_V1"  # 例: "/data/Market-1501-v15.09.15"
MODEL_NAME = "/data/llm/AI-ModelScope/R-4B" # 例: "/data/llm/AI-ModelScope/R-4B"
BASE_URL = "http://localhost:8000/v1"

RUN_PHASE_1_LIST = True         # 阶段一：List-wise 基线扫描 
RUN_PHASE_2_TOUR = True    # 阶段二 A：Tournament 锦标赛扫描 
RUN_PHASE_2_PAIR = True    # 阶段二 B：Pair-wise 线性扫描 

def run_experiment(agent, query_size, gallery_size, trials):
    """
    执行单组实验的包装函数
    """
    print(f"\n" + "="*60)
    print(f"🚀 正在启动实验: Agent={agent} | Query={query_size} | Gallery={gallery_size} | Trials={trials}")
    print(f"="*60)
    
    cmd = [
        PYTHON_EXEC, EVAL_SCRIPT,
        "--agent", agent,
        "--query_size", str(query_size),
        "--gallery_size", str(gallery_size),
        "--trials", str(trials)
    ]
    
    # 如果指定了数据集或模型路径，则追加参数
    if DATA_DIR:
        cmd.extend(["--data_dir", DATA_DIR])
    if MODEL_NAME:
        cmd.extend(["--model", MODEL_NAME])
    if BASE_URL:                               # 👈 新增这一段！
        cmd.extend(["--base_url", BASE_URL])

    start_time = time.time()
    try:
        # 启动子进程执行实验，实时输出日志到控制台
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ 实验异常中断 (Agent: {agent}, Q: {query_size}, G: {gallery_size})。错误码: {e.returncode}")
    except KeyboardInterrupt:
        print("\n🛑 实验被手动终止！")
        exit(0)
        
    elapsed = time.time() - start_time
    print(f"✅ 本组实验完成，耗时: {elapsed/60:.2f} 分钟\n")


def main():
    print("🌟 自动实验网格 (Grid Search) 启动！🌟")
    total_tasks = 0

    if RUN_PHASE_1_LIST:
        agent = "listwise"
        query_sizes = [1]  
        gallery_sizes = [5, 10, 15, 20, 30, 50, 100]
        for q in query_sizes:
            for g in gallery_sizes:
                run_experiment(agent, q, g, DEFAULT_TRIALS)
                total_tasks += 1

    if RUN_PHASE_2_TOUR:
        agent = "tournament"
        query_sizes = [1]
        gallery_sizes = [5, 10, 15, 20, 30, 50, 100]
        for q in query_sizes:
            for g in gallery_sizes:
                run_experiment(agent, q, g, DEFAULT_TRIALS)
                total_tasks += 1

    if RUN_PHASE_2_PAIR:
        agent = "pairwise"
        query_sizes = [1]
        gallery_sizes = [5, 10, 15, 20, 30, 50, 100]
        for q in query_sizes:
            for g in gallery_sizes:
                run_experiment(agent, q, g, DEFAULT_TRIALS)
                total_tasks += 1

    print("🎉 所有网格测试任务均已执行完毕！")
    print(f"统计数据已自动追加到 experiment_summary.csv 中，共执行了 {total_tasks} 组参数设定。")

if __name__ == "__main__":
    main()