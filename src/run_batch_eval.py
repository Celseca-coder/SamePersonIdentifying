#!/usr/bin/env python3
"""
run_batch_eval.py — 批量评测脚本
对 N 个不同的行人 ID 各跑一次 LangChain CLIP 向量检索，汇总统计 Rank-1/Rank-3 准确率。

每次 trial 使用独立的临时 Chroma 数据库，避免跨 trial 数据污染。
用法：
    python run_batch_eval.py                    # 默认 50 trials，读 config.yaml
    python run_batch_eval.py --trials 20        # 自定义 trial 数量
    python run_batch_eval.py --report out.md    # 指定报告输出路径
"""

import os
import sys
import yaml
import random
import shutil
import tempfile
import argparse
import time
from collections import defaultdict

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from reid_agent import LangChainBatchImageRetrievalManager
from data_loader import Market1501Dataset
from memory_plugin import MemoryPlugin


# ---------------------------------------------------------------------------
# 核心工具函数
# ---------------------------------------------------------------------------

def get_all_pids(dataset):
    """从 query 集合中整理出所有有效的 person ID 及其图片路径。"""
    pid_to_queries = defaultdict(list)
    for p in dataset.query_paths:
        pid, _ = dataset._parse_filename(p)
        if pid != -1:
            pid_to_queries[pid].append(p)
    return pid_to_queries


def build_test_case_for_pid(dataset, pid, gallery_size, query_size, pid_to_queries):
    """
    为指定 person_pid 构造一次测试用例：
    - query_paths: 该人的 query_size 张查询图
    - gallery_paths: 1 张正样本 + (gallery_size-1) 张负样本，顺序随机
    - ground_truth_idx: 正样本在 gallery 中的索引
    返回 None 表示该 PID 在 gallery 中无正样本（跳过）。
    """
    # Query 采样
    pid_queries = pid_to_queries.get(pid, [])
    actual_query_size = min(query_size, len(pid_queries))
    query_subset = random.sample(pid_queries, actual_query_size)

    # Gallery 采样
    positives = [p for p in dataset.gallery_paths if dataset._parse_filename(p)[0] == pid]
    negatives = [p for p in dataset.gallery_paths if dataset._parse_filename(p)[0] != pid]

    if not positives:
        return None, None, None

    target_positive = random.choice(positives)
    num_negatives = min(len(negatives), gallery_size - 1)
    curr_negatives = random.sample(negatives, num_negatives)

    gallery_subset = [target_positive] + curr_negatives
    random.shuffle(gallery_subset)
    gt_idx = gallery_subset.index(target_positive)

    return query_subset, gallery_subset, gt_idx


def evaluate_results(results, dataset):
    """统计一组检索结果的 Rank-1 / Rank-3 命中数。"""
    correct_rank1 = 0
    correct_rank3 = 0
    valid_total = 0

    for result in results:
        if "error" in result:
            continue
        valid_total += 1
        query_pid, _ = dataset._parse_filename(result["query"])
        matches = result.get("top_k_matches", [])

        for rank, match in enumerate(matches):
            match_pid, _ = dataset._parse_filename(match)
            if match_pid == query_pid:
                if rank == 0:
                    correct_rank1 += 1
                correct_rank3 += 1
                break  # 找到最高命中即可

    return correct_rank1, correct_rank3, valid_total


def run_single_trial(dataset, pid, gallery_size, query_size, pid_to_queries, trial_idx):
    """
    对单个 person_pid 跑完整的一次检索评测。
    每次调用都创建独立的临时 Chroma 数据库，结束后自动清理。
    返回 dict 或 None（失败时）。
    """
    query_paths, gallery_paths, _ = build_test_case_for_pid(
        dataset, pid, gallery_size, query_size, pid_to_queries
    )
    if query_paths is None:
        return None

    # 独立的临时 DB，防止数据跨 trial 污染
    tmp_db = tempfile.mkdtemp(prefix=f"reid_trial_{trial_idx:03d}_")
    try:
        manager = LangChainBatchImageRetrievalManager(persist_directory=tmp_db)
        if not manager.vectorstore:
            return None

        manager.build_gallery_batch(gallery_paths, batch_size=32)
        results = manager.automate_batch_query(query_paths, k=3)

        rank1, rank3, valid = evaluate_results(results, dataset)
        n = valid if valid > 0 else 1  # 防止除零

        return {
            "trial": trial_idx,
            "pid": pid,
            "total_queries": valid,
            "rank1_hits": rank1,
            "rank3_hits": rank3,
            "rank1_acc": rank1 / n,
            "rank3_acc": rank3 / n,
        }
    finally:
        shutil.rmtree(tmp_db, ignore_errors=True)


# ---------------------------------------------------------------------------
# 报告生成
# ---------------------------------------------------------------------------

def generate_report(trial_results, config_snapshot, elapsed, output_path):
    total = len(trial_results)
    if total == 0:
        return

    avg_r1 = sum(r["rank1_acc"] for r in trial_results) / total
    avg_r3 = sum(r["rank3_acc"] for r in trial_results) / total
    r1_hits = sum(r["rank1_hits"] for r in trial_results)
    r3_hits = sum(r["rank3_hits"] for r in trial_results)
    total_queries = sum(r["total_queries"] for r in trial_results)

    lines = []
    lines.append("# Batch ReID Evaluation Report\n")
    lines.append("## Config\n")
    for k, v in config_snapshot.items():
        lines.append(f"- **{k}**: {v}")
    lines.append("")
    lines.append("## Summary\n")
    lines.append(f"| Metric | Value |")
    lines.append(f"|---|---|")
    lines.append(f"| Total Trials | {total} |")
    lines.append(f"| Total Queries | {total_queries} |")
    lines.append(f"| Avg Rank-1 Accuracy | **{avg_r1:.2%}** |")
    lines.append(f"| Avg Rank-3 Accuracy | **{avg_r3:.2%}** |")
    lines.append(f"| Overall Rank-1 Hits | {r1_hits} / {total_queries} |")
    lines.append(f"| Overall Rank-3 Hits | {r3_hits} / {total_queries} |")
    lines.append(f"| Elapsed Time | {elapsed:.1f}s |")
    lines.append("")
    lines.append("## Per-Trial Results\n")
    lines.append("| Trial | Person ID | Queries | Rank-1 | Rank-3 | R1 Acc | R3 Acc |")
    lines.append("|---|---|---|---|---|---|---|")
    for r in trial_results:
        lines.append(
            f"| {r['trial']} | {r['pid']} | {r['total_queries']} "
            f"| {r['rank1_hits']} | {r['rank3_hits']} "
            f"| {r['rank1_acc']:.0%} | {r['rank3_acc']:.0%} |"
        )

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print(f"\nReport saved → {output_path}")


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Batch ReID Evaluation (LangChain CLIP)")
    parser.add_argument("--config", type=str, default="/mnt/l/CV/config.yaml")
    parser.add_argument("--trials", type=int, default=None,
                        help="Number of unique person IDs to evaluate (default: 50)")
    parser.add_argument("--report", type=str, default="/mnt/l/CV/batch_eval_report.md")
    args = parser.parse_args()

    # 读取 config
    config = {}
    if os.path.exists(args.config):
        with open(args.config, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}

    data_dir     = config.get("data_dir", "/mnt/l/CV/data/1/Market-1501-v15.09.15")
    gallery_size = config.get("gallery_size", 100)
    query_size   = config.get("query_size", 5)
    num_trials   = args.trials or config.get("batch_trials", 15)

    # 热插拔记忆模块：config 中 use_memory: false 时全部 no-op
    mem = MemoryPlugin.from_config(config)

    config_snapshot = {
        "data_dir": data_dir,
        "gallery_size": gallery_size,
        "query_size": query_size,
        "num_trials": num_trials,
        "retrieval_mode": "LangChain CLIP Vector DB",
    }

    print("=" * 50)
    print("🚀 Batch ReID Evaluation")
    print(f"   Trials      : {num_trials}")
    print(f"   Gallery size: {gallery_size}")
    print(f"   Query size  : {query_size}")
    print("=" * 50)

    # 加载数据集，整理 PID 映射
    dataset = Market1501Dataset(data_dir)
    pid_to_queries = get_all_pids(dataset)

    # 过滤出 query 中有足够图片的 PID，并且 gallery 里也存在正样本
    gallery_pids = {dataset._parse_filename(p)[0] for p in dataset.gallery_paths}
    valid_pids = [
        pid for pid, paths in pid_to_queries.items()
        if len(paths) >= 1 and pid in gallery_pids and pid != -1
    ]

    if len(valid_pids) < num_trials:
        print(f"Warning: Only {len(valid_pids)} valid PIDs found, "
              f"reducing trials to {len(valid_pids)}.")
        num_trials = len(valid_pids)

    # 无放回采样 num_trials 个不同 PID，保证每次评测的是不同的人
    selected_pids = random.sample(valid_pids, num_trials)

    # 逐 trial 执行
    trial_results = []
    skipped = 0
    t_start = time.time()

    for i, pid in enumerate(selected_pids, start=1):
        print(f"\n[Trial {i:>3}/{num_trials}] Person ID: {pid:>4}", end="  ", flush=True)
        result = run_single_trial(dataset, pid, gallery_size, query_size, pid_to_queries, i)

        if result is None:
            print("⚠ Skipped (no gallery match)")
            skipped += 1
            continue

        trial_results.append(result)
        r1 = f"{result['rank1_acc']:.0%}"
        r3 = f"{result['rank3_acc']:.0%}"
        hits = f"R1={result['rank1_hits']}/{result['total_queries']}"
        print(f"Rank-1: {r1:>4}  Rank-3: {r3:>4}  ({hits})")

        # 记录每次 trial 到短时记忆
        mem.add_trial(
            prompt="LangChain CLIP",
            is_correct=result["rank1_hits"] > 0,
            analysis=f"PID={pid}, R1={result['rank1_hits']}/{result['total_queries']}, R3={result['rank3_hits']}/{result['total_queries']}"
        )

    elapsed = time.time() - t_start

    # 最终汇总
    completed = len(trial_results)
    if completed == 0:
        print("\nNo trials completed.")
        return

    avg_r1 = sum(r["rank1_acc"] for r in trial_results) / completed
    avg_r3 = sum(r["rank3_acc"] for r in trial_results) / completed

    print("\n" + "=" * 50)
    print(f"✅ Batch Evaluation Complete")
    print(f"   Completed : {completed} trials  (skipped: {skipped})")
    print(f"   Avg Rank-1: {avg_r1:.2%}")
    print(f"   Avg Rank-3: {avg_r3:.2%}")
    print(f"   Time      : {elapsed:.1f}s  (~{elapsed/completed:.1f}s/trial)")
    print("=" * 50)

    # session 结束后将汇总写入长时记忆
    mem.summarize_to_long_term(
        f"Batch [LangChain CLIP, {completed} trials] Avg R1={avg_r1:.2%} R3={avg_r3:.2%}"
    )

    generate_report(trial_results, config_snapshot, elapsed, args.report)


if __name__ == "__main__":
    main()
