import os
import sys
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

sys.path.append(os.path.abspath("./src"))

from reid_agent import LangChainBatchImageRetrievalManager
from data_loader import Market1501Dataset
from memory_plugin import MemoryPlugin

import yaml

def main():
    config_path = "/mnt/l/CV/config.yaml"
    config = {}
    if os.path.exists(config_path):
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)

    data_dir     = config.get("data_dir", "/mnt/l/CV/data/1/Market-1501-v15.09.15")
    gallery_size = config.get("gallery_size", 100)
    query_size   = config.get("query_size", 1)

    # 热插拔记忆模块
    mem = MemoryPlugin.from_config(config)

    manager = LangChainBatchImageRetrievalManager(persist_directory="./langchain_reid_db")
    if not manager.vectorstore:
        print("请检查并安装依赖：pip install langchain-experimental open_clip_torch chromadb torch")
        return

    dataset = Market1501Dataset(data_dir)
    query_paths, gallery_paths, ground_truth_idx = dataset.get_test_case(
        gallery_size=gallery_size, query_size=query_size
    )

    print(f"正在建立包含 {len(gallery_paths)} 张图片的索引库...")
    manager.build_gallery_batch(gallery_paths, batch_size=32)

    print(f"\n正在检索 Query: {query_paths}")
    results = manager.automate_batch_query(query_paths, k=3)

    correct_rank1 = 0
    correct_rank3 = 0
    total_queries = len(results)

    for result in results:
        print("\n--- 检索结果 ---")
        print("查询目标图片:", result["query"])
        if "error" in result:
            print(f"Error querying: {result['error']}")
            continue

        query_pid, _ = dataset._parse_filename(result["query"])
        print("最相似的 Top-3 匹配图片:")
        matches = result.get("top_k_matches", [])

        hit_rank1 = False
        hit_rank3 = False

        for idx, match in enumerate(matches):
            match_pid, _ = dataset._parse_filename(match)
            is_match = (match_pid == query_pid)
            mark = "✅" if is_match else "❌"
            print(f"  {idx + 1}. {match} {mark}")
            if is_match:
                hit_rank3 = True
                if idx == 0:
                    hit_rank1 = True

        if hit_rank1: correct_rank1 += 1
        if hit_rank3: correct_rank3 += 1

        mem.add_trial(
            prompt="LangChain CLIP",
            is_correct=hit_rank1,
            analysis=f"QueryPID={query_pid}, R1={'hit' if hit_rank1 else 'miss'}, R3={'hit' if hit_rank3 else 'miss'}"
        )

    print("\n" + "="*30)
    print(f"🚀 评测结果统计 (检索模式: LangChain CLIP Vector DB)")
    print(f"Total Queries: {total_queries}")
    if total_queries > 0:
        r1 = correct_rank1 / total_queries
        r3 = correct_rank3 / total_queries
        print(f"Rank-1 Accuracy: {r1:.2%}")
        print(f"Rank-3 Accuracy: {r3:.2%}")
        mem.summarize_to_long_term(
            f"Single run [LangChain CLIP, {total_queries} queries] R1={r1:.2%} R3={r3:.2%}",
            accuracy=r1,
        )
    print("="*30)

if __name__ == '__main__':
    main()
