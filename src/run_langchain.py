import os
import sys
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# 确保能找到 src 目录下的模块
sys.path.append(os.path.abspath("./src"))

from reid_agent import LangChainBatchImageRetrievalManager
from data_loader import Market1501Dataset

def main():
    # 1. 实例化我们的 LangChain 检索管理器
    # 会在当前目录下创建一个叫做 "langchain_reid_db" 的文件夹用于持久化数据库
    manager = LangChainBatchImageRetrievalManager(persist_directory="./langchain_reid_db")
    
    # 防止依赖未正确安装
    if not manager.vectorstore:
        print("请检查并安装依赖：pip install langchain-experimental open_clip_torch chromadb torch")
        return

    # 2. 准备数据 (假设我们从您的数据集中抽样)
    data_dir = "/mnt/l/CV/data/1/Market-1501-v15.09.15"
    dataset = Market1501Dataset(data_dir)
    
    # 提取测试数据：我们抓取一个 Query 图片和对应的数十个/数百个 Gallery 图片
    query_path, gallery_paths, ground_truth_idx = dataset.get_test_case(gallery_size=100)
    
    # 3. 运行：成批构建并保存图库到向量数据库数据库
    print(f"正在建立包含 {len(gallery_paths)} 张图片的索引库...")
    manager.build_gallery_batch(gallery_paths, batch_size=32)
    
    # 4. 运行：自动化成批地检索 Query 
    print(f"\n正在检索 Query: {query_path}")
    # 这里为了演示用单个 query 构建一个列表
    queries = [query_path]
    results = manager.automate_batch_query(queries, k=3)
    
    # 5. 查看返回的结果
    for result in results:
        print("\n--- 检索结果 ---")
        print("查询目标图片:", result["query"])
        print("最相似的 Top-3 匹配图片:")
        for idx, match in enumerate(result["top_k_matches"]):
            print(f"  {idx + 1}. {match}")

if __name__ == '__main__':
    main()