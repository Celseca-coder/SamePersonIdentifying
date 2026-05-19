import argparse
import sys
import os
import yaml
from openai import OpenAI
from PIL import Image, ImageFont, ImageDraw
import time      # 用于计算耗时
import csv
import random
import numpy as np

# 强制固定随机种子，保证每次运行抽取到的测试图片一模一样！
random.seed(42)
np.random.seed(42)

# Ensure src is in path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_loader import Market1501Dataset
from reid_agent import MockReIDAgent, OpenAIReIDAgent, RandomReIDAgent, LangChainReIDAgent, EvolutionaryReIDAgent, PairWiseLocalAgent, ListWiseLocalAgent, TournamentLocalAgent

def generate_csv_report(results, accuracy, agent_name, gallery_size, query_size, max_positives, output_file="experiment_summary.csv"):
    file_exists = os.path.isfile(output_file)
    avg_time = sum(r['time_sec'] for r in results) / len(results) if results else 0
    avg_tokens = sum(r['tokens'] for r in results) / len(results) if results else 0
    total_time = sum(r['time_sec'] for r in results)
    total_tokens = sum(r['tokens'] for r in results)

    with open(output_file, mode='a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if not file_exists:
            # 写入表头
            writer.writerow(["Agent", "Query_Size", "Gallery_Size", "Max_Positives", "Trials", "Rank-1_Acc", "Avg_Time_sec", "Total_Time_sec", "Avg_Tokens", "Total_Tokens"])
        
        # 写入本次实验的汇总数据
        writer.writerow([agent_name, query_size, gallery_size, max_positives, len(results), f"{accuracy:.4f}", f"{avg_time:.2f}", f"{total_time:.2f}", f"{avg_tokens:.1f}", f"{total_tokens}"])
    print(f"📊 实验汇总数据已追加至表格: {output_file}")

def generate_trial_images(results, agent_name, output_dir="vis_results"):
    """为每一个 Trial 生成可视化的拼接图片并保存 (适配多目标逻辑)"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        print(f"创建可视化结果保存目录: {output_dir}")

    try:
        font = ImageFont.truetype("arial.ttf", 24)
        small_font = ImageFont.truetype("arial.ttf", 18)
    except IOError:
        font = ImageFont.load_default()
        small_font = ImageFont.load_default()

    for i, res in enumerate(results):
        try:
            query_img = Image.open(res['query_full_path'][0])
            gallery_imgs = [Image.open(p) for p in res['gallery_full_paths']]
        except Exception as e:
            print(f"加载图片失败，跳过 Trial {i+1}: {e}")
            continue

        # 稍微放大图片方便观察 (128x64 -> 192x96)
        img_h, img_w = 192, 96
        target_size = (img_w, img_h)
        query_img = query_img.resize(target_size, Image.Resampling.LANCZOS)
        gallery_imgs = [img.resize(target_size, Image.Resampling.LANCZOS) for img in gallery_imgs]

        # 布局计算
        padding, text_height, arrow_w = 10, 40, 60
        total_width = target_size[0] + arrow_w + (target_size[0] * len(gallery_imgs)) + (padding * (len(gallery_imgs) + 2))
        total_height = target_size[1] + text_height + 30

        canvas = Image.new('RGB', (total_width, total_height), 'white')
        draw = ImageDraw.Draw(canvas)

        # 绘制 Query
        current_x = padding
        canvas.paste(query_img, (current_x, text_height))
        draw.text((current_x, 10), "Query", fill="black", font=font)
        # Query 统一用蓝色框
        draw.rectangle([current_x, text_height, current_x + target_size[0], text_height + target_size[1]], outline="#2196F3", width=3)

        # 绘制箭头
        current_x += target_size[0] + padding
        draw.text((current_x + 10, text_height + target_size[1]//2 - 10), "➔", fill="black", font=font)
        current_x += arrow_w

        # 绘制 Gallery
        gt_indices = res['gt_indices']
        pred_indices = res['pred_indices']
        
        for idx, g_img in enumerate(gallery_imgs):
            canvas.paste(g_img, (current_x, text_height))
            draw.text((current_x + 5, text_height + target_size[1] + 5), f"G-{idx}", fill="black", font=small_font)

            # --- 核心修改：多答案的红绿橙边框判定 ---
            border_color = None
            if idx in pred_indices and idx in gt_indices:
                border_color = "#4CAF50" # 🟩 绿色：True Positive (抓对了)
            elif idx in pred_indices and idx not in gt_indices:
                border_color = "#f44336" # 🟥 红色：False Positive (抓错了/误报)
            elif idx not in pred_indices and idx in gt_indices:
                border_color = "#FF9800" # 🟧 橙色：False Negative (漏抓了)
            
            if border_color:
                draw.rectangle([current_x, text_height, current_x + target_size[0], text_height + target_size[1]], outline=border_color, width=4)
            
            current_x += target_size[0] + padding

        # 保存
        status = "Perfect" if res['is_perfect'] else "Flawed"
        canvas.save(os.path.join(output_dir, f"trial_{i+1:02d}_{status}.png"))


def generate_report(results, accuracy, agent_name, output_file="report.md"):
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(f"# Person Re-Identification Evaluation Report\n\n")
        f.write(f"- **Agent**: {agent_name}\n")
        f.write(f"- **Rank-1 Accuracy**: {accuracy:.2%}\n")
        f.write(f"- **Total Trials**: {len(results)}\n\n")
        
        f.write(f"## Detailed Results\n")
        f.write(f"| Trial | Query Image | Gallery Size | Ground Truth | Prediction | Correct? |\n")
        f.write(f"|---|---|---|---|---|---|\n")
        for i, res in enumerate(results):
            emoji = "✅" if res['correct'] else "❌"
            f.write(f"| {i+1} | {res['query']} | {res['gallery_size']} | {res['ground_truth_idx']} | {res['predicted_idx']} | {emoji} |\n")
        
        f.write("\n## Analysis\n")
        f.write("- **Correct Cases**: Explain what worked well (e.g. distinct clothing, clear query).\n")
        f.write("- **Incorrect Cases**: Explain failures (e.g. occlusion, low resolution, similar distractors).\n")
        f.write("- **Recommendation**: Use higher resolution inputs or better prompt engineering for LMMs.\n")


# 【修改】接收 log_path 和 vis_dir 作为独立参数
def run_evaluation(agent_name, data_dir, num_trials=10, query_size=1, gallery_size=10, max_positives=1, api_key=None, report_path="report.md", log_path="log.md", vis_dir="images", model="gpt-4o", base_url=None):
    print(f"Loading dataset from {data_dir}...")
    dataset = Market1501Dataset(data_dir)
    
    if agent_name == "mock":
        agent = MockReIDAgent()
    elif agent_name == "openai":
        if not api_key:
            print("Error: API Key needed for OpenAI agent.")
            return
        agent = OpenAIReIDAgent(api_key=api_key, model=model, base_url=base_url)
    elif agent_name == "langchain":
        if not api_key:
            print("Error: API Key needed for LangChain agent.")
            return
        agent = LangChainReIDAgent(api_key=api_key, model=model, base_url=base_url)
    elif agent_name == "random":
        agent = RandomReIDAgent()
    elif agent_name == "pairwise-local":
        agent = PairWiseLocalAgent(api_key=api_key, model=model, base_url=base_url)
    elif agent_name == "listwise-local":
        agent = ListWiseLocalAgent(api_key=api_key, model=model, base_url=base_url)
    elif agent_name == "tournament-local":
        agent = TournamentLocalAgent(api_key=api_key, model=model, base_url=base_url, group_size=5)
    elif agent_name == "evo":
        if not api_key:
            print("Error: API Key needed for Evolutionary agent.")
            return
        agent = EvolutionaryReIDAgent(api_key=api_key, model=model, base_url=base_url)
    else:
        print(f"Unknown agent: {agent_name}")
        return
    
    # 【修改】直接使用传入的独立 log_path
    if hasattr(agent, 'log_file'):
        agent.log_file = log_path
    elif hasattr(agent, 'base_agent') and hasattr(agent.base_agent, 'log_file'):
        agent.base_agent.log_file = log_path

    print(f"Starting evaluation with {agent_name} agent...")
    print(f"Trials: {num_trials}, Gallery Size: {gallery_size}")
    
    correct_count = 0
    results = []

    for i in range(num_trials):
        # 1. Get case
        query_paths, gallery_paths, ground_truth_idx = dataset.get_test_case(query_size=query_size, gallery_size=gallery_size, max_positives=max_positives)
        
        start_time = time.time()
        # 提取当前的 Token 表盘数
        if isinstance(agent, EvolutionaryReIDAgent):
            start_tokens = getattr(agent.base_agent, 'total_tokens_used', 0)
        else:
            start_tokens = getattr(agent, 'total_tokens_used', 0)
        
        # 2. Predict
        try:
            if isinstance(agent, EvolutionaryReIDAgent):
                pred_idx = agent.predict(query_paths, gallery_paths, ground_truth_idx=ground_truth_idx)
            else:
                if isinstance(agent, (PairWiseLocalAgent, ListWiseLocalAgent, TournamentLocalAgent)):
                    pred_idx, _ = agent.predict(query_paths, gallery_paths)
                else:
                    pred_idx = agent.predict(query_paths, gallery_paths)
        except Exception as e:
            print(f"Prediction failed: {e}")
            pred_idx = -1
            
        end_time = time.time()
        if isinstance(agent, EvolutionaryReIDAgent):
            end_tokens = getattr(agent.base_agent, 'total_tokens_used', 0)
        else:
            end_tokens = getattr(agent, 'total_tokens_used', 0)
            
        trial_time = end_time - start_time
        trial_tokens = end_tokens - start_tokens
        
        # 3. Evaluate
        if isinstance(ground_truth_idx, list):
            if isinstance(pred_idx, list):
                is_correct = (set(pred_idx) == set(ground_truth_idx))
            else:
                is_correct = (pred_idx in ground_truth_idx)
        else:
            is_correct = (pred_idx == ground_truth_idx)

        if is_correct:
            correct_count += 1
        
        gt_list = ground_truth_idx if isinstance(ground_truth_idx, list) else [ground_truth_idx]
        pred_list = pred_idx if isinstance(pred_idx, list) else ([pred_idx] if pred_idx != -1 else [])
        
        # Log
        results.append({
            "query": str([os.path.basename(p) for p in query_paths]),
            "gallery_size": len(gallery_paths),
            "ground_truth_idx": ground_truth_idx,
            "predicted_idx": pred_idx,
            "correct": is_correct,
            "query_full_path": query_paths, 
            "gallery_full_paths": gallery_paths,
            "gt_indices": gt_list,
            "pred_indices": pred_list,
            "is_perfect": is_correct,
            "time_sec": trial_time,   
            "tokens": trial_tokens
        })
        
        print(f"Trial {i+1}/{num_trials}: {'Correct' if is_correct else 'Incorrect'} (GT: {ground_truth_idx}, Pred: {pred_idx})")
    
    accuracy = correct_count / num_trials
    print("="*30)
    print(f"Final Rank-1 Accuracy: {accuracy:.2%}")
    print("="*30)
    
    generate_report(results, accuracy, agent_name, report_path)
    print(f"Report saved to {report_path}")
    
    # 【修改】直接使用传入的 vis_dir
    generate_trial_images(results, agent_name, vis_dir)
    
    # 注意：CSV 我们依旧保存在根目录，因为它是所有参数追加在一起的总体汇总表
    generate_csv_report(results, accuracy, agent_name, gallery_size, query_size, max_positives)

    # 对于进化代理，在任务结束时将总结写入长时记忆
    if isinstance(agent, EvolutionaryReIDAgent):
        summary = f"Session completed with {accuracy:.2%} accuracy. Total trials: {num_trials}."
        agent.persistent_memory.summarize_to_long_term(summary)
        print("Progress summarized to long-term memory.")
    
    return accuracy, results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--agent", type=str, default=None, choices=["mock", "openai", "langchain", "random", "pairwise-local", "listwise-local", "tournament-local", "evo"])
    parser.add_argument("--trials", type=int, default=None)
    parser.add_argument("--query_size", type=int, default=1)
    parser.add_argument("--gallery_size", type=int, default=None)
    parser.add_argument("--max_positives", type=int, default=1)
    parser.add_argument("--api_key", type=str, default=None)
    parser.add_argument("--report", type=str, default="report.md")
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--base_url", type=str, default=None)
    
    args = parser.parse_args()

    # Load from YAML if exists
    config = {}
    if os.path.exists(args.config):
        with open(args.config, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)

    # CLI args override YAML config
    agent = args.agent or config.get("agent", "mock")
    data_dir = args.data_dir or config.get("data_dir", r"l:\CV\data\1\Market-1501-v15.09.15")
    trials = args.trials or config.get("trials", 10)
    gallery_size = args.gallery_size or config.get("gallery_size", 10)
    api_key = args.api_key or config.get("api_key") or "EMPTY"
    model = args.model or config.get("model", "gpt-4o")
    base_url = args.base_url or config.get("base_url")
    max_positives = config.get("max_positives", 1)
    query_size = args.query_size or config.get("query_size", 1)
    
    # ---------------------------------------------------------
    # 【核心改动：自动建立文件夹与路径拼接】
    # ---------------------------------------------------------
    # 1. 自动创建三大文件夹
    os.makedirs("report", exist_ok=True)
    os.makedirs("log", exist_ok=True)
    os.makedirs("images", exist_ok=True)
    
    # 2. 构造基础文件名
    base_name = f"{agent}_q{query_size}_g{gallery_size}"
    
    # 3. 分配到对应文件夹下
    if args.report == "report.md":
        report_path = os.path.join("report", f"report_{base_name}.md")
    else:
        report_path = args.report

    log_path = os.path.join("log", f"inference_log_{base_name}.md")
    vis_dir = os.path.join("images", f"vis_{base_name}")
    # ---------------------------------------------------------

    run_evaluation(
        agent_name=agent, 
        data_dir=data_dir, 
        num_trials=trials, 
        gallery_size=gallery_size, 
        query_size=query_size,
        max_positives=max_positives, 
        api_key=api_key, 
        report_path=report_path, 
        log_path=log_path,        # <--- 将独立日志路径传入
        vis_dir=vis_dir,          # <--- 将独立图片文件夹传入
        model=model, 
        base_url=base_url
    )