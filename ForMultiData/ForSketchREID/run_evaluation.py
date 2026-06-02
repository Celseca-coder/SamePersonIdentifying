import argparse
import sys
import os
import yaml
from PIL import Image, ImageFont, ImageDraw
import time      
import csv
import random
import numpy as np

# 强制固定随机种子，保证每次运行抽取到的测试图片一模一样！
random.seed(42)
np.random.seed(42)

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_loader import PKUSketchDataset
from reid_agent import PairWiseLocalAgent, ListWiseLocalAgent, TournamentLocalAgent, EvolutionaryReIDAgent, SKETCH_GUIDANCE

def generate_csv_report(results, accuracy, agent_name, gallery_size, query_size, trials, output_file="multi_modal_experiment_summary.csv"):
    file_exists = os.path.isfile(output_file)
    avg_time = sum(r['time_sec'] for r in results) / len(results) if results else 0
    avg_tokens = sum(r['tokens'] for r in results) / len(results) if results else 0
    total_time = sum(r['time_sec'] for r in results)
    total_tokens = sum(r['tokens'] for r in results)

    with open(output_file, mode='a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["Dataset", "Agent", "Query_Size", "Gallery_Size", "Trials", "Rank-1_Acc", "Avg_Time_sec", "Total_Time_sec", "Avg_Tokens", "Total_Tokens"])
        
        # 写入本次实验的汇总数据
        writer.writerow(["PKUSketch", agent_name, query_size, gallery_size, trials, f"{accuracy:.4f}", f"{avg_time:.2f}", f"{total_time:.2f}", f"{avg_tokens:.1f}", f"{total_tokens}"])
    print(f"📊 实验汇总数据已追加至表格: {output_file}")

def generate_trial_images(results, agent_name, output_dir="vis_results"):
    """为每一个 Trial 生成可视化的拼接图片并保存"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        print(f"🖼️ 创建可视化结果保存目录: {output_dir}")

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

        # 调整大小，适配草图和图片的比例 (将原先的128x64改为更方正的128x256或维持等比，这里统一用 128x256 方便看人)
        img_w, img_h = 128, 256
        target_size = (img_w, img_h)
        query_img = query_img.resize(target_size, Image.Resampling.LANCZOS)
        gallery_imgs = [img.resize(target_size, Image.Resampling.LANCZOS) for img in gallery_imgs]

        padding, text_height, arrow_w = 10, 40, 60
        total_width = target_size[0] + arrow_w + (target_size[0] * len(gallery_imgs)) + (padding * (len(gallery_imgs) + 2))
        total_height = target_size[1] + text_height + 30

        canvas = Image.new('RGB', (total_width, total_height), 'white')
        draw = ImageDraw.Draw(canvas)

        # 绘制 Query (草图)
        current_x = padding
        canvas.paste(query_img, (current_x, text_height))
        draw.text((current_x, 10), "Sketch Query", fill="black", font=font)
        draw.rectangle([current_x, text_height, current_x + target_size[0], text_height + target_size[1]], outline="#2196F3", width=3)

        # 绘制箭头
        current_x += target_size[0] + padding
        draw.text((current_x + 10, text_height + target_size[1]//2 - 10), "➔", fill="black", font=font)
        current_x += arrow_w

        # 绘制 Gallery (原图)
        gt_indices = res['gt_indices']
        pred_indices = res['pred_indices']
        
        for idx, g_img in enumerate(gallery_imgs):
            canvas.paste(g_img, (current_x, text_height))
            draw.text((current_x + 5, text_height + target_size[1] + 5), f"G-{idx}", fill="black", font=small_font)

            border_color = None
            if idx in pred_indices and idx in gt_indices:
                border_color = "#4CAF50" # 🟩 绿色：True Positive 
            elif idx in pred_indices and idx not in gt_indices:
                border_color = "#f44336" # 🟥 红色：False Positive 
            elif idx not in pred_indices and idx in gt_indices:
                border_color = "#FF9800" # 🟧 橙色：False Negative 
            
            if border_color:
                draw.rectangle([current_x, text_height, current_x + target_size[0], text_height + target_size[1]], outline=border_color, width=4)
            
            current_x += target_size[0] + padding

        status = "Perfect" if res['is_perfect'] else "Flawed"
        canvas.save(os.path.join(output_dir, f"trial_{i+1:02d}_{status}.png"))

def generate_report(results, accuracy, agent_name, output_file="report.md"):
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(f"# Multi-Modal (Sketch) Re-Identification Report\n\n")
        f.write(f"- **Dataset**: PKUSketchRE-ID_V1\n")
        f.write(f"- **Agent Strategy**: {agent_name}\n")
        f.write(f"- **Rank-1 Accuracy**: {accuracy:.2%}\n")
        f.write(f"- **Total Trials**: {len(results)}\n\n")
        
        f.write(f"## Detailed Results\n")
        f.write(f"| Trial | Sketch Query | Gallery Size | Ground Truth | Prediction | Correct? | Tokens |\n")
        f.write(f"|---|---|---|---|---|---|---|\n")
        for i, res in enumerate(results):
            emoji = "✅" if res['correct'] else "❌"
            f.write(f"| {i+1} | {res['query']} | {res['gallery_size']} | {res['ground_truth_idx']} | {res['predicted_idx']} | {emoji} | {res['tokens']} |\n")

def run_sketch_evaluation(agent_name, data_dir, num_trials=10, query_size=1, gallery_size=20, api_key=None, report_path="report.md", log_path="log.md", vis_dir="images", model="gpt-4o", base_url=None):
    print(f"Loading PKUSketch dataset from {data_dir}...")
    dataset = PKUSketchDataset(data_dir)
    
    # 策略路由
    if agent_name == "pairwise":
        agent = PairWiseLocalAgent(api_key=api_key, model=model, base_url=base_url)
    elif agent_name == "listwise":
        agent = ListWiseLocalAgent(api_key=api_key, model=model, base_url=base_url)
    elif agent_name == "tournament":
        agent = TournamentLocalAgent(api_key=api_key, model=model, base_url=base_url, group_size=5)
    elif agent_name == "evo":
        agent = EvolutionaryReIDAgent(api_key=api_key, model=model, base_url=base_url)
    else:
        print(f"Unknown agent: {agent_name}")
        return
    
    # 设置日志路径
    if hasattr(agent, 'log_file'):
        agent.log_file = log_path
    elif hasattr(agent, 'base_agent') and hasattr(agent.base_agent, 'log_file'):
        agent.base_agent.log_file = log_path

    print(f"Starting Multi-Modal evaluation with {agent_name} agent...")
    print(f"Trials: {num_trials}, Query Size: {query_size}, Gallery Size: {gallery_size}")
    
    correct_count = 0
    results = []

    for i in range(num_trials):
        # 1. 抽取草图专属测试用例 (Query是草图，Gallery是原图)
        query_paths, gallery_paths, gt_indices = dataset.get_test_case(query_size=query_size, gallery_size=gallery_size)
        
        start_time = time.time()
        start_tokens = getattr(agent.base_agent if hasattr(agent, 'base_agent') else agent, 'total_tokens_used', 0)
        
        # 2. Predict (注入草图专属 GUIDANCE)
        try:
            if isinstance(agent, EvolutionaryReIDAgent):
                # 进化体需要知道标准答案用于反思
                pred_idx = agent.predict(query_paths, gallery_paths, ground_truth_idx=gt_indices[0], guidance=SKETCH_GUIDANCE)
            else:
                pred_idx, _ = agent.predict(query_paths, gallery_paths, guidance=SKETCH_GUIDANCE)
        except Exception as e:
            print(f"Prediction failed: {e}")
            pred_idx = -1
            
        end_time = time.time()
        end_tokens = getattr(agent.base_agent if hasattr(agent, 'base_agent') else agent, 'total_tokens_used', 0)
            
        trial_time = end_time - start_time
        trial_tokens = end_tokens - start_tokens
        
        # 3. Evaluate
        is_correct = (pred_idx in gt_indices)
        if is_correct:
            correct_count += 1
        
        pred_list = [pred_idx] if pred_idx != -1 else []
        
        # Log
        results.append({
            "query": str([os.path.basename(p) for p in query_paths]),
            "gallery_size": len(gallery_paths),
            "ground_truth_idx": gt_indices,
            "predicted_idx": pred_idx,
            "correct": is_correct,
            "query_full_path": query_paths, 
            "gallery_full_paths": gallery_paths,
            "gt_indices": gt_indices,
            "pred_indices": pred_list,
            "is_perfect": is_correct,
            "time_sec": trial_time,   
            "tokens": trial_tokens
        })
        
        print(f"Trial {i+1}/{num_trials}: {'✅ Correct' if is_correct else '❌ Incorrect'} (GT: {gt_indices}, Pred: {pred_idx})")
    
    accuracy = correct_count / num_trials
    print("\n" + "="*60)
    
    # 【核心：严格按照你实验文档要求的最终输出格式】
    report_sentence = f"多模态数据集混合评测：当输入query大小为（{query_size}）gallery大小为（{gallery_size}）时，大模型（使用{agent_name}策略）的rank-1的成功率为（{accuracy:.2%}）"
    print("🏆 【实验文档要求输出格式】:")
    print(report_sentence)
    print("="*60 + "\n")
    
    generate_report(results, accuracy, agent_name, report_path)
    generate_trial_images(results, agent_name, vis_dir)
    generate_csv_report(results, accuracy, agent_name, gallery_size, query_size, num_trials)

    return accuracy, results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--data_dir", type=str, default="/home/user/GSK/heyalan/Reid/data/multi_dataset/PKUSketchRE-ID_V1")
    parser.add_argument("--agent", type=str, default="listwise", choices=["pairwise", "listwise", "tournament", "evo"])
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument("--query_size", type=int, default=1) # 草图只能是1
    parser.add_argument("--gallery_size", type=int, default=5)
    parser.add_argument("--api_key", type=str, default="local-test")
    parser.add_argument("--model", type=str, default="/data/llm/AI-ModelScope/R-4B")
    parser.add_argument("--base_url", type=str, default="http://localhost:8000/v1")
    
    args = parser.parse_args()

    # CLI args override logic
    agent = args.agent 
    data_dir = args.data_dir
    trials = args.trials 
    gallery_size = args.gallery_size
    api_key = args.api_key 
    model = args.model
    base_url = args.base_url 
    query_size = args.query_size 
    
    os.makedirs("report_sketch", exist_ok=True)
    os.makedirs("log_sketch", exist_ok=True)
    os.makedirs("images_sketch", exist_ok=True)
    
    base_name = f"{agent}_q{query_size}_g{gallery_size}"
    
    report_path = os.path.join("report_sketch", f"report_{base_name}.md")
    log_path = os.path.join("log_sketch", f"inference_log_{base_name}.md")
    vis_dir = os.path.join("images_sketch", f"vis_{base_name}")

    run_sketch_evaluation(
        agent_name=agent, 
        data_dir=data_dir, 
        num_trials=trials, 
        gallery_size=gallery_size, 
        query_size=query_size,
        api_key=api_key, 
        report_path=report_path, 
        log_path=log_path,        
        vis_dir=vis_dir,          
        model=model, 
        base_url=base_url
    )