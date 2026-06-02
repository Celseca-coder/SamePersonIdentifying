import argparse
import sys
import os
import yaml
from openai import OpenAI
from PIL import Image, ImageFont, ImageDraw
import time      
import csv
import random
import numpy as np

# 强制固定随机种子，保证每次运行抽取到的测试图片一模一样！
random.seed(42)
np.random.seed(42)

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 导入 RGBNT 专属数据加载器
from data_loader import RGBNTDataset
from reid_agent import MockReIDAgent, OpenAIReIDAgent, RandomReIDAgent, LangChainReIDAgent, EvolutionaryReIDAgent, PairWiseLocalAgent, ListWiseLocalAgent, TournamentLocalAgent

# ================= 针对混合图库特制的专家 Guidance =================
MIXED_MULTIMODAL_GUIDANCE = """
    🚨 CRITICAL INSTRUCTION: MULTI-SPECTRAL RE-IDENTIFICATION (HARD MODE).
    The Query image is a standard visible light (RGB) photo.
    HOWEVER, the Gallery Candidates are a MIXTURE of three different camera sensors. You must adapt your reasoning dynamically:
    
    1. If the candidate is an RGB image: Match color, texture, and structure directly.
    2. If the candidate is Grayscale (NIR - Near Infrared): ALL COLOR IS LOST. Do NOT reject based on color mismatch. Match structural patterns (stripes, logos) and accessories (glasses, hats).
    3. If the candidate is a Heatmap (Thermal Infrared): COLOR AND TEXTURE ARE BOTH LOST. You can ONLY match body silhouette, clothing boundaries (shorts vs pants), and the presence of backpacks/bags (which appear as dark/cold geometric blocks).
    
    WARNING: Do NOT reject a candidate simply because the color doesn't match the query if the candidate is clearly from a Thermal or NIR camera!
"""
# ==================================================================

def generate_csv_report(results, accuracy, agent_name, gallery_size, query_size, max_positives, output_file="multi_modal_experiment_summary.csv"):
    file_exists = os.path.isfile(output_file)
    avg_time = sum(r['time_sec'] for r in results) / len(results) if results else 0
    avg_tokens = sum(r['tokens'] for r in results) / len(results) if results else 0
    total_time = sum(r['time_sec'] for r in results)
    total_tokens = sum(r['tokens'] for r in results)

    with open(output_file, mode='a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["Dataset", "Agent", "Query_Size", "Gallery_Size", "Max_Positives", "Trials", "Rank-1_Acc", "Avg_Time_sec", "Total_Time_sec", "Avg_Tokens", "Total_Tokens"])
        
        writer.writerow(["RGBNT201_Mixed", agent_name, query_size, gallery_size, max_positives, len(results), f"{accuracy:.4f}", f"{avg_time:.2f}", f"{total_time:.2f}", f"{avg_tokens:.1f}", f"{total_tokens}"])
    print(f"📊 实验汇总数据已追加至表格: {output_file}")

def generate_trial_images(results, agent_name, output_dir="vis_results"):
    """升级版可视化：支持多 Query 视角的并排展示"""
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
            # 兼容多个 Query
            query_paths = res['query_full_path']
            query_imgs = [Image.open(p) for p in query_paths]
            gallery_imgs = [Image.open(p) for p in res['gallery_full_paths']]
        except Exception as e:
            print(f"加载图片失败，跳过 Trial {i+1}: {e}")
            continue

        img_h, img_w = 192, 96
        target_size = (img_w, img_h)
        
        query_imgs = [img.resize(target_size, Image.Resampling.LANCZOS) for img in query_imgs]
        gallery_imgs = [img.resize(target_size, Image.Resampling.LANCZOS) for img in gallery_imgs]

        padding, text_height, arrow_w = 10, 40, 60
        
        # 动态计算总宽度 (容纳所有 Query 图)
        query_section_width = len(query_imgs) * target_size[0] + (len(query_imgs) - 1) * padding
        total_width = padding + query_section_width + arrow_w + (target_size[0] * len(gallery_imgs)) + (padding * (len(gallery_imgs) + 2))
        total_height = target_size[1] + text_height + 30

        canvas = Image.new('RGB', (total_width, total_height), 'white')
        draw = ImageDraw.Draw(canvas)

        # 绘制所有 Query 视角图
        current_x = padding
        draw.text((current_x, 10), f"Query ({len(query_imgs)} views)", fill="black", font=font)
        
        for q_img in query_imgs:
            canvas.paste(q_img, (current_x, text_height))
            # 统一用蓝色框标出所有的 Query
            draw.rectangle([current_x, text_height, current_x + target_size[0], text_height + target_size[1]], outline="#2196F3", width=3)
            current_x += target_size[0] + padding

        # 绘制箭头
        draw.text((current_x + 10, text_height + target_size[1]//2 - 10), "➔", fill="black", font=font)
        current_x += arrow_w

        # 绘制 Gallery 图 (红绿橙边框判定不变)
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
        f.write(f"# RGBNT Multi-Spectral ReID Report\n\n")
        f.write(f"- **Dataset**: RGBNT201 (Mixed Gallery)\n")
        f.write(f"- **Agent**: {agent_name}\n")
        f.write(f"- **Rank-1 Accuracy**: {accuracy:.2%}\n")
        f.write(f"- **Total Trials**: {len(results)}\n\n")
        
        f.write(f"## Detailed Results\n")
        f.write(f"| Trial | Query Images | Gallery Size | Ground Truth | Prediction | Correct? |\n")
        f.write(f"|---|---|---|---|---|---|\n")
        for i, res in enumerate(results):
            emoji = "✅" if res['correct'] else "❌"
            # 简化 query 显示
            q_display = f"{len(eval(res['query']))} views"
            f.write(f"| {i+1} | {q_display} | {res['gallery_size']} | {res['ground_truth_idx']} | {res['predicted_idx']} | {emoji} |\n")

def run_evaluation(agent_name, data_dir, num_trials=10, query_size=1, gallery_size=10, max_positives=1, api_key=None, report_path="report.md", log_path="log.md", vis_dir="images", model="gpt-4o", base_url=None):
    print(f"Loading RGBNT dataset from {data_dir}...")
    dataset = RGBNTDataset(data_dir)
    
    if agent_name == "mock": agent = MockReIDAgent()
    elif agent_name == "openai": agent = OpenAIReIDAgent(api_key=api_key, model=model, base_url=base_url)
    elif agent_name == "langchain": agent = LangChainReIDAgent(api_key=api_key, model=model, base_url=base_url)
    elif agent_name == "random": agent = RandomReIDAgent()
    elif agent_name == "pairwise": agent = PairWiseLocalAgent(api_key=api_key, model=model, base_url=base_url)
    elif agent_name == "listwise": agent = ListWiseLocalAgent(api_key=api_key, model=model, base_url=base_url)
    elif agent_name == "tournament": agent = TournamentLocalAgent(api_key=api_key, model=model, base_url=base_url, group_size=5)
    elif agent_name == "evo": agent = EvolutionaryReIDAgent(api_key=api_key, model=model, base_url=base_url)
    else:
        print(f"Unknown agent: {agent_name}")
        return
    
    if hasattr(agent, 'log_file'):
        agent.log_file = log_path
    elif hasattr(agent, 'base_agent') and hasattr(agent.base_agent, 'log_file'):
        agent.base_agent.log_file = log_path

    print(f"Starting Multi-Spectral evaluation with {agent_name} agent...")
    print(f"Trials: {num_trials}, Query Size: {query_size}, Gallery Size: {gallery_size}")
    
    correct_count = 0
    results = []

    for i in range(num_trials):
        # 1. 抽取混合光谱的测试用例 (Query 是 RGB，Gallery 包含 RGB, NIR, Thermal)
        # 注意: 这里的 gt_indices 已经是一个列表
        query_paths, gallery_paths, ground_truth_idx = dataset.get_mixed_gallery_test_case(query_mod="RGB", query_size=query_size, gallery_size=gallery_size)
        
        start_time = time.time()
        start_tokens = getattr(agent.base_agent if hasattr(agent, 'base_agent') else agent, 'total_tokens_used', 0)
        
        # 2. Predict (注入混合光谱专属 Guidance)
        try:
            if isinstance(agent, EvolutionaryReIDAgent):
                pred_idx = agent.predict(query_paths, gallery_paths, ground_truth_idx=ground_truth_idx, guidance=MIXED_MULTIMODAL_GUIDANCE)
            elif isinstance(agent, (PairWiseLocalAgent, ListWiseLocalAgent, TournamentLocalAgent)):
                pred_idx, _ = agent.predict(query_paths, gallery_paths, guidance=MIXED_MULTIMODAL_GUIDANCE)
            else:
                pred_idx = agent.predict(query_paths, gallery_paths)
        except Exception as e:
            print(f"Prediction failed: {e}")
            pred_idx = -1
            
        end_time = time.time()
        end_tokens = getattr(agent.base_agent if hasattr(agent, 'base_agent') else agent, 'total_tokens_used', 0)
            
        trial_time = end_time - start_time
        trial_tokens = end_tokens - start_tokens
        
        # 3. Evaluate 
        # 因为 rgbnt_data_loader 保证了目标正样本仅有 1 个，所以可以直接用 in 或 ==
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
        
        print(f"Trial {i+1}/{num_trials}: {'✅ Correct' if is_correct else '❌ Incorrect'} (GT: {ground_truth_idx}, Pred: {pred_idx})")
    
    accuracy = correct_count / num_trials
    print("\n" + "="*60)
    # 严格遵照你的实验文档输出格式
    report_sentence = f"多模态数据集混合评测：当输入query大小为（{query_size}）gallery大小为（{gallery_size}）时，大模型（使用{agent_name}策略）在【可见光搜混合光谱】任务下的rank-1的成功率为（{accuracy:.2%}）"
    print("🏆 【实验文档要求输出格式】:")
    print(report_sentence)
    print("="*60 + "\n")
    
    generate_report(results, accuracy, agent_name, report_path)
    generate_trial_images(results, agent_name, vis_dir)
    generate_csv_report(results, accuracy, agent_name, gallery_size, query_size, max_positives)

    if isinstance(agent, EvolutionaryReIDAgent):
        summary = f"Session completed with {accuracy:.2%} accuracy. Total trials: {num_trials}."
        agent.persistent_memory.summarize_to_long_term(summary)
    
    return accuracy, results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    # 默认路径指向 RGBNT201
    parser.add_argument("--data_dir", type=str, default="/home/user/GSK/heyalan/Reid/data/multi_dataset/RGBNT201")
    parser.add_argument("--agent", type=str, default="listwise-local", choices=["mock", "openai", "langchain", "random", "pairwise", "listwise", "tournament", "evo"])
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument("--query_size", type=int, default=3)
    parser.add_argument("--gallery_size", type=int, default=10)
    parser.add_argument("--max_positives", type=int, default=1) # 混合光谱默认每个 ID 在 Gallery 中只放 1 个 GT
    parser.add_argument("--api_key", type=str, default="local-test")
    parser.add_argument("--model", type=str, default="/data/llm/AI-ModelScope/R-4B")
    parser.add_argument("--base_url", type=str, default="http://localhost:8000/v1")
    
    args = parser.parse_args()

    agent = args.agent 
    data_dir = args.data_dir 
    trials = args.trials 
    gallery_size = args.gallery_size
    api_key = args.api_key 
    model = args.model 
    base_url = args.base_url 
    max_positives = args.max_positives 
    query_size = args.query_size 
    
    # ---------------------------------------------------------
    # 创建独立的 RGBNT 隔离文件夹
    # ---------------------------------------------------------
    os.makedirs("report_rgbnt", exist_ok=True)
    os.makedirs("log_rgbnt", exist_ok=True)
    os.makedirs("images_rgbnt", exist_ok=True)
    
    base_name = f"{agent}_q{query_size}_g{gallery_size}"
    
    report_path = os.path.join("report_rgbnt", f"report_{base_name}.md")
    log_path = os.path.join("log_rgbnt", f"inference_log_{base_name}.md")
    vis_dir = os.path.join("images_rgbnt", f"vis_{base_name}")

    run_evaluation(
        agent_name=agent, 
        data_dir=data_dir, 
        num_trials=trials, 
        gallery_size=gallery_size, 
        query_size=query_size,
        max_positives=max_positives, 
        api_key=api_key, 
        report_path=report_path, 
        log_path=log_path,        
        vis_dir=vis_dir,          
        model=model, 
        base_url=base_url
    )