import argparse
import sys
import os
import io
from PIL import Image, ImageDraw, ImageFont

# Ensure src is in path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_loader import Market1501Dataset
from reid_agent import MockReIDAgent, OpenAIReIDAgent, RandomReIDAgent, LangChainReIDAgent, QwenReIDAgent, LocalReIDAgent

def generate_trial_images(results, agent_name, output_dir="vis_results"):
    """为每一个 Trial 生成可视化的拼接图片并保存 (适配多目标逻辑)"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"创建可视化结果保存目录: {output_dir}")

    try:
        font = ImageFont.truetype("arial.ttf", 24)
        small_font = ImageFont.truetype("arial.ttf", 18)
    except IOError:
        font = ImageFont.load_default()
        small_font = ImageFont.load_default()

    for i, res in enumerate(results):
        try:
            query_img = Image.open(res['query_full_path'])
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

def generate_report(results, metrics, agent_name, output_file="report.md"):
    """生成 Markdown 报告 (适配多目标指标)"""
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(f"# Person Re-Identification Evaluation Report\n\n")
        f.write(f"- **Agent**: {agent_name}\n")
        f.write(f"- **Perfect Match Rate**: {metrics['perfect_rate']:.2%} (完全没有误报和漏报的比例)\n")
        f.write(f"- **Overall Precision**: {metrics['precision']:.2%} (模型抓的人里，有多少是真正确的)\n")
        f.write(f"- **Overall Recall**: {metrics['recall']:.2%} (真正的目标里，模型抓到了多少)\n")
        f.write(f"- **Total Trials**: {len(results)}\n\n")
        
        f.write(f"## Detailed Results\n")
        f.write(f"| Trial | Query Image | Ground Truth | Prediction | Perfect? |\n")
        f.write(f"|---|---|---|---|---|\n")
        for i, res in enumerate(results):
            emoji = "✅" if res['is_perfect'] else "⚠️"
            # 将列表转换为字符串展示
            gt_str = str(res['gt_indices']) if res['gt_indices'] else "None"
            pred_str = str(res['pred_indices']) if res['pred_indices'] else "None"
            
            f.write(f"| {i+1} | {res['query']} | {gt_str} | {pred_str} | {emoji} |\n")

def run_evaluation(agent_name, data_dir, num_trials=10, gallery_size=10, api_key=None, base_url="", report_path="report.md"):
    print(f"Loading dataset from {data_dir}...")
    dataset = Market1501Dataset(data_dir)
    
    # 初始化 Agent
    if agent_name == "local": agent = LocalReIDAgent(base_url=base_url)
    elif agent_name == "mock": agent = MockReIDAgent()
    elif agent_name == "random": agent = RandomReIDAgent()
    else: agent = OpenAIReIDAgent(api_key=api_key, base_url=base_url)

    results = []
    total_tp = 0
    total_fp = 0
    total_fn = 0
    perfect_trials = 0

    for i in range(num_trials):
        # 接收改版后返回的 gt_indices 列表
        query_path, gallery_paths, gt_indices = dataset.get_test_case(gallery_size=gallery_size)
        try:
            # 接收改版后返回的 predicted_indices 列表
            pred_indices = agent.predict(query_path, gallery_paths)
        except Exception as e:
            print(f"Prediction failed: {e}")
            pred_indices = []
        
        # --- 核心修改：计算 TP, FP, FN ---
        tp = len(set(pred_indices) & set(gt_indices))
        fp = len(set(pred_indices) - set(gt_indices))
        fn = len(set(gt_indices) - set(pred_indices))
        
        total_tp += tp
        total_fp += fp
        total_fn += fn
        
        # 判断这次测试是否完美（没抓错，也没漏抓）
        is_perfect = (set(pred_indices) == set(gt_indices))
        if is_perfect:
            perfect_trials += 1
        
        results.append({
            "query": os.path.basename(query_path),
            "query_full_path": query_path,
            "gallery_full_paths": gallery_paths,
            "gt_indices": gt_indices,
            "pred_indices": pred_indices,
            "is_perfect": is_perfect
        })
        print(f"Trial {i+1}/{num_trials}: {'Perfect ✅' if is_perfect else 'Flawed ⚠️'}")

    # 计算全局指标，防止分母为0
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    perfect_rate = perfect_trials / num_trials

    metrics = {
        "perfect_rate": perfect_rate,
        "precision": precision,
        "recall": recall
    }
    
    # 生成报告和图片
    generate_report(results, metrics, agent_name, report_path)
    vis_dir = report_path.replace(".md", "") + "_images"
    generate_trial_images(results, agent_name, vis_dir)
    
    print(f"✅ 评估完成。")
    print(f"📊 Perfect Match Rate: {perfect_rate:.2%}")
    print(f"🎯 Precision: {precision:.2%} | 🔍 Recall: {recall:.2%}")
    print(f"📄 Markdown报告: {report_path}")
    print(f"🖼️ 可视化图片目录: {vis_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="/home/user/GSK/heyalan/Reid/dataset/Market-1501-v15.09.15")
    parser.add_argument("--agent", type=str, default="local")
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument("--gallery_size", type=int, default=10)
    parser.add_argument("--base_url", type=str, default="http://localhost:8000/v1")
    parser.add_argument("--api_key", type=str, default="sk-xxx")
    parser.add_argument("--report", type=str, default="report_R-4B_multi.md")
    args = parser.parse_args()
    
    run_evaluation(args.agent, args.data_dir, args.trials, args.gallery_size, args.api_key, args.base_url, args.report)