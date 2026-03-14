import argparse
import sys
import os

# Ensure src is in path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_loader import Market1501Dataset
from reid_agent import MockReIDAgent, OpenAIReIDAgent, RandomReIDAgent, LangChainReIDAgent, QwenReIDAgent


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

def run_evaluation(agent_name, data_dir, num_trials=10, gallery_size=10, api_key=None,base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",report_path="report.md"):
    print(f"Loading dataset from {data_dir}...")
    dataset = Market1501Dataset(data_dir)
    
    if agent_name == "mock":
        agent = MockReIDAgent()
    elif agent_name == "openai":
        if not api_key:
            print("Error: API Key needed for OpenAI agent.")
            return
        agent = OpenAIReIDAgent(api_key=api_key,base_url=base_url)
    elif agent_name == "langchain":
        if not api_key:
            print("Error: API Key needed for LangChain agent.")
            return
        agent = LangChainReIDAgent(api_key=api_key,base_url=base_url)
    elif agent_name == "qwen":
        if not api_key:
            print("Error: API Key needed for Qwen agent.")
            return
        # 这里默认调用 qwen3.5-plus
        agent = QwenReIDAgent(api_key=api_key,base_url=base_url)
    elif agent_name == "random":
        agent = RandomReIDAgent()
    else:
        print(f"Unknown agent: {agent_name}")
        return

    print(f"Starting evaluation with {agent_name} agent...")
    print(f"Trials: {num_trials}, Gallery Size: {gallery_size}")
    
    correct_count = 0
    results = []

    for i in range(num_trials):
        # 1. Get case
        query_path, gallery_paths, ground_truth_idx = dataset.get_test_case(gallery_size=gallery_size)
        
        # 2. Predict
        try:
            pred_idx = agent.predict(query_path, gallery_paths)
        except Exception as e:
            print(f"Prediction failed: {e}")
            pred_idx = -1
        
        # 3. Evaluate
        is_correct = (pred_idx == ground_truth_idx)
        if is_correct:
            correct_count += 1
        
        # Log
        results.append({
            "query": os.path.basename(query_path),
            "gallery_size": len(gallery_paths),
            "ground_truth_idx": ground_truth_idx,
            "predicted_idx": pred_idx,
            "correct": is_correct
        })
        
        print(f"Trial {i+1}/{num_trials}: {'Correct' if is_correct else 'Incorrect'} (GT: {ground_truth_idx}, Pred: {pred_idx})")

    accuracy = correct_count / num_trials
    print("="*30)
    print(f"Final Rank-1 Accuracy: {accuracy:.2%}")
    print("="*30)
    
    generate_report(results, accuracy, agent_name, report_path)
    print(f"Report saved to {report_path}")
    
    return accuracy, results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default=r"/mnt/l/CV/data/1/Market-1501-v15.09.15")
    parser.add_argument("--agent", type=str, default="mock", 
                        choices=["mock", "openai", "langchain", "random", "qwen"])
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument("--gallery_size", type=int, default=10)
    parser.add_argument("--base_url",type=str, default="https://dashscope.aliyuncs.com/compatible-mode/v1")
    parser.add_argument("--api_key", type=str, default="sk-1941059ddd9e496381a7fd1a72fd3666")
    parser.add_argument("--report", type=str, default="report.md")
    
    args = parser.parse_args()
    
    run_evaluation(args.agent, args.data_dir, args.trials, args.gallery_size, args.api_key,args.base_url, args.report)
