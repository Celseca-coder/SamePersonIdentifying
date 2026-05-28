import argparse
import sys
import os
import yaml

# Ensure src is in path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_loader import Market1501Dataset
from reid_agent import MockReIDAgent, OpenAIReIDAgent, RandomReIDAgent, LangChainReIDAgent, EvolutionaryReIDAgent
from memory_plugin import MemoryPlugin

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

def run_evaluation(agent_name, data_dir, num_trials=10, gallery_size=10, query_size=1, api_key=None, report_path="report.md", model="gpt-4o", base_url=None, use_memory=True):
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
    elif agent_name == "evo":
        if not api_key:
            print("Error: API Key needed for Evolutionary agent.")
            return
        # You can now specify backend="langchain" to use LangChain inside EVO
        agent = EvolutionaryReIDAgent(api_key=api_key, model=model, base_url=base_url, backend="langchain", use_memory=use_memory)
    else:
        print(f"Unknown agent: {agent_name}")
        return

    print(f"Starting evaluation with {agent_name} agent...")
    print(f"Trials: {num_trials}, Gallery Size: {gallery_size}, Query Size: {query_size}")

    # 无论使用何种 agent，都通过统一的 persistent_memory 记录记忆。
    # evo agent 已内置（热插拔）persistent_memory，其他 agent 在此处创建。
    if isinstance(agent, EvolutionaryReIDAgent):
        persistent_memory = agent.persistent_memory
    else:
        persistent_memory = MemoryPlugin(enabled=use_memory)

    correct_count = 0
    results = []

    for i in range(num_trials):
        # 1. Get case
        query_paths, gallery_paths, ground_truth_idx = dataset.get_test_case(gallery_size=gallery_size, query_size=query_size)
        
        # 2. Predict
        try:
            if isinstance(agent, EvolutionaryReIDAgent):
                pred_idx = agent.predict(query_paths, gallery_paths, ground_truth_idx=ground_truth_idx)
            else:
                pred_idx = agent.predict(query_paths, gallery_paths)
        except Exception as e:
            print(f"Prediction failed: {e}")
            pred_idx = -1
        
        # 3. Evaluate
        is_correct = (pred_idx == ground_truth_idx)
        if is_correct:
            correct_count += 1
        
        # Log
        file_names = [os.path.basename(q) for q in query_paths]
        results.append({
            "query": ", ".join(file_names),
            "gallery_size": len(gallery_paths),
            "ground_truth_idx": ground_truth_idx,
            "predicted_idx": pred_idx,
            "correct": is_correct
        })
        
        # 所有 agent 类型统一记录每次 trial 到短时记忆
        persistent_memory.add_trial(
            prompt=agent_name,
            is_correct=is_correct,
            analysis=f"Pred: {pred_idx}, GT: {ground_truth_idx}"
        )

        print(f"Trial {i+1}/{num_trials}: {'Correct' if is_correct else 'Incorrect'} (GT: {ground_truth_idx}, Pred: {pred_idx})")

    accuracy = correct_count / num_trials
    print("="*30)
    print(f"Final Rank-1 Accuracy: {accuracy:.2%}")
    print("="*30)
    
    generate_report(results, accuracy, agent_name, report_path)
    print(f"Report saved to {report_path}")

    # 所有 agent 类型在任务结束时统一将总结写入长时记忆
    # evo agent 额外传入本次搜索到的最佳 Prompt
    best_prompt = agent.journal.best_prompt if isinstance(agent, EvolutionaryReIDAgent) else None
    summary = f"Session [{agent_name}] completed with {accuracy:.2%} accuracy. Total trials: {num_trials}."
    persistent_memory.summarize_to_long_term(summary, best_prompt=best_prompt)
    print(f"Progress summarized to long-term memory." + (f" Best prompt saved." if best_prompt else ""))
    
    return accuracy, results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--agent", type=str, default=None, choices=["mock", "openai", "langchain", "random", "evo"])
    parser.add_argument("--trials", type=int, default=None)
    parser.add_argument("--gallery_size", type=int, default=None)
    parser.add_argument("--query_size", type=int, default=None)
    parser.add_argument("--api_key", type=str, default=None)
    parser.add_argument("--report", type=str, default="report.md")
    
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
    query_size = args.query_size or config.get("query_size", 1)
    api_key = args.api_key or config.get("api_key")
    model = config.get("model", "gpt-4o")
    base_url = config.get("base_url")
    use_memory = config.get("use_memory", True)
    report_path = args.report # Usually specified via CLI or default

    run_evaluation(agent, data_dir, trials, gallery_size, query_size, api_key, report_path, model=model, base_url=base_url, use_memory=use_memory)

