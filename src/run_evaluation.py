import argparse
import sys
import os
import yaml
import logging
from datetime import datetime

# Ensure src is in path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_loader import Market1501Dataset
from reid_agent import MockReIDAgent, OpenAIReIDAgent, RandomReIDAgent, LangChainReIDAgent, EvolutionaryReIDAgent
from memory_plugin import MemoryPlugin


def setup_logging(query_size, gallery_size, trials, log_dir=None):
    """
    配置 logging：
    - 控制台：仅输出 INFO 及以上（trial 进度、最终结果）
    - 文件：输出全部 DEBUG 日志（模型响应、reflector 输出等）
    文件名格式：YYYYMMDD_HHMMSS_q{query_size}_g{gallery_size}_t{trials}.log
    """
    if log_dir is None:
        # 默认放在 src/ 同级的 logs/ 目录
        log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "logs")
    os.makedirs(log_dir, exist_ok=True)

    timestamp  = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_name   = f"{timestamp}_q{query_size}_g{gallery_size}_t{trials}.log"
    log_path   = os.path.join(log_dir, log_name)

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)

    # 控制台：INFO+（干净输出）
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter("%(message)s"))

    # 文件：DEBUG（完整日志）
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(
        "[%(asctime)s][%(levelname)s] %(message)s", datefmt="%H:%M:%S"
    ))

    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)

    logging.info(f"Log file → {log_path}")
    return log_path


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


def run_evaluation(agent_name, data_dir, num_trials=10, gallery_size=10, query_size=1,
                   api_key=None, report_path="report.md", model="gpt-4o",
                   base_url=None, use_memory=True):

    logging.info(f"Loading dataset from {data_dir}...")
    dataset = Market1501Dataset(data_dir)

    if agent_name == "mock":
        agent = MockReIDAgent()
    elif agent_name == "openai":
        if not api_key:
            logging.error("API Key needed for OpenAI agent.")
            return
        agent = OpenAIReIDAgent(api_key=api_key, model=model, base_url=base_url)
    elif agent_name == "langchain":
        if not api_key:
            logging.error("API Key needed for LangChain agent.")
            return
        agent = LangChainReIDAgent(api_key=api_key, model=model, base_url=base_url)
    elif agent_name == "random":
        agent = RandomReIDAgent()
    elif agent_name == "evo":
        if not api_key:
            logging.error("API Key needed for Evolutionary agent.")
            return
        backend = config.get("backend", "openai")
        agent = EvolutionaryReIDAgent(api_key=api_key, model=model, base_url=base_url,
                                      backend=backend, use_memory=use_memory)
    else:
        logging.error(f"Unknown agent: {agent_name}")
        return

    logging.info(f"Starting evaluation with {agent_name} agent...")
    logging.info(f"Trials: {num_trials}, Gallery Size: {gallery_size}, Query Size: {query_size}")

    workspace_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if isinstance(agent, EvolutionaryReIDAgent):
        persistent_memory = agent.persistent_memory
    else:
        persistent_memory = MemoryPlugin(enabled=use_memory, workspace_path=workspace_path)

    correct_count = 0
    results = []

    for i in range(num_trials):
        query_paths, gallery_paths, ground_truth_idx = dataset.get_test_case(
            gallery_size=gallery_size, query_size=query_size
        )

        try:
            if isinstance(agent, EvolutionaryReIDAgent):
                pred_idx = agent.predict(query_paths, gallery_paths, ground_truth_idx=ground_truth_idx)
            else:
                pred_idx = agent.predict(query_paths, gallery_paths)
        except Exception as e:
            logging.warning(f"Prediction failed: {e}")
            pred_idx = -1

        is_correct = (pred_idx == ground_truth_idx)
        if is_correct:
            correct_count += 1

        file_names = [os.path.basename(q) for q in query_paths]
        results.append({
            "query":           ", ".join(file_names),
            "gallery_size":    len(gallery_paths),
            "ground_truth_idx": ground_truth_idx,
            "predicted_idx":   pred_idx,
            "correct":         is_correct
        })

        # evo agent 在 predict() 内部已用真实 MCTS Prompt 调用了 add_trial，
        # 此处只对其他 agent 补充记录，避免写入无意义的 agent 名称。
        if not isinstance(agent, EvolutionaryReIDAgent):
            persistent_memory.add_trial(
                prompt=agent_name,
                is_correct=is_correct,
                analysis=f"Pred: {pred_idx}, GT: {ground_truth_idx}"
            )

        status = "Correct ✅" if is_correct else "Incorrect ❌"
        logging.info(f"Trial {i+1:>3}/{num_trials}: {status}  (GT: {ground_truth_idx}, Pred: {pred_idx})")

    accuracy = correct_count / num_trials
    logging.info("=" * 30)
    logging.info(f"Final Rank-1 Accuracy: {accuracy:.2%}")
    logging.info("=" * 30)

    generate_report(results, accuracy, agent_name, report_path)
    logging.info(f"Report saved to {report_path}")

    best_prompt = agent.journal.best_prompt if isinstance(agent, EvolutionaryReIDAgent) else None
    summary = f"Session [{agent_name}] completed with {accuracy:.2%} accuracy. Total trials: {num_trials}."
    persistent_memory.summarize_to_long_term(summary, best_prompt=best_prompt, accuracy=accuracy)
    logging.info("Progress summarized to long-term memory." + (" Best prompt saved." if best_prompt else ""))

    return accuracy, results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",       type=str, default="config.yaml")
    parser.add_argument("--data_dir",     type=str, default=None)
    parser.add_argument("--agent",        type=str, default=None,
                        choices=["mock", "openai", "langchain", "random", "evo"])
    parser.add_argument("--trials",       type=int, default=None)
    parser.add_argument("--gallery_size", type=int, default=None)
    parser.add_argument("--query_size",   type=int, default=None)
    parser.add_argument("--api_key",      type=str, default=None)
    parser.add_argument("--report",       type=str, default="report.md")
    parser.add_argument("--log_dir",      type=str, default=None,
                        help="日志目录，默认为项目根目录下的 logs/")

    args = parser.parse_args()

    # Load config
    config = {}
    config_path = args.config
    if not os.path.exists(config_path):
        parent_config = os.path.join("..", args.config)
        if os.path.exists(parent_config):
            config_path = parent_config
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

    # CLI args override YAML
    agent        = args.agent        or config.get("agent",       "mock")
    data_dir     = args.data_dir     or config.get("data_dir",    "")
    trials       = args.trials       or config.get("trials",      10)
    gallery_size = args.gallery_size or config.get("gallery_size", 10)
    query_size   = args.query_size   or config.get("query_size",   1)
    api_key      = args.api_key      or config.get("api_key")
    model        = config.get("model",    "gpt-4o")
    base_url     = config.get("base_url")
    use_memory   = config.get("use_memory", True)
    report_path  = args.report if args.report != "report.md" else config.get("report", "report.md")

    # 设置 logging（必须在所有 print/logging 调用之前）
    setup_logging(query_size, gallery_size, trials, log_dir=args.log_dir)

    run_evaluation(agent, data_dir, trials, gallery_size, query_size,
                   api_key, report_path, model=model, base_url=base_url,
                   use_memory=use_memory)
