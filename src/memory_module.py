import json
import os

class ReIDMemoryModule:
    def __init__(self, workspace_path=None):
        # 默认使用本文件所在目录，确保无论从哪里启动脚本，
        # 记忆文件都写入 src/memory/，而非随工作目录漂移
        if workspace_path is None:
            workspace_path = os.path.dirname(os.path.abspath(__file__))

        memory_dir = os.path.join(workspace_path, "memory")
        if not os.path.exists(memory_dir):
            os.makedirs(memory_dir)

        self.short_term_path = os.path.join(memory_dir, "short_term_memory.json")
        self.long_term_path = os.path.join(memory_dir, "long_term_memory.json")

        self.short_term = self._load_json(self.short_term_path, {"trials": [], "current_patterns": []})
        self.long_term = self._load_json(
            self.long_term_path,
            {"global_experiences": [], "high_level_wisdom": "", "best_prompts": []}
        )

    def _load_json(self, path, default):
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            # 兼容旧格式：补充缺失字段，避免 KeyError
            for key, val in default.items():
                if key not in data:
                    data[key] = val
            return data
        return default

    def _save_json(self, path, data):
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4, ensure_ascii=False)

    def add_trial(self, prompt, is_correct, analysis):
        """记录单次尝试到短时记忆"""
        self.short_term["trials"].append({
            "prompt_summary": prompt[:50] + "...",
            "is_correct": is_correct,
            "analysis_highlights": analysis[:100] if analysis else ""
        })
        self._save_json(self.short_term_path, self.short_term)

    def summarize_to_long_term(self, session_summary, best_prompt=None):
        """将当前 Session 的总结存入长时记忆，可选同时持久化本次最佳 Prompt"""
        self.long_term["global_experiences"].append(session_summary)
        # 维护长时记忆长度上限
        if len(self.long_term["global_experiences"]) > 10:
            self.long_term["global_experiences"].pop(0)

        # 持久化本次 session 的最佳 Prompt（保留最近 5 条）
        if best_prompt:
            self.long_term["best_prompts"].append(best_prompt)
            if len(self.long_term["best_prompts"]) > 5:
                self.long_term["best_prompts"].pop(0)

        self._save_json(self.long_term_path, self.long_term)

        # 清空短时记忆
        self.short_term = {"trials": [], "current_patterns": []}
        self._save_json(self.short_term_path, self.short_term)

    def get_compressed_context(self):
        """生成传递给 Agent 的压缩记忆上下文"""
        context = "### Working Memory (Last few trials highlights):\n"
        recent = self.short_term["trials"][-5:]
        for t in recent:
            status = "Success" if t["is_correct"] else "Failure"
            context += f"- {status}: {t['analysis_highlights']}\n"

        context += "\n### Long-term Wisdom (General ReID Tips):\n"
        for exp in self.long_term["global_experiences"][-3:]:
            context += f"- {exp}\n"

        # 注入历史最佳 Prompt，让新 session 可直接复用高效策略
        if self.long_term.get("best_prompts"):
            context += "\n### Best Prompts from Previous Sessions:\n"
            for bp in self.long_term["best_prompts"][-3:]:
                context += f"- {bp}\n"

        return context
