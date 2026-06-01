import json
import os
import random

# 判断 session 是否"成功"的默认准确率阈值
DEFAULT_SUCCESS_THRESHOLD = 0.5

class ReIDMemoryModule:
    def __init__(self, workspace_path=None):
        # 默认使用本文件所在目录，确保路径不随工作目录漂移
        if workspace_path is None:
            workspace_path = os.path.dirname(os.path.abspath(__file__))

        memory_dir = os.path.join(workspace_path, "memory")
        if not os.path.exists(memory_dir):
            os.makedirs(memory_dir)

        self.short_term_path = os.path.join(memory_dir, "short_term_memory.json")
        self.long_term_path  = os.path.join(memory_dir, "long_term_memory.json")

        _st_default = {
            "trials":           [],   # 每次 trial 的简要记录
            "error_cases":      [],   # 判错案例（含 FP/FN 图路径和 thinking）
            "current_patterns": []
        }
        _lt_default = {
            "success_experiences": [],   # 高于阈值的 session 总结
            "failure_experiences": [],   # 低于阈值的 session 总结
            "fp_reflections":      [],   # reflector 针对 FP（认错人）生成的 prompt 建议
            "fn_reflections":      [],   # reflector 针对 FN（漏人）生成的 prompt 建议
            "best_prompts":        [],   # 历史最佳 Prompt（仅 evo agent 填入）
            "high_level_wisdom":   ""
        }
        self.short_term = self._load_json(self.short_term_path, _st_default)
        self.long_term  = self._load_json(self.long_term_path,  _lt_default)
        self._migrate_legacy_long_term(_lt_default)

    # ------------------------------------------------------------------
    # 内部工具
    # ------------------------------------------------------------------

    def _load_json(self, path, default):
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            # 补齐缺失字段，避免 KeyError
            for k, v in default.items():
                if k not in data:
                    data[k] = v
            return data
        return default

    def _save_json(self, path, data):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)

    def _migrate_legacy_long_term(self, default):
        """将旧版字段迁移到新结构，避免 KeyError，迁移后写盘。"""
        changed = False
        # 旧版 global_experiences → success_experiences
        if "global_experiences" in self.long_term:
            for entry in self.long_term.pop("global_experiences"):
                self.long_term["success_experiences"].append(entry)
            changed = True
        # 补齐所有缺失字段
        for k, v in default.items():
            if k not in self.long_term:
                self.long_term[k] = v
                changed = True
        if changed:
            self._save_json(self.long_term_path, self.long_term)

    @staticmethod
    def _cap(lst, max_len=10):
        """原地保留列表最后 max_len 条，丢弃最旧的。"""
        if len(lst) > max_len:
            del lst[:len(lst) - max_len]

    # ------------------------------------------------------------------
    # 公共接口
    # ------------------------------------------------------------------

    def add_trial(self, prompt: str, is_correct: bool, analysis: str) -> None:
        """记录单次尝试到短时记忆，立即写盘。"""
        self.short_term["trials"].append({
            "prompt_summary":      (prompt[:50] + "...") if len(prompt) > 50 else prompt,
            "is_correct":          is_correct,
            "analysis_highlights": (analysis[:100] if analysis else "")
        })
        self._save_json(self.short_term_path, self.short_term)

    def add_error_case(
        self,
        query_paths: list,
        wrong_path: str,
        correct_path: str,
        thinking: str = ""
    ) -> None:
        """
        记录一次判错案例到短时记忆。
        - wrong_path  : predictor 选中的图（FP 视角：认错人）
        - correct_path: 真正正确的图（FN 视角：漏掉的人）
        - thinking    : predictor 的推理文本
        """
        self.short_term["error_cases"].append({
            "query_paths":  query_paths,
            "wrong_path":   wrong_path,
            "correct_path": correct_path,
            "thinking":     (thinking[:300] if thinking else "")
        })
        self._cap(self.short_term["error_cases"], max_len=20)
        self._save_json(self.short_term_path, self.short_term)

    def sample_reflection_cases(self):
        """
        随机各抽取一个 FP 案例和一个 FN 案例用于 reflector 反思。
        两者可以来自同一条记录（错误案例同时包含 FP 和 FN 视角）。
        返回 (fp_case, fn_case)，无案例时返回 (None, None)。
        """
        cases = self.short_term["error_cases"]
        if not cases:
            return None, None
        fp_case = random.choice(cases)
        fn_case = random.choice(cases)
        return fp_case, fn_case

    def add_reflection(self, fp_reflection: str = None, fn_reflection: str = None) -> None:
        """将 reflector 生成的 Prompt 建议写入长时记忆。"""
        changed = False
        if fp_reflection:
            self.long_term["fp_reflections"].append(fp_reflection)
            self._cap(self.long_term["fp_reflections"], max_len=5)
            changed = True
        if fn_reflection:
            self.long_term["fn_reflections"].append(fn_reflection)
            self._cap(self.long_term["fn_reflections"], max_len=5)
            changed = True
        if changed:
            self._save_json(self.long_term_path, self.long_term)

    def summarize_to_long_term(
        self,
        session_summary: str,
        best_prompt: str = None,
        accuracy: float = None,
        success_threshold: float = DEFAULT_SUCCESS_THRESHOLD
    ) -> None:
        """
        将当前 session 的总结写入长时记忆。

        - accuracy >= success_threshold → success_experiences
        - accuracy <  success_threshold → failure_experiences
        - accuracy 为 None             → 默认归入 success_experiences（向后兼容）
        """
        is_successful = (accuracy >= success_threshold) if accuracy is not None else True

        # session 整体总结写入 success / failure_experiences
        entry = {"summary": session_summary}
        if accuracy is not None:
            entry["accuracy"] = round(accuracy, 4)
        if best_prompt:
            entry["best_prompt"] = best_prompt

        target = "success_experiences" if is_successful else "failure_experiences"
        self.long_term[target].append(entry)
        self._cap(self.long_term[target])

        # 最佳 Prompt 单独维护（最多 5 条）
        if best_prompt:
            self.long_term["best_prompts"].append(best_prompt)
            self._cap(self.long_term["best_prompts"], max_len=5)

        # 无论 session 整体是否成功，把所有单次判错案例都摘要写入 failure_experiences
        for case in self.short_term.get("error_cases", []):
            wrong_name   = os.path.basename(case.get("wrong_path")   or "unknown")
            correct_name = os.path.basename(case.get("correct_path") or "unknown")
            thinking_snippet = (case.get("thinking") or "")[:150]
            failure_entry = {
                "summary":  f"FP matched {wrong_name}, FN missed {correct_name}",
                "thinking": thinking_snippet,
            }
            self.long_term["failure_experiences"].append(failure_entry)
        self._cap(self.long_term["failure_experiences"])

        self._save_json(self.long_term_path, self.long_term)

        # 清空短时记忆（含 error_cases）
        self.short_term = {"trials": [], "error_cases": [], "current_patterns": []}
        self._save_json(self.short_term_path, self.short_term)

    def get_compressed_context(self) -> str:
        """生成传递给 predictor 的压缩记忆上下文。"""
        lines = ["### Working Memory (Last few trials):"]
        for t in self.short_term["trials"][-5:]:
            status = "✅ Success" if t["is_correct"] else "❌ Failure"
            lines.append(f"  - {status}: {t['analysis_highlights']}")

        lines.append("\n### Long-term: Successful Sessions:")
        for exp in self.long_term["success_experiences"][-3:]:
            summary = exp["summary"] if isinstance(exp, dict) else str(exp)
            acc_tag = f" (R1={exp['accuracy']:.0%})" if isinstance(exp, dict) and "accuracy" in exp else ""
            lines.append(f"  - ✅{acc_tag} {summary}")

        lines.append("\n### Long-term: Failure Cases:")
        for exp in self.long_term["failure_experiences"][-5:]:
            summary  = exp.get("summary", str(exp)) if isinstance(exp, dict) else str(exp)
            acc_tag  = f" (R1={exp['accuracy']:.0%})" if isinstance(exp, dict) and "accuracy" in exp else ""
            thinking = exp.get("thinking", "") if isinstance(exp, dict) else ""
            line = f"  - ❌{acc_tag} {summary}"
            if thinking:
                line += f" | thinking: {thinking[:80]}…"
            lines.append(line)

        # reflector 反思结果：注入到 predictor 的 Prompt 中
        if self.long_term.get("fp_reflections"):
            lines.append("\n### Reflector: Avoid False Positives (认错人):")
            for r in self.long_term["fp_reflections"][-2:]:
                lines.append(f"  - ⚠️ {r}")

        if self.long_term.get("fn_reflections"):
            lines.append("\n### Reflector: Catch False Negatives (漏人):")
            for r in self.long_term["fn_reflections"][-2:]:
                lines.append(f"  - 🔍 {r}")

        if self.long_term.get("best_prompts"):
            lines.append("\n### Best Prompts from Previous Sessions:")
            for bp in self.long_term["best_prompts"][-3:]:
                lines.append(f"  - {bp}")

        return "\n".join(lines)
