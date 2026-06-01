"""
memory_plugin.py — 记忆模块热插拔封装

通过 config 中的 use_memory 字段控制是否启用持久化记忆。
调用方代码无需感知是否启用，所有方法在禁用时均为 no-op，
不创建任何文件也不产生任何副作用。

典型用法：
    from memory_plugin import MemoryPlugin

    # 从 config dict 构造（推荐）
    mem = MemoryPlugin.from_config(config)

    # 手动构造
    mem = MemoryPlugin(enabled=True)

    # 调用方式与 ReIDMemoryModule 完全一致
    mem.add_trial(prompt="...", is_correct=True, analysis="...")
    mem.summarize_to_long_term("session summary", best_prompt="...")
    ctx = mem.get_compressed_context()
"""

import os
from memory_module import ReIDMemoryModule


class MemoryPlugin:
    """
    ReIDMemoryModule 的热插拔包装层。

    - enabled=True ：代理到 ReIDMemoryModule，行为与原来完全一致。
    - enabled=False：所有方法均为 no-op，不写入任何文件。
    """

    def __init__(self, enabled: bool = True, workspace_path: str = None):
        self.enabled = enabled
        self._module: ReIDMemoryModule | None = None

        if enabled:
            if workspace_path is not None:
                self._module = ReIDMemoryModule(workspace_path=workspace_path)
            else:
                self._module = ReIDMemoryModule()
        else:
            print("[MemoryPlugin] Memory saving disabled.")

    # ------------------------------------------------------------------
    # 工厂方法
    # ------------------------------------------------------------------

    @classmethod
    def from_config(cls, config: dict, workspace_path: str = None) -> "MemoryPlugin":
        """
        从 config dict 中读取 use_memory 字段构造实例。
        config 中没有该字段时默认 True（向后兼容）。
        """
        enabled = bool(config.get("use_memory", True))
        return cls(enabled=enabled, workspace_path=workspace_path)

    # ------------------------------------------------------------------
    # 与 ReIDMemoryModule 完全相同的公共接口
    # （disabled 时全部为 no-op，调用方无需任何 if 判断）
    # ------------------------------------------------------------------

    def add_trial(self, prompt: str, is_correct: bool, analysis: str) -> None:
        """记录单次尝试到短时记忆。"""
        if self.enabled and self._module:
            self._module.add_trial(prompt, is_correct, analysis)

    def summarize_to_long_term(
        self,
        session_summary: str,
        best_prompt: str = None,
        accuracy: float = None,
        success_threshold: float = 0.5,
    ) -> None:
        """将 session 总结写入长时记忆。
        accuracy >= success_threshold 归入成功经验，否则归入失败经验。"""
        if self.enabled and self._module:
            self._module.summarize_to_long_term(
                session_summary,
                best_prompt=best_prompt,
                accuracy=accuracy,
                success_threshold=success_threshold,
            )

    def get_compressed_context(self) -> str:
        """生成传递给 Agent 的压缩记忆上下文；未启用时返回空字符串。"""
        if self.enabled and self._module:
            return self._module.get_compressed_context()
        return ""

    def add_error_case(
        self,
        query_paths: list,
        wrong_path: str,
        correct_path: str,
        thinking: str = ""
    ) -> None:
        """记录判错案例（FP wrong_path + FN correct_path + thinking）。"""
        if self.enabled and self._module:
            self._module.add_error_case(query_paths, wrong_path, correct_path, thinking)

    def sample_reflection_cases(self):
        """随机抽取 1 个 FP 和 1 个 FN 案例；未启用时返回 (None, None)。"""
        if self.enabled and self._module:
            return self._module.sample_reflection_cases()
        return None, None

    def add_reflection(self, fp_reflection: str = None, fn_reflection: str = None) -> None:
        """将 reflector 生成的 Prompt 建议写入长时记忆。"""
        if self.enabled and self._module:
            self._module.add_reflection(fp_reflection, fn_reflection)