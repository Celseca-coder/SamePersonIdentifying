import base64
import random
import os
import logging
from PIL import Image
import io
import dashscope
import time
import re
import math
from functools import wraps
from memory_plugin import MemoryPlugin

# --- AIDEML Inspired MCTS & Memory Components ---

class SearchNode:
    def __init__(self, prompt, parent=None):
        self.prompt = prompt
        self.parent = parent
        self.children = []
        self.visits = 0
        self.reward = 0.0
        self.performance_history = []

    def uct_score(self, C=1.414):
        if self.visits == 0:
            return float('inf')
        exploitation = self.reward / self.visits
        exploration = C * math.sqrt(math.log(self.parent.visits) / self.visits) if self.parent else 0
        return exploitation + exploration

class Journal:
    def __init__(self):
        self.nodes = []
        self.best_reward = -1.0
        self.best_prompt = None
        self.root = None

    def add_node(self, prompt, parent=None):
        node = SearchNode(prompt, parent)
        if parent:
            parent.children.append(node)
        else:
            self.root = node
        self.nodes.append(node)
        return node

    def update_mcts(self, node, reward):
        curr = node
        while curr:
            curr.visits += 1
            curr.reward += reward
            curr = curr.parent
        if reward > self.best_reward:
            self.best_reward = reward
            self.best_prompt = node.prompt

class MemoryManager:
    def __init__(self):
        self.success_memory = [] # Store successful prompt snippets/strategies
        self.failure_memory = [] # Store cases where model failed

    def add_experience(self, prompt, result, ground_truth):
        if result == ground_truth:
            self.success_memory.append(f"Prompt worked for ID {ground_truth}")
        else:
            self.failure_memory.append(f"Prompt failed: Guessed {result} instead of {ground_truth}")

    def get_contextual_guidance(self):
        # Summarize guidance for the LLM
        guidance = ""
        if self.success_memory:
            guidance += "\nPast successful strategies: " + "; ".join(self.success_memory[-3:])
        if self.failure_memory:
            guidance += "\nAvoid these mistakes: " + "; ".join(self.failure_memory[-3:])
        return guidance

def retry_on_exception(max_retries=3, delay=2, default=-1):
    """装饰器：失败重试，全部失败后返回 default（可自定义，如 (-1, "") 用于 predict_with_thinking）。"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    logging.debug(f"[Retry] Attempt {attempt + 1} failed: {e}")
                    if attempt < max_retries - 1:
                        time.sleep(delay)
            logging.debug(f"All {max_retries} attempts failed.")
            return default
        return wrapper
    return decorator

class BaseReIDAgent:
    def predict(self, query_path, gallery_paths):
        raise NotImplementedError

class MockReIDAgent(BaseReIDAgent):
    def predict(self, query_paths, gallery_paths):
        # 兼容多查询，取第一个查询图片的 ID
        query_path = query_paths[0] if isinstance(query_paths, list) else query_paths
        q_pid = int(os.path.basename(query_path).split('_')[0])
        for i, path in enumerate(gallery_paths):
            g_pid = int(os.path.basename(path).split('_')[0])
            if g_pid == q_pid:
                return i
        return 0

class RandomReIDAgent(BaseReIDAgent):
    def predict(self, query_paths, gallery_paths):
        return random.randint(0, len(gallery_paths) - 1)

class LangChainReIDAgent(BaseReIDAgent):
    def __init__(self, api_key, model="gpt-4o", base_url=None):
        self.api_key = api_key
        self.model = model
        if api_key:
            os.environ["OPENAI_API_KEY"] = api_key
        try:
            from langchain_openai import ChatOpenAI
            # 兼容多种版本的 api_key 参数名
            self.llm = ChatOpenAI(
                model=model, 
                api_key=api_key, 
                openai_api_key=api_key, 
                base_url=base_url,
                temperature=0.0
            )
        except ImportError:
            logging.warning("Please install langchain-openai: pip install langchain-openai")
            self.llm = None

    def _encode_image(self, image_path):
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode('utf-8')

    def _build_content(self, query_paths, gallery_paths, custom_prompt=None):
        """构建发给 LLM 的 multimodal content 列表（predictor 和 reflector 共用）。"""
        text_prompt = custom_prompt or (
            "Identify the person in Query Image(s) from Gallery. "
            "Think step by step, then end with ONLY the index number."
        )
        content = [{"type": "text", "text": text_prompt}]
        for i, q_path in enumerate(query_paths):
            content.append({"type": "text", "text": f"Query Image {i}:"})
            content.append({"type": "image_url", "image_url": {
                "url": f"data:image/jpeg;base64,{self._encode_image(q_path)}"
            }})
        content.append({"type": "text", "text": "Gallery Candidates:"})
        for i, path in enumerate(gallery_paths):
            content.append({"type": "text", "text": f"Index {i}:"})
            content.append({"type": "image_url", "image_url": {
                "url": f"data:image/jpeg;base64,{self._encode_image(path)}"
            }})
        return content

    @retry_on_exception(max_retries=3, delay=3)
    def predict(self, query_paths, gallery_paths, custom_prompt=None):
        """仅返回预测索引（兼容旧接口）。"""
        idx, _ = self.predict_with_thinking(query_paths, gallery_paths, custom_prompt)
        return idx

    @retry_on_exception(max_retries=3, delay=3, default=(-1, ""))
    def predict_with_thinking(self, query_paths, gallery_paths, custom_prompt=None):
        """
        Predictor 角色：返回 (predicted_index, thinking_text)。
        thinking_text 是模型的完整推理过程，供 reflector 分析。
        """
        if not self.llm:
            return -1, ""
        if not isinstance(query_paths, list):
            query_paths = [query_paths]

        content = self._build_content(query_paths, gallery_paths, custom_prompt)
        from langchain_core.messages import HumanMessage
        response = self.llm.invoke([HumanMessage(content=content)])
        thinking = response.content
        match = re.search(r'\d+', thinking)
        idx = int(match.group()) if match else -1
        return idx, thinking

class OpenAIReIDAgent(BaseReIDAgent):
    def __init__(self, api_key, model="gpt-4o", base_url=None):
        self.api_key = api_key
        self.model = model
        if api_key:
            os.environ["OPENAI_API_KEY"] = api_key
        try:
            from openai import OpenAI
            # 统一加入 base_url
            self.client = OpenAI(api_key=api_key, base_url=base_url)
        except ImportError:
            logging.warning("Please install openai: pip install openai")
            self.client = None

    def _encode_image(self, image_path):
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode('utf-8')

    def _parse_index(self, content: str) -> int:
        """从模型响应文本中解析出预测的 gallery 索引。"""
        import json as _json
        try:
            for candidate in reversed(re.findall(r'\{[^{}]*\}', content)):
                try:
                    data = _json.loads(candidate)
                    if "index" in data:
                        return int(data["index"])
                except Exception:
                    continue
        except Exception:
            pass
        m = re.search(r'"index"\s*:\s*(\d+)', content, re.IGNORECASE)
        if m:
            return int(m.group(1))
        m = re.search(r'\d+', content)
        return int(m.group()) if m else -1

    def _build_messages(self, query_paths, gallery_paths):
        """构建发给 OpenAI API 的 messages 列表。"""
        if not isinstance(query_paths, list):
            query_paths = [query_paths]
        query_content = []
        for i, q_path in enumerate(query_paths):
            query_content.extend([
                {"type": "text", "text": f"Query Image {i}:"},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{self._encode_image(q_path)}"}}
            ])
        gallery_content = []
        for i, path in enumerate(gallery_paths):
            gallery_content.extend([
                {"type": "text", "text": f"Image {i}:"},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{self._encode_image(path)}"}}
            ])
        return [
            {
                "role": "system",
                "content": (
                    "You are an expert in Person Re-Identification (ReID). "
                    "Your task is to identify the SAME person from a gallery of candidates based on a query image.\n"
                    "Follow these steps strictly:\n"
                    "1. Analyze the Query Image: Describe upper clothing, lower clothing, shoes, and accessories.\n"
                    "2. Compare with Gallery: Check each candidate against these features.\n"
                    "3. Eliminate Distractors: Ignore people with different clothing colors or styles.\n"
                    "4. Decision: Return JSON: {\"analysis\": \"...\", \"index\": <number>}."
                )
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Query Images (Look for this person, multiple angles provided):"},
                    *query_content,
                    {"type": "text", "text": "Gallery Candidates:"},
                    *gallery_content,
                    {"type": "text", "text": "Which index matches the Query person? Think step by step."}
                ]
            }
        ]

    @retry_on_exception(max_retries=3, delay=3)
    def predict(self, query_paths, gallery_paths):
        """仅返回预测索引（兼容旧接口）。"""
        idx, _ = self._raw_call(query_paths, gallery_paths)
        return idx

    def _raw_call(self, query_paths, gallery_paths):
        """核心 API 调用，返回 (index, thinking)，由 predict / predict_with_thinking 共用。"""
        if not self.client:
            return -1, ""
        messages = self._build_messages(query_paths, gallery_paths)
        response = self.client.chat.completions.create(
            model=self.model, messages=messages, max_tokens=1024, temperature=0.0
        )
        if isinstance(response, str):
            logging.debug(f"[DEBUG] API Error: got string response: {response[:200]}...")
            return -1, ""
        content = response.choices[0].message.content.strip()
        logging.debug(f"[DEBUG] Model Response: {content}")
        return self._parse_index(content), content

    @retry_on_exception(max_retries=3, delay=3, default=(-1, ""))
    def predict_with_thinking(self, query_paths, gallery_paths):
        """Predictor 角色：返回 (predicted_index, thinking_text)。"""
        return self._raw_call(query_paths, gallery_paths)

class QwenReIDAgent(BaseReIDAgent):
    def __init__(self, api_key, model="qwen3.5-plus", base_url=None):
        self.api_key = api_key
        self.model = model
        dashscope.api_key = api_key
        # Only set base_url if it is not the OpenAI compatible one, as dashscope SDK uses native endpoint
        if base_url and "compatible-mode" not in base_url:
            dashscope.base_http_api_url = base_url

    @retry_on_exception(max_retries=3, delay=3)
    def _call_api(self, query_paths, gallery_paths):
        """调用 Qwen API，返回 (index, thinking_text)。"""
        if not isinstance(query_paths, list):
            query_paths = [query_paths]
        content = [{"text": "Identify the person in Query Images from Gallery. Think step by step, then return ONLY the index number."}]
        for i, q_path in enumerate(query_paths):
            content.append({"text": f"Query Image {i}:"})
            content.append({"image": f"file://{os.path.abspath(q_path)}"})
        content.append({"text": "Gallery below:"})
        for i, path in enumerate(gallery_paths):
            content.append({"text": f"Index {i}:"})
            content.append({"image": f"file://{os.path.abspath(path)}"})
        response = dashscope.MultiModalConversation.call(
            model=self.model,
            messages=[{"role": "user", "content": content}]
        )
        if response.status_code == 200:
            result_text = response.output.choices[0].message.content[0]['text']
            logging.debug(f"[DEBUG] Qwen Response: {result_text}")
            match = re.search(r'\d+', result_text)
            idx = int(match.group()) if match else -1
            return idx, result_text
        else:
            raise Exception(f"Qwen API Error: {response.code} - {response.message}")

    @retry_on_exception(max_retries=3, delay=3)
    def predict(self, query_paths, gallery_paths):
        """仅返回预测索引（兼容旧接口）。"""
        idx, _ = self._call_api(query_paths, gallery_paths)
        return idx

    @retry_on_exception(max_retries=3, delay=3, default=(-1, ""))
    def predict_with_thinking(self, query_paths, gallery_paths):
        """返回 (predicted_index, thinking_text)，供 reflector 使用。"""
        return self._call_api(query_paths, gallery_paths)

class EvolutionaryReIDAgent(BaseReIDAgent):
    """
    Advanced ReID Agent implementing AIDEML-inspired:
    - MCTS (Monte Carlo Tree Search) for prompt selection
    - Fireworks Policy for 'exploding' (variation) best prompts
    - Memory Management for guidance
    """
    def __init__(self, api_key, model="gpt-4o", base_url=None, backend="openai", use_memory=True):
        self.journal = Journal()
        self.memory = MemoryManager()
        
        # 使用 MemoryPlugin 支持热插拔
        workspace_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.persistent_memory = MemoryPlugin(enabled=use_memory, workspace_path=workspace_path)
        
        self.api_key = api_key
        self.model = model
        
        # 根据 backend 选择基础 Agent
        if backend == "langchain":
            self.base_agent = LangChainReIDAgent(api_key, model, base_url)
        elif backend == "qwen":
            self.base_agent = QwenReIDAgent(api_key, model, base_url)
        else:
            self.base_agent = OpenAIReIDAgent(api_key, model, base_url)
        
        # Initial seed prompt
        initial_prompt = """Analyze clothing features (upper/lower/shoes) and match the Query to one Gallery index."""
        self.journal.add_node(initial_prompt)

    def _safe_encode(self, path):
        """安全图片编码，出错时返回 None。"""
        try:
            if hasattr(self.base_agent, '_encode_image'):
                return self.base_agent._encode_image(path)
        except Exception:
            pass
        return None

    def _call_reflector(self, fp_case, fn_case):
        """
        Reflector 角色：接收 FP/FN 错误案例和 predictor 的 thinking 过程，
        输出两条 Prompt 建议——一条避免认错人（FP），一条避免漏人（FN）。
        返回 (fp_fix, fn_fix)，任一可为 None。
        """
        if not hasattr(self.base_agent, 'llm') or not self.base_agent.llm:
            return None, None

        content = [{
            "type": "text",
            "text": (
                "You are a ReID expert reviewing prediction errors to improve future prompts.\n"
                "Below are two error cases. For each, suggest ONE concise prompt sentence to fix it.\n"
            )
        }]

        def _append_image(path, label):
            enc = self._safe_encode(path)
            if enc:
                content.append({"type": "text", "text": label})
                content.append({"type": "image_url",
                                 "image_url": {"url": f"data:image/jpeg;base64,{enc}"}})

        # FP 案例：认错人
        if fp_case:
            content.append({"type": "text", "text": (
                f"\n## False Positive — Model matched the WRONG person\n"
                f"Predictor's thinking: {fp_case.get('thinking', 'N/A')[:300]}"
            )})
            for q in (fp_case.get("query_paths") or [])[:1]:
                _append_image(q, "Query:")
            if fp_case.get("wrong_path"):
                _append_image(fp_case["wrong_path"], "Incorrectly matched (wrong person selected):")

        # FN 案例：漏人
        if fn_case:
            content.append({"type": "text", "text": (
                f"\n## False Negative — Model MISSED the correct person\n"
                f"Predictor's thinking: {fn_case.get('thinking', 'N/A')[:300]}"
            )})
            for q in (fn_case.get("query_paths") or [])[:1]:
                _append_image(q, "Query:")
            if fn_case.get("correct_path"):
                _append_image(fn_case["correct_path"], "Missed correct match:")

        content.append({"type": "text", "text": (
            "\nRespond in exactly this format (no extra text):\n"
            "FP_FIX: <one sentence to avoid selecting visually similar but wrong people>\n"
            "FN_FIX: <one sentence to avoid missing the correct person>"
        )})

        try:
            from langchain_core.messages import HumanMessage
            response = self.base_agent.llm.invoke([HumanMessage(content=content)])
            text = response.content.strip()
            logging.debug(f"[Reflector] {text}")

            fp_fix = fn_fix = None
            for line in text.split('\n'):
                line = line.strip()
                if line.upper().startswith("FP_FIX:"):
                    fp_fix = line[7:].strip()
                elif line.upper().startswith("FN_FIX:"):
                    fn_fix = line[7:].strip()
            return fp_fix, fn_fix
        except Exception as e:
            logging.debug(f"[Reflector] Error: {e}")
            return None, None

    def _get_evolved_prompt(self, base_node):
        """Fireworks/Explosion: Generate variations of the best prompt using the LLM itself or simple logic"""
        # 获取持久化记忆的压缩上下文
        memory_context = self.persistent_memory.get_compressed_context()
        guidance = self.memory.get_contextual_guidance()
        
        # 模拟根据记忆进化的 Prompt
        variations = [
            f"{base_node.prompt} (Context: {memory_context}) | Strategy: Focus on tiny unique textures.",
            f"{base_node.prompt} (Context: {memory_context}) | Strategy: Pay attention to bag shapes/straps.",
            f"{base_node.prompt} (Context: {memory_context}) | Strategy: Cross-check footwear color and type."
        ]
        return random.choice(variations)

    def _select_node(self):
        """MCTS selection based on UCT"""
        if not self.journal.nodes:
            return None
        # Basic UCT selection among processed nodes
        best_node = max(self.journal.nodes, key=lambda n: n.uct_score())
        return best_node

    def predict(self, query_paths, gallery_paths, ground_truth_idx=None):
        # 兼容单路径输入
        if not isinstance(query_paths, list):
            query_paths = [query_paths]

        # ── 1. MCTS Selection ──────────────────────────────────────────
        parent_node = self._select_node()

        # ── 2. Fireworks Expansion ─────────────────────────────────────
        if parent_node and parent_node.visits > 0:
            new_prompt = self._get_evolved_prompt(parent_node)
            current_node = self.journal.add_node(new_prompt, parent=parent_node)
        else:
            current_node = parent_node

        eval_prompt = current_node.prompt if current_node else None

        # ── 3. Predictor：执行 ReID，同时获取 thinking 过程 ───────────
        import inspect

        def _supports_custom_prompt(agent_method):
            """检查方法签名是否接受 custom_prompt 参数。"""
            try:
                sig = inspect.signature(agent_method)
                return "custom_prompt" in sig.parameters
            except (ValueError, TypeError):
                return False

        thinking = ""
        try:
            if hasattr(self.base_agent, 'predict_with_thinking'):
                pwt = self.base_agent.predict_with_thinking
                if _supports_custom_prompt(pwt):
                    prediction, thinking = pwt(query_paths, gallery_paths, custom_prompt=eval_prompt)
                else:
                    prediction, thinking = pwt(query_paths, gallery_paths)
            elif _supports_custom_prompt(self.base_agent.predict):
                prediction = self.base_agent.predict(query_paths, gallery_paths, custom_prompt=eval_prompt)
            else:
                prediction = self.base_agent.predict(query_paths, gallery_paths)
        except Exception as e:
            logging.warning(f"[Predictor] Failed: {e}")
            prediction = -1

        # ── 4. Backpropagation & Memory Update ────────────────────────
        if ground_truth_idx is not None:
            is_correct = (prediction == ground_truth_idx)
            reward = 1.0 if is_correct else 0.0

            if current_node:
                self.journal.update_mcts(current_node, reward)

            self.memory.add_experience(
                current_node.prompt if current_node else "Initial",
                prediction, ground_truth_idx
            )
            self.persistent_memory.add_trial(
                prompt=current_node.prompt if current_node else "Initial",
                is_correct=is_correct,
                analysis=f"Pred: {prediction}, GT: {ground_truth_idx}"
            )

            # ── 5. Reflector：仅在 predictor 判错时触发 ───────────────
            if not is_correct:
                wrong_path   = (gallery_paths[prediction]
                                if isinstance(prediction, int) and 0 <= prediction < len(gallery_paths)
                                else None)
                correct_path = (gallery_paths[ground_truth_idx]
                                if 0 <= ground_truth_idx < len(gallery_paths)
                                else None)

                # 将本次错误案例写入短时记忆
                self.persistent_memory.add_error_case(
                    query_paths=query_paths,
                    wrong_path=wrong_path,
                    correct_path=correct_path,
                    thinking=thinking
                )

                # 随机抽取 1 个 FP 案例 + 1 个 FN 案例进行反思
                fp_case, fn_case = self.persistent_memory.sample_reflection_cases()
                fp_fix, fn_fix = self._call_reflector(fp_case, fn_case)

                # 将反思结果持久化到长时记忆，供下一次 Prompt 进化使用
                if fp_fix or fn_fix:
                    self.persistent_memory.add_reflection(fp_fix, fn_fix)

        return prediction
