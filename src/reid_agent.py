import base64
import random
import os
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

def retry_on_exception(max_retries=3, delay=2):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    print(f"\n[Retry] Attempt {attempt + 1} failed: {e}")
                    if attempt < max_retries - 1:
                        time.sleep(delay)
            print(f"All {max_retries} attempts failed.")
            return -1
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
            print("Please install langchain-openai: pip install langchain-openai")
            self.llm = None

    def _encode_image(self, image_path):
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode('utf-8')

    @retry_on_exception(max_retries=3, delay=3)
    def predict(self, query_paths, gallery_paths):
        if not self.llm: return -1
        
        # 兼容多查询
        if not isinstance(query_paths, list):
            query_paths = [query_paths]

        content = [
            {"type": "text", "text": "Identify the person in Query Image from Gallery. Return ONLY the index number."}
        ]
        
        for i, q_path in enumerate(query_paths):
            base64_query = self._encode_image(q_path)
            content.append({"type": "text", "text": f"Query Image {i}:"})
            content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_query}"}})
        
        content.append({"type": "text", "text": "Gallery Candidates:"})
        for i, path in enumerate(gallery_paths):
            base64_img = self._encode_image(path)
            content.append({"type": "text", "text": f"Index {i}:"})
            content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_img}"}})

        from langchain_core.messages import HumanMessage
        response = self.llm.invoke([HumanMessage(content=content)])
        match = re.search(r'\d+', response.content)
        return int(match.group()) if match else -1

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
            print("Please install openai: pip install openai")
            self.client = None

    def _encode_image(self, image_path):
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode('utf-8')

    @retry_on_exception(max_retries=3, delay=3)
    def predict(self, query_paths, gallery_paths):
        if not self.client: return -1

        # 兼容多查询
        if not isinstance(query_paths, list):
            query_paths = [query_paths]

        query_content = []
        for i, q_path in enumerate(query_paths):
            base64_query = self._encode_image(q_path)
            query_content.extend([
                {"type": "text", "text": f"Query Image {i}:"},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_query}"}}
            ])

        gallery_content = []
        for i, path in enumerate(gallery_paths):
            base64_img = self._encode_image(path)
            gallery_content.extend([
                {"type": "text", "text": f"Image {i}:"},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_img}"}}
            ])

        messages = [
            {
                "role": "system", 
                "content": """You are an expert in Person Re-Identification (ReID). 
Your task is to identify the SAME person from a gallery of candidates based on a query image.
Follow these steps strictly:
1. Analyze the Query Image: Describe the person's upper clothing (color, pattern, style), lower clothing, shoes, and visible accessories (bag, hat).
2. Compare with Gallery: For each candidate, check if these specific features match.
3. Eliminate Distractors: Ignore people with different clothing colors or styles, even if the background is similar.
4. Decision: Select the index of the person who is the same identity. Return the final answer in JSON format: {"analysis": "...", "index": <number>}."""
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

        response = self.client.chat.completions.create(
            model=self.model, messages=messages, max_tokens=1024, temperature=0.0
        )
        if isinstance(response, str):
            print(f"\n[DEBUG] API Error: Expected object but got string: {response[:200]}...")
            return -1

        content = response.choices[0].message.content.strip()
        print(f"\n[DEBUG] Model Response: {content}") # Debug print
        
        # 优化解析逻辑：优先解析 JSON 中的 index 字段
        try:
            # 找到最后一个 { 并尝试解析 JSON
            json_str_match = re.findall(r'\{[^{}]*\}', content)
            if json_str_match:
                import json
                # 尝试解析最后一个包含 "index" 的 JSON 对象
                for candidate in reversed(json_str_match):
                    try:
                        data = json.loads(candidate)
                        if "index" in data:
                            return int(data["index"])
                    except:
                        continue
        except:
            pass

        # 如果 JSON 解析失败，再回退到正则匹配数字，但要排除掉前面的分析文本中的干扰数字
        # 寻找 "index": 或 "index" 之后的数字
        index_match = re.search(r'"index"\s*:\s*(\d+)', content, re.IGNORECASE)
        if index_match:
            return int(index_match.group(1))

        match = re.search(r'\d+', content)
        return int(match.group()) if match else -1

class QwenReIDAgent(BaseReIDAgent):
    def __init__(self, api_key, model="qwen3.5-plus", base_url=None):
        self.api_key = api_key
        self.model = model
        dashscope.api_key = api_key
        # Only set base_url if it is not the OpenAI compatible one, as dashscope SDK uses native endpoint
        if base_url and "compatible-mode" not in base_url:
            dashscope.base_http_api_url = base_url

    @retry_on_exception(max_retries=3, delay=3)
    def predict(self, query_paths, gallery_paths):
        # 兼容多查询
        if not isinstance(query_paths, list):
            query_paths = [query_paths]

        content = [{"text": "Identify the person in Query Images from Gallery. Return ONLY the index number."}]
        for i, q_path in enumerate(query_paths):
            content.append({"text": f"Query Image {i}:"})
            content.append({"image": f"file://{os.path.abspath(q_path)}"})
        
        content.append({"text": "Gallery below:"})

        for i, path in enumerate(gallery_paths):
            content.append({"text": f"Index {i}:"})
            content.append({"image": f"file://{os.path.abspath(path)}"})

        messages = [{"role": "user", "content": content}]
        
        # 2. Call API
        response = dashscope.MultiModalConversation.call(
            model=self.model,
            messages=messages
        )
        
        if response.status_code == 200:
            result_text = response.output.choices[0].message.content[0]['text']
            match = re.search(r'\d+', result_text)
            return int(match.group()) if match else -1
        else:
            # Raise exception to trigger retry
            raise Exception(f"Qwen API Error: {response.code} - {response.message}")

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

    def predict(self, query_path, gallery_paths, ground_truth_idx=None):
        # 1. Selection (MCTS)
        parent_node = self._select_node()
        
        # 2. Expansion/Variation (Fireworks 'Explosion')
        if parent_node and parent_node.visits > 0:
            new_prompt = self._get_evolved_prompt(parent_node)
            current_node = self.journal.add_node(new_prompt, parent=parent_node)
        else:
            current_node = parent_node

        # 3. Simulation (Execution)
        # Use child agent prediction
        prediction = self.base_agent.predict(query_path, gallery_paths)
        
        # 4. Backpropagation & Memory Update
        if ground_truth_idx is not None:
            is_correct = (prediction == ground_truth_idx)
            reward = 1.0 if is_correct else 0.0
            if current_node:
                self.journal.update_mcts(current_node, reward)
            
            # 同时更新运行时内存和磁盘持久化内存
            self.memory.add_experience(current_node.prompt if current_node else "Initial", prediction, ground_truth_idx)
            
            # 记录到短时记忆文件
            self.persistent_memory.add_trial(
                prompt=current_node.prompt if current_node else "Initial", 
                is_correct=is_correct,
                analysis=f"Pred: {prediction}, GT: {ground_truth_idx}"
            )
            
        return prediction
