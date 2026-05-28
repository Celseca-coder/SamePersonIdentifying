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
from memory_module import ReIDMemoryModule

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
    def predict(self, query_paths, gallery_paths):
        raise NotImplementedError

class MockReIDAgent(BaseReIDAgent):
    def predict(self, query_paths, gallery_paths):
        q_pid = int(os.path.basename(query_paths[0]).split('_')[0])
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
        try:
            from langchain_openai import ChatOpenAI
            # 统一加入 base_url
            self.llm = ChatOpenAI(model=model, api_key=api_key, base_url=base_url)
        except ImportError:
            print("Please install langchain-openai: pip install langchain-openai")
            self.llm = None

    def _encode_image(self, image_path):
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode('utf-8')

    @retry_on_exception(max_retries=3, delay=3)
    def predict(self, query_paths, gallery_paths, custom_prompt=None):
        if not self.llm: return -1
        
        text_prompt = custom_prompt if custom_prompt else "Identify the person in Query Image(s) from Gallery. Return ONLY the index number."
        content = [
            {"type": "text", "text": text_prompt}
        ]
        
        for i, q_path in enumerate(query_paths):
            base64_query = self._encode_image(q_path)
            content.append({"type": "text", "text": f"Query Image {i}:"})
            content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_query}"}})
        
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
                    {"type": "text", "text": "Query Image(s) (Look for this person):"},
                    *query_content,
                    {"type": "text", "text": "Gallery Candidates:"},
                    *gallery_content,
                    {"type": "text", "text": "Which index matches the Query Image(s)? Think step by step."}
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
        content = [{"text": "Identify the person in Query Image(s) from Gallery. Return ONLY the index number."}]
        for i, q_path in enumerate(query_paths):
            content.append({"text": f"Query Image {i}:"})
            content.append({"image": f"file://{os.path.abspath(q_path)}"})
        content.append({"text": "Query Image(s) are above. Gallery below:"})

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
    def __init__(self, api_key, model="gpt-4o", base_url=None, backend="openai"):
        self.journal = Journal()
        self.memory = MemoryManager()
        # 显式传入本文件所在目录，确保记忆始终写入 src/memory/
        self.persistent_memory = ReIDMemoryModule(
            workspace_path=os.path.dirname(os.path.abspath(__file__))
        )
        self.api_key = api_key
        self.model = model
        
        # 允许 Evo 选择 LangChain 或 OpenAI 作为底层调用框架
        if backend == "langchain":
            self.base_agent = LangChainReIDAgent(api_key, model, base_url)
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

    def predict(self, query_paths, gallery_paths, ground_truth_idx=None):
        # 1. Selection (MCTS)
        parent_node = self._select_node()
        
        # 2. Expansion/Variation (Fireworks 'Explosion')
        if parent_node and parent_node.visits > 0:
            new_prompt = self._get_evolved_prompt(parent_node)
            current_node = self.journal.add_node(new_prompt, parent=parent_node)
        else:
            current_node = parent_node

        # 3. Simulation (Execution)
        # Use child agent prediction with the evolved prompt
        eval_prompt = current_node.prompt if current_node else None
        
        try:
            # Check if base_agent supports custom_prompt (like LangChain)
            prediction = self.base_agent.predict(query_paths, gallery_paths, custom_prompt=eval_prompt)
        except TypeError:
            # Fallback for Qwen or others that might not support it yet
            prediction = self.base_agent.predict(query_paths, gallery_paths)
            
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

class LangChainBatchImageRetrievalManager:
    """
    利用 Langchain 封装好的图像检索与向量数据库能力，支持对大量图像数据进行自动化、成批处理。
    适用场景：对庞大的 Gallery 图库进行离线向量化并建立索引，然后对大批量的 Query 实现自动化检索。
    注意：此部分依赖本地检索多模态能力，需安装 langchain-experimental, open_clip_torch, chromadb 等库。
    """
    def __init__(self, persist_directory="./langchain_reid_db"):
        try:
            from langchain_experimental.open_clip import OpenCLIPEmbeddings
            from langchain_chroma import Chroma
            
            # 初始化多模态嵌入模型 (如 CLIP) 进行图像特征提取
            self.embedding_model = OpenCLIPEmbeddings(model_name="ViT-B-32", checkpoint="laion2b_s34b_b79k")
            
            # 使用 Langchain 的 Chroma 构建本地向量库进行图像检索
            self.vectorstore = Chroma(
                collection_name="reid_gallery",
                embedding_function=self.embedding_model,
                persist_directory=persist_directory
            )
        except ImportError:
            print("请先安装对应依赖项: pip install langchain-experimental open_clip_torch chromadb torch")
            self.vectorstore = None
            self.embedding_model = None

    def build_gallery_batch(self, gallery_paths, batch_size=32):
        if not self.vectorstore:
            return

        print(f"Starting batch processing for {len(gallery_paths)} gallery images...")
        total_batches = (len(gallery_paths) + batch_size - 1) // batch_size
        
        for i in range(0, len(gallery_paths), batch_size):
            batch_paths = gallery_paths[i:i + batch_size]
            try:
                # 手动传入 metadatas，把路径存进去
                metadatas = [{"source": p} for p in batch_paths]
                self.vectorstore.add_images(uris=batch_paths, metadatas=metadatas)
                print(f"Successfully indexed batch {i//batch_size + 1}/{total_batches}")
            except Exception as e:
                print(f"Error indexing batch {i//batch_size + 1}: {e}")

    def automate_batch_query(self, query_paths, k=5):
        """
        利用构建好的向量数据库，自动化对一批 Query 进行图像检索操作，返回 Top-K 最相近的图像路径。
        """
        if not self.vectorstore:
            return []

        results = []
        print(f"Automating batch retrieval for {len(query_paths)} queries...")
        for idx, q_path in enumerate(query_paths):
            try:
                # 提取 Query 的图像特征进行相似度检索 (Similarity Search by Vector)
                q_emb = self.embedding_model.embed_image([q_path])[0]
                matched_docs = self.vectorstore.similarity_search_by_vector(q_emb, k=k)
                
                # 读取 metadata 中的来源路径
                matched_uris = [doc.metadata.get("source", "Unknown") for doc in matched_docs]
                results.append({"query": q_path, "top_k_matches": matched_uris})
                
                if (idx + 1) % 10 == 0:
                    print(f"Processed {idx + 1} queries...")
            except Exception as e:
                print(f"Error querying image {q_path}: {e}")
                results.append({"query": q_path, "error": str(e)})
                
        return results

    def process_with_llm_batch(self, llm_chain, batch_inputs):
        """
        利用 Langchain 原生的 .batch() 并发接口，自动化并行处理 LLM 任务 (例如批量重识别推理)，
        极大提升大规模数据下的调用效率。
        """
        print(f"Sending batch of {len(batch_inputs)} tasks to LLM...")
        try:
            # concurrency 可以配置最大并发数
            responses = llm_chain.batch(batch_inputs, config={"max_concurrency": 5})
            return responses
        except Exception as e:
            print(f"LLM Batch execution failed: {e}")
            return []
