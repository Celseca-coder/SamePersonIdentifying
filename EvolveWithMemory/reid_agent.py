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
import textwrap

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
    def predict(self, query_path, gallery_paths):
        q_pid = int(os.path.basename(query_path).split('_')[0])
        for i, path in enumerate(gallery_paths):
            g_pid = int(os.path.basename(path).split('_')[0])
            if g_pid == q_pid:
                return i
        return 0

class RandomReIDAgent(BaseReIDAgent):
    def predict(self, query_path, gallery_paths):
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
    def predict(self, query_path, gallery_paths):
        if not self.llm: return -1
        
        base64_query = self._encode_image(query_path)
        content = [
            {"type": "text", "text": "Identify the person in Query Image from Gallery. Return ONLY the index number."}
        ]
        content.append({"type": "text", "text": "Query Image:"})
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
    def predict(self, query_path, gallery_paths):
        if not self.client: return -1

        base64_query = self._encode_image(query_path)
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
                    {"type": "text", "text": "Query Image (Look for this person):"},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_query}"}},
                    {"type": "text", "text": "Gallery Candidates:"},
                    *gallery_content,
                    {"type": "text", "text": "Which index matches the Query Image? Think step by step."}
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
    def predict(self, query_path, gallery_paths):
        content = [{"text": "Identify the person in Query Image from Gallery. Return ONLY the index number."}]
        # Use file:// path for local images
        content.append({"image": f"file://{os.path.abspath(query_path)}"})
        content.append({"text": "Query Image is above. Gallery below:"})

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

class LocalReIDAgent(BaseReIDAgent):
    def __init__(self, api_key="local-test", model="/data/llm/AI-ModelScope/R-4B", base_url="http://localhost:8000/v1"):
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        self.log_file = "reid_inference_log.md"
        print(f"🚀 vLLM API (Predictor) Rank-1 模式已激活。")

    def _encode_image(self, image_path):
        img = Image.open(image_path)
        img = img.resize((224, 448), Image.Resampling.LANCZOS)
        buffered = io.BytesIO()
        img.save(buffered, format="JPEG")
        return base64.b64encode(buffered.getvalue()).decode('utf-8')

    def predict(self, query_path, gallery_paths, guidance="无"):
        query_base64 = self._encode_image(query_path)
        matches = []   
        analyses = [] # 保存列表，防止截断

        for idx, g_path in enumerate(gallery_paths):
            g_base64 = self._encode_image(g_path)
            
            prompt_text = textwrap.dedent(f"""\
                # SYSTEM INSTRUCTION: CRITICAL OUTPUT CONSTRAINT
                
                【🚨来自过去的深刻教训 (GUIDANCE)🚨】: 
                {guidance}
                请你务必在本次分析中严格遵守上述教训！
                
                You are an AI vision expert specializing in Pedestrian Re-Identification.
                Task: Determine if the Query image and the Gallery Candidate contain the EXACT SAME person.

                RULES:
                    1. NO HALLUCINATION.
                    2. HARD NEGATIVE REJECTION: Actively search for irreconcilable differences.
                    3. FINE-GRAINED: Specify exact colors and materials.

                INSTRUCTIONS:
                    ### Step 0: Visibility Check
                    ### Step 1: Head & Gender
                    ### Step 2: Upper Body
                    ### Step 3: Lower Body & Carried Objects
                    ### Step 4: Final Verdict
                    Reasoning: [One concise sentence]
                    [VERDICT]: MATCH  (or)  [VERDICT]: MISMATCH
            """)
            
            content = [
                {"type": "text", "text": prompt_text},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{query_base64}"}},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{g_base64}"}}
            ]

            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": content}],
                    max_tokens=512, 
                    temperature=0.1 
                )
                res_content = response.choices[0].message.content.strip()
                match_result = re.search(r'\[VERDICT\]:\s*\[?(MATCH|MISMATCH)\]?', res_content, re.IGNORECASE)

                if match_result:
                    is_match = (match_result.group(1).upper() == 'MATCH')
                    matches.append(is_match)
                else:
                    matches.append(False)
                    
                analyses.append(res_content)

            except Exception as e:
                matches.append(False)
                analyses.append(f"Error: {str(e)}")

        # 核心：Rank-1 评测只取第一个判定为 True 的索引
        predicted_idx = matches.index(True) if True in matches else -1
        
        return predicted_idx, analyses

class EvolutionaryReIDAgent(BaseReIDAgent):
    def __init__(self, api_key="local-test", model="/data/llm/AI-ModelScope/R-4B", base_url="http://localhost:8000/v1"):
        self.persistent_memory = ReIDMemoryModule()
        self.active_guidance_list = [] 
        self.base_agent = LocalReIDAgent(api_key, model, base_url)
    
    def _get_current_guidance(self):
        if not self.active_guidance_list:
            return "目前还没有经验，请自由发挥。"
        return "\n".join([f"- {g}" for g in self.active_guidance_list[-3:]])

    def visual_reflect_wrong_match(self, query_path, wrong_path, correct_path, wrong_reason, correct_reason):
        """场景1：认错人（False Positive）的反思"""
        print("\n[Reflector] 🔍 触发多模态视觉针对wrong_match反思机制...")
        query_b64 = self.base_agent._encode_image(query_path)
        wrong_b64 = self.base_agent._encode_image(wrong_path)
        correct_b64 = self.base_agent._encode_image(correct_path)
        
        prompt = textwrap.dedent(f"""\
            你是一个严厉且资深的刑侦视觉督导专家。你的下属（Predictor）在行人重识别时认错人了！
            
            【下属当时的案发推理录音】：
            1. 他把 图片2(无关路人) 错认成目标的理由是：
            {wrong_reason}
            
            2. 他把 图片3(真正的目标) 排除的理由是：
            {correct_reason}
            
            【你的复盘任务】：
            图片1是目标人物(Query)。图片2是错认的路人(Wrong)。图片3是真正的目标(Correct)。
            请狠狠地批判下属的推理过程！指出他被图片2的什么伪装特征骗了？又为什么忽略了图片3的核心特征？
            最后，请总结出一条强硬、简短的【视觉排查铁律】。必须以 `[GUIDANCE]:` 开头输出。
        """)
        
        content = [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{query_b64}"}},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{wrong_b64}"}},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{correct_b64}"}}
        ]
        
        try:
            response = self.base_agent.client.chat.completions.create(
                model=self.base_agent.model,
                messages=[{"role": "user", "content": content}],
                max_tokens=512,
                temperature=0.6 # Reflector需要一点发散性和批判性
            )
            res_content = response.choices[0].message.content.strip()
            print(f"\n[Reflector] 导师痛骂:\n{res_content}\n")
            
            # 提取 GUIDANCE
            match = re.search(r'\[GUIDANCE\]:\s*(.*)', res_content, re.IGNORECASE)
            guidance = match.group(1) if match else "交叉核对细节，严禁仅凭单一部位颜色下定论！"
            return guidance
            
        except Exception as e:
            print(f"[Reflector] 反思器宕机: {e}")
            return "仔细检查所有配件，拒绝视觉幻觉！"
        
    def visual_reflect_miss_hit(self, query_path, correct_path, missed_reason):
        """场景2：没认出来/漏认（False Negative）的反思"""
        print("\n[Reflector] 🔍 触发【漏认(Miss)】反思机制...")
        query_b64 = self.base_agent._encode_image(query_path)
        correct_b64 = self.base_agent._encode_image(correct_path)
        
        prompt = textwrap.dedent(f"""\
            你是一个严厉的刑侦视觉督导专家。你的下属（Predictor）犯了一个“漏网之鱼”的错误，判定目标为【不匹配】。
            
            【下属排查真实目标时的错误推理录音】：
            {missed_reason}
            
            【你的复盘任务】：
            图片1 是目标人物 (Query)。图片2 是真实目标 (Correct Match)。这绝对是同一个人！
            请批判下属的推理：他是不是对某个特征的要求太苛刻了？（比如光线偏色、姿态变化、小面积遮挡）
            请总结出一条强硬的【视觉排查铁律】，指导他学会透过视觉干扰抓本质。必须以 `[GUIDANCE]:` 开头输出。
        """)
        
        content = [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{query_b64}"}},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{correct_b64}"}}
        ]
        
        try:
            response = self.base_agent.client.chat.completions.create(
                model=self.base_agent.model,
                messages=[{"role": "user", "content": content}],
                max_tokens=512,
                temperature=0.6 # Reflector需要一点发散性和批判性
            )
            res_content = response.choices[0].message.content.strip()
            print(f"\n[Reflector] 导师痛骂:\n{res_content}\n")
            
            # 提取 GUIDANCE
            match = re.search(r'\[GUIDANCE\]:\s*(.*)', res_content, re.IGNORECASE)
            guidance = match.group(1) if match else "交叉核对细节，严禁仅凭单一部位颜色下定论！"
            return guidance
            
        except Exception as e:
            print(f"[Reflector] 反思器宕机: {e}")
            return "仔细检查所有配件，拒绝视觉幻觉！"

    def predict(self, query_path, gallery_paths, ground_truth_idx=None):
        current_guidance = self._get_current_guidance()
        print(f"\n[Predictor] 正在执行单目标 Rank-1 识别...")
        
        # 此时 pred_idx 是一个整数 (例如 3，或 -1)
        pred_idx, analyses_list = self.base_agent.predict(query_path, gallery_paths, guidance=current_guidance)
        
        if ground_truth_idx is not None:
            is_correct = (pred_idx == ground_truth_idx)

            if not is_correct:
                new_guidance = None
                if pred_idx != -1:
                    # 【认错人】：它选了别人，触发三图对比！
                    wrong_path = gallery_paths[pred_idx]
                    correct_path = gallery_paths[ground_truth_idx]
                    wrong_reason = analyses_list[pred_idx]
                    correct_reason = analyses_list[ground_truth_idx]
                    
                    print("\n[Reflector] 检测到错认(FP)，触发排雷反思...")
                    new_guidance = self.visual_reflect_wrong_match(
                        query_path, wrong_path, correct_path, wrong_reason, correct_reason
                    )
                else:
                    # 【没认出】：它输出了 -1 (全盘否定)，触发两图对比！
                    correct_path = gallery_paths[ground_truth_idx]
                    missed_reason = analyses_list[ground_truth_idx]
                    
                    print("\n[Reflector] 检测到漏认(FN)，触发召回反思...")
                    new_guidance = self.visual_reflect_miss_hit(
                        query_path, correct_path, missed_reason
                    )
                
                if new_guidance:
                    print(f"✨ [Memory] 习得新规矩: {new_guidance}")
                    self.active_guidance_list.append(new_guidance)
            
            # 记录日志
            log_analysis = f"Pred: {pred_idx}, GT: {ground_truth_idx}"
            self.persistent_memory.add_trial(prompt=current_guidance, is_correct=is_correct, analysis=log_analysis)
            
        return pred_idx