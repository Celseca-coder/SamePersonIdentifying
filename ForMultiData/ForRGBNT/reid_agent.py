import base64
import random
import os
from PIL import Image, ImageDraw, ImageFont
import io
import dashscope
import time
import re
import math
from functools import wraps
from memory_module import ReIDMemoryModule
import textwrap
from openai import OpenAI

# --- 全局 ReID 核心原则 (移除了强行绑定颜色的词汇) ---
SHARED_REID_RULES = """
CRITICAL RULES FOR REID TASK (STRICT ENFORCEMENT):
1. ANTI-HALLUCINATION CONSTRAINT: NEVER invent visual features. Reject heavily occluded or cropped images immediately.
2. HARD NEGATIVE REJECTION: Actively search for irreconcilable structural differences. However, DO NOT treat an item as a contradiction if it could simply be hidden by the person's body due to a different camera viewpoint.
3. FINE-GRAINED GRANULARITY: Describe structural shapes, inferred materials, patterns, and specific accessories. IGNORE COLOR entirely if cross-modal sensors (Thermal/NIR) are involved.
4. CONFIDENCE SCORING RUBRIC: 
   - 90-100: Unique identifying features (topology/accessories) match perfectly across multiple views.
   - 70-89: Generic clothing structure matches, but lacks unique accessories.
   - 0-50: Significant structural contradiction found.
"""

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

# ==========================================
# 策略一：Pair-wise (一对一精准排查)
# ==========================================
class PairWiseLocalAgent(BaseReIDAgent):
    def __init__(self, api_key="local-test", model="/data/llm/AI-ModelScope/R-4B", base_url="http://localhost:8000/v1"):
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        self.log_file = "reid_inference_log.md"
        self.total_tokens_used = 0
        print(f"🚀 vLLM API 模式已激活，完整推理日志将保存至: {self.log_file}")

    def _encode_image(self, image_path):
        img = Image.open(image_path)
        img = img.resize((224, 448), Image.Resampling.LANCZOS)
        buffered = io.BytesIO()
        img.save(buffered, format="JPEG")
        return base64.b64encode(buffered.getvalue()).decode('utf-8')

    def predict(self, query_paths, gallery_paths, guidance="无"):
        query_contents = []
        for i, q_path in enumerate(query_paths):
            q_base64 = self._encode_image(q_path)
            query_contents.extend([
                {"type": "text", "text": f"[Query Image {i+1}]:"},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{q_base64}"}}
            ])
        
        results_list = []   
        analyses = []
        
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(f"\n## 🔍 New Pair-wise Trial Analysis\n")
            q_names = [os.path.basename(p) for p in query_paths]
            f.write(f"- **Query Images ({len(query_paths)} views)**: {q_names}\n\n")
            
        guidance_block = f"### 🚨 TASK SPECIFIC GUIDANCE 🚨\n{guidance}\n*SYSTEM WARNING: Strictly adhere to this guidance!*" if guidance and guidance != "无" else ""

        for idx, g_path in enumerate(gallery_paths):
            g_base64 = self._encode_image(g_path)
            
            prompt_text = textwrap.dedent(f"""\
                # SYSTEM INSTRUCTION: CRITICAL OUTPUT CONSTRAINT
                {guidance_block}
                
                You are a highly analytical AI vision expert specializing in Pedestrian Re-Identification. 
                Your task is to determine if the person shown in the [Query Images] (which provide MULTIPLE VIEWS of the EXACT SAME target person) and the Gallery Candidate image contain the EXACT SAME person.
               
                {SHARED_REID_RULES}
                
                INSTRUCTIONS:
                Use extremely concise, bulleted notes (max 10 words per field) to avoid token truncation.

                ### Step 1: Synthesize Query Profile (Multi-View Integration)
                - Combined Profile: [Synthesize clothing structure and accessories across ALL Query Images into ONE complete 3D profile. Note items visible in one view but hidden in others.]

                ### Step 2: Head & Gender Comparison
                - Synthesized Query: [Short description]
                - Gallery Candidate: [Short description]
                - Contradiction?: [Yes/No]

                ### Step 3: Upper Body Comparison
                - Synthesized Query: [Exact sleeve length, visible patterns, clothing topology]
                - Gallery Candidate: [Exact sleeve length, visible patterns, clothing topology]
                - Contradiction?: [Yes/No]

                ### Step 4: Lower Body & Accessories Comparison
                - Synthesized Query: [Length, pants/skirt, carried items like bags/phones]
                - Gallery Candidate: [Length, pants/skirt, carried items]
                - Contradiction?: [Yes/No]

                ### Step 5: Final Verdict
                Reasoning: [Provide exactly ONE concise sentence stating the strongest reason]
                [CONFIDENCE]: <Number from 0 to 100>
                [VERDICT]: MATCH  (or)  [VERDICT]: MISMATCH
            """)
            
            content = [{"type": "text", "text": prompt_text}] + query_contents + [
                {"type": "text", "text": f"### Gallery Candidate {idx}:"},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{g_base64}"}}
            ]

            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": content}],
                    max_tokens=2048, 
                    temperature=0.0
                )

                if hasattr(response, 'usage') and response.usage:
                    self.total_tokens_used += response.usage.total_tokens
                
                res_content = response.choices[0].message.content.strip()

                match_result = re.search(r'\[VERDICT\][^\w]*(MATCH|MISMATCH)', res_content, re.IGNORECASE)
                conf_result = re.search(r'\[CONFIDENCE\][^\d]*(\d+)', res_content, re.IGNORECASE)
                confidence = int(conf_result.group(1)) if conf_result else 0
                confidence = min(max(confidence, 0), 100)
                
                if match_result:
                    verdict_str = match_result.group(1).upper()
                    is_match = (verdict_str == 'MATCH')
                    status_text = f"✅ 匹配 (Conf: {confidence})" if is_match else f"❌ 不匹配 (Conf: {confidence})"
                    results_list.append({"idx": idx, "is_match": is_match, "confidence": confidence})
                else:
                    status_text = "⚠️ 解析异常"
                    results_list.append({"idx": idx, "is_match": False, "confidence": 0})

                analyses.append(res_content)

                with open(self.log_file, "a", encoding="utf-8") as f:
                    f.write(f"### 🖼️ Candidate {idx} -> **{status_text}**\n")
                    f.write(f"> {res_content.replace(chr(10), '<br>')}\n\n")

            except Exception as e:
                results_list.append({"idx": idx, "is_match": False, "confidence": 0}) 
                analyses.append(f"Error: {e}")

        true_candidates = [res for res in results_list if res["is_match"]]
        
        predicted_idx = -1
        if true_candidates:
            true_candidates.sort(key=lambda x: x["confidence"], reverse=True)
            predicted_idx = true_candidates[0]["idx"]
        else:
            valid_results = [res for res in results_list if res["confidence"] > 0]
            if valid_results:
                valid_results.sort(key=lambda x: x["confidence"], reverse=True)
                predicted_idx = valid_results[0]["idx"]
                
        return predicted_idx, analyses

# ==========================================
# 策略二：List-wise (一次性看所有图)
# ==========================================
class ListWiseLocalAgent(BaseReIDAgent):
    def __init__(self, api_key="local-test", model="/data/llm/AI-ModelScope/R-4B", base_url="http://localhost:8000/v1"):
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        self.log_file = "reid_inference_log.md"
        self.total_tokens_used = 0

    def _encode_image(self, img_obj_or_path):
        if isinstance(img_obj_or_path, str):
            img = Image.open(img_obj_or_path)
            img = img.resize((224, 448), Image.Resampling.LANCZOS)
        else:
            img = img_obj_or_path
        buffered = io.BytesIO()
        img.save(buffered, format="JPEG", quality=85)
        return base64.b64encode(buffered.getvalue()).decode('utf-8')

    def _create_gallery_grid(self, gallery_paths):
        img_w, img_h = 160, 320  
        padding = 10
        text_h = 40
        cols = 5 
        rows = math.ceil(len(gallery_paths) / cols)
        
        total_w = cols * img_w + (cols + 1) * padding
        total_h = rows * (img_h + text_h) + (rows + 1) * padding
        
        grid_img = Image.new('RGB', (total_w, total_h), 'white')
        draw = ImageDraw.Draw(grid_img)
        
        try:
            font = ImageFont.truetype("arial.ttf", 30)
        except IOError:
            font = ImageFont.load_default()
            
        for idx, g_path in enumerate(gallery_paths):
            row = idx // cols
            col = idx % cols
            x = padding + col * (img_w + padding)
            y = padding + row * (img_h + text_h + padding)
            
            sub_img = Image.open(g_path).resize((img_w, img_h), Image.Resampling.LANCZOS)
            grid_img.paste(sub_img, (x, y + text_h))
            
            draw.rectangle([x, y, x + img_w, y + text_h], fill="red")
            draw.text((x + 10, y + 2), f"Index {idx}", fill="white", font=font)
            
        return grid_img

    def predict(self, query_paths, gallery_paths, guidance="无"):
        query_contents = []
        for i, q_path in enumerate(query_paths):
            q_base64 = self._encode_image(q_path)
            query_contents.extend([
                {"type": "text", "text": f"[Query Image {i+1}]:"},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{q_base64}"}}
            ])
            
        gallery_grid_img = self._create_gallery_grid(gallery_paths)
        grid_base64 = self._encode_image(gallery_grid_img)
        
        prompt_text = textwrap.dedent(f"""\
            # SYSTEM INSTRUCTION
            {guidance}
            
            You are a highly analytical AI vision expert specializing in Pedestrian Re-Identification.
            
            TASK: 
            I will provide you with [Query Images] showing a specific target person from potentially MULTIPLE camera angles.
            Then, I will provide a SINGLE composite image called [Gallery Grid]. This grid contains {len(gallery_paths)} candidate images.
            
            YOUR OBJECTIVE:
            Find the exact same person from the [Gallery Grid] that matches the person in the [Query Images].
            
            HOW TO THINK (in your <think> tags):
            1. Synthesize a complete 3D profile from ALL Query Images. Note structural features, clothing topology, and accessories.
            2. Visually scan the Gallery Grid. Eliminate candidates with obvious structural mismatches (Ignore color if instructed by guidance).
            3. For the remaining candidates, compare detailed topology (shoes, bags, boundaries).
            
            CRITICAL FORMAT RULE:
            After your <think> process, you MUST output your final decision exactly in this format on a new line:
            [MATCH_INDEX]: <number>
            (If none match, output [MATCH_INDEX]: -1)
        """)
        
        content = [{"type": "text", "text": prompt_text}] + query_contents + [
            {"type": "text", "text": "### [Gallery Grid Image]:"},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{grid_base64}"}},
            {"type": "text", "text": "Take a deep breath and carefully look at the images to find the match. Remember to output [MATCH_INDEX]: <number> at the very end."}
        ]

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": content}],
                max_tokens=1500, 
                temperature=0.0
            )
            
            if hasattr(response, 'usage') and response.usage:
                self.total_tokens_used += response.usage.total_tokens
                
            res_content = response.choices[0].message.content.strip()
            match = re.search(r'\[MATCH_INDEX\]:\s*(-?\d+)', res_content, re.IGNORECASE)
            pred_idx = int(match.group(1)) if match else -1
            
            with open(self.log_file, "a", encoding="utf-8") as f:
                f.write(f"\n## 🔍 List-wise Grid Trial (Gallery Size: {len(gallery_paths)})\n")
                f.write(f"Model decided Index: {pred_idx}\n> {res_content}\n\n")
                
            return pred_idx, [res_content]
            
        except Exception as e:
            print(f"List-wise API Error: {e}")
            return -1, [str(e)]

# ==========================================
# 策略三：Tournament (锦标赛分组淘汰)
# ==========================================
class TournamentLocalAgent(BaseReIDAgent):
    def __init__(self, api_key="local-test", model="/data/llm/AI-ModelScope/R-4B", base_url="http://localhost:8000/v1", group_size=5):
        self.referee = ListWiseLocalAgent(api_key, model, base_url)
        self.group_size = group_size 
        self.log_file = "reid_inference_log.md"

    @property
    def total_tokens_used(self):
        return self.referee.total_tokens_used

    def predict(self, query_paths, gallery_paths, guidance="无"):
        total_candidates = len(gallery_paths)
        if total_candidates <= self.group_size:
            return self.referee.predict(query_paths, gallery_paths, guidance)
            
        winners_info = [] 
        for i in range(0, total_candidates, self.group_size):
            group_paths = gallery_paths[i : i + self.group_size]
            local_indices = list(range(i, i + len(group_paths))) 
            
            local_winner_idx, _ = self.referee.predict(query_paths, group_paths, guidance)
            if local_winner_idx != -1 and local_winner_idx < len(group_paths):
                winners_info.append({"global_idx": local_indices[local_winner_idx], "path": group_paths[local_winner_idx]})

        if not winners_info:
            return -1, ["No winners from group stage."]
        if len(winners_info) == 1:
            return winners_info[0]["global_idx"], ["Won by default in final."]
            
        final_paths = [w["path"] for w in winners_info]
        final_global_indices = [w["global_idx"] for w in winners_info]
        
        final_local_idx, analysis = self.referee.predict(query_paths, final_paths, guidance)
        
        if final_local_idx != -1 and final_local_idx < len(final_paths):
            absolute_winner = final_global_indices[final_local_idx]
        else:
            absolute_winner = -1
            
        return absolute_winner, analysis

# ==========================================
# 控制双路反思的调度器 Agent
# ==========================================
class EvolutionaryReIDAgent(BaseReIDAgent):
    def __init__(self, api_key="local-test", model="/data/llm/AI-ModelScope/R-4B", base_url="http://localhost:8000/v1"):
        self.persistent_memory = ReIDMemoryModule()
        self.active_guidance_list = [] 
        self.base_agent = PairWiseLocalAgent(api_key, model, base_url) 
    
    def _get_current_guidance(self):
        if not self.active_guidance_list:
            return ""
        return "\n### PAST MEMORY GUIDANCE ###\n" + "\n".join([f"- {g}" for g in self.active_guidance_list[-3:]])

    def visual_reflect_wrong_match(self, query_paths, wrong_paths, correct_paths, wrong_reason, correct_reason):
        print("\n[Reflector] 🔍 触发多模态视觉针对错认(FP)反思机制...")
        
        prompt = textwrap.dedent(f"""\
            你是一个严厉且资深的刑侦视觉督导专家。你的下属（Predictor）在跨光谱/多视角行人重识别时认错人了！
            
            【下属当时的错误推理录音】：
            1. 错认理由：{wrong_reason}
            2. 排除真目标理由：{correct_reason}
            
            【你的复盘任务】：
            前面的图片是目标人物的多个视角(Query)。接下来是错认的路人(Wrong Match)，最后是真正的目标(Correct Match)。
            请狠狠地批判下属的推理过程！指出他在多视角融合时忽略了什么？在跨模态时被什么伪装特征骗了？
            最后，总结出一条强硬的【视觉排查铁律】。必须以 `[GUIDANCE]:` 开头输出。
        """)
        
        content = [{"type": "text", "text": prompt}]
        
        # 将所有 Query 视角图传入 Reflector
        for i, q in enumerate(query_paths):
            q_b64 = self.base_agent._encode_image(q)
            content.extend([
                {"type": "text", "text": f"[Query Image {i+1}]:"},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{q_b64}"}}
            ])
            
        wrong_b64 = self.base_agent._encode_image(wrong_paths[0])
        correct_b64 = self.base_agent._encode_image(correct_paths[0])
        
        content.extend([
            {"type": "text", "text": "[Wrong Match (错认的路人)]:"},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{wrong_b64}"}},
            {"type": "text", "text": "[Correct Match (真正的目标)]:"},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{correct_b64}"}}
        ])
        
        try:
            response = self.base_agent.client.chat.completions.create(
                model=self.base_agent.model,
                messages=[{"role": "user", "content": content}],
                max_tokens=512,
                temperature=0.6 
            )
            if hasattr(response, 'usage') and response.usage:
                self.base_agent.total_tokens_used += response.usage.total_tokens
                
            res_content = response.choices[0].message.content.strip()
            print(f"\n[Reflector] 导师痛骂:\n{res_content}\n")
            
            match = re.search(r'\[GUIDANCE\]:\s*(.*)', res_content, re.IGNORECASE)
            return match.group(1) if match else "交叉核对多视角结构细节，拒绝幻觉！"
        except Exception as e:
            return "仔细检查细节！"
        
    def visual_reflect_miss_hit(self, query_paths, correct_paths, missed_reason):
        print("\n[Reflector] 🔍 触发多模态针对漏认(FN)反思机制...")
        
        prompt = textwrap.dedent(f"""\
            你是一个严厉的刑侦视觉督导专家。你的下属（Predictor）犯了一个“漏网之鱼”的错误，判定目标为【不匹配】。
            
            【下属排查真实目标时的错误推理录音】：
            {missed_reason}
            
            【你的复盘任务】：
            前面的图片是目标人物的多个视角(Query)。最后是真实目标 (Correct Match)。这绝对是同一个人！
            请批判下属的推理：他是不是对某个特征要求太苛刻了？是不是没有综合利用多个 Query 视角的信息？是不是被跨传感器的干扰骗了？
            请总结出一条强硬的【视觉排查铁律】。必须以 `[GUIDANCE]:` 开头输出。
        """)
        
        content = [{"type": "text", "text": prompt}]
        
        for i, q in enumerate(query_paths):
            q_b64 = self.base_agent._encode_image(q)
            content.extend([
                {"type": "text", "text": f"[Query Image {i+1}]:"},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{q_b64}"}}
            ])
            
        correct_b64 = self.base_agent._encode_image(correct_paths[0])
        content.extend([
            {"type": "text", "text": "[Correct Match (真正的目标)]:"},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{correct_b64}"}}
        ])
        
        try:
            response = self.base_agent.client.chat.completions.create(
                model=self.base_agent.model,
                messages=[{"role": "user", "content": content}],
                max_tokens=512,
                temperature=0.6 
            )
            if hasattr(response, 'usage') and response.usage:
                self.base_agent.total_tokens_used += response.usage.total_tokens
                
            res_content = response.choices[0].message.content.strip()
            print(f"\n[Reflector] 导师痛骂:\n{res_content}\n")
            
            match = re.search(r'\[GUIDANCE\]:\s*(.*)', res_content, re.IGNORECASE)
            return match.group(1) if match else "透视表面颜色差异，关注人体拓扑本质！"
        except Exception as e:
            return "仔细核对跨模态特征！"

    def predict(self, query_paths, gallery_paths, ground_truth_idx=None, guidance="无"):
        memory_guidance = self._get_current_guidance()
        
        # 将外部的任务 Guidance (如 MIXED_MULTIMODAL_GUIDANCE) 与 Evo 自己的反思记忆组合
        if guidance != "无" and guidance.strip():
            combined_guidance = f"{guidance}\n\n{memory_guidance}"
        else:
            combined_guidance = memory_guidance
            
        print(f"\n[Predictor] 正在执行识别 (Query Size: {len(query_paths)})...")
        pred_idx, analyses_list = self.base_agent.predict(query_paths, gallery_paths, guidance=combined_guidance)
        
        if ground_truth_idx is not None:
            # 兼容 DataLoader 返回多个 Ground Truth 的情况（列表）
            if isinstance(ground_truth_idx, list):
                is_correct = pred_idx in ground_truth_idx
                # 取第一个正样本用来给 Reflector 做复盘
                gt_target = ground_truth_idx[0] if ground_truth_idx else -1
            else:
                is_correct = (pred_idx == ground_truth_idx)
                gt_target = ground_truth_idx

            if not is_correct:
                new_guidance = None
                if pred_idx != -1:
                    wrong_paths = [gallery_paths[pred_idx]]
                    correct_paths = [gallery_paths[gt_target]]
                    wrong_reason = analyses_list[pred_idx] if pred_idx < len(analyses_list) else "无记录"
                    correct_reason = analyses_list[gt_target] if gt_target < len(analyses_list) else "无记录"
                    
                    print("\n[Reflector] 检测到错认(FP)，触发排雷反思...")
                    new_guidance = self.visual_reflect_wrong_match(
                        query_paths, wrong_paths, correct_paths, wrong_reason, correct_reason
                    )
                else:
                    correct_paths = [gallery_paths[gt_target]]
                    missed_reason = analyses_list[gt_target] if gt_target < len(analyses_list) else "无记录"
                    
                    print("\n[Reflector] 检测到漏认(FN)，触发召回反思...")
                    new_guidance = self.visual_reflect_miss_hit(
                        query_paths, correct_paths, missed_reason
                    )
                
                if new_guidance:
                    print(f"✨ [Memory] 习得新规矩: {new_guidance}")
                    self.active_guidance_list.append(new_guidance)
            
            log_analysis = f"Pred: {pred_idx}, GT: {ground_truth_idx}"
            self.persistent_memory.add_trial(prompt=combined_guidance, is_correct=is_correct, analysis=log_analysis)
            
        # 严格保持统一的返回格式（仅返回最佳预测索引，供评测脚本核对）
        return pred_idx