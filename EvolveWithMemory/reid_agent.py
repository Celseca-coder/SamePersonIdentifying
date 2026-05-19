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

# --- AIDEML Inspired MCTS & Memory Components ---

SHARED_REID_RULES = """
CRITICAL RULES FOR REID TASK (STRICT ENFORCEMENT):
1. ANTI-HALLUCINATION CONSTRAINT: NEVER invent visual features. Reject heavily occluded or cropped images immediately.
2. HARD NEGATIVE REJECTION: Actively search for irreconcilable structural differences. However, DO NOT treat an item as a contradiction if it could simply be hidden by the person's body due to a different camera viewpoint (e.g., a backpack visible from the back but hidden from the front).
3. FINE-GRAINED GRANULARITY: Do NOT use generic terms. Specify exact color shades and infer materials.
4. CONFIDENCE SCORING RUBRIC: 
   - 90-100: Unique identifying features match perfectly.
   - 70-89: Generic clothing matches, but lacks unique accessories.
   - 0-50: Significant contradiction found.
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

class PairWiseLocalAgent(BaseReIDAgent):
    def __init__(self, api_key="local-test", model="/data/llm/AI-ModelScope/R-4B", base_url="http://localhost:8000/v1"):
        # 建立与 vLLM 的连接
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        self.log_file = "reid_inference_log.md"
        self.total_tokens_used = 0
        # 为防止旧数据干扰，每次初始化可以选择清空（如您所需），但在大循环中通常是追加
        # 如果需要完全覆盖之前的日志，取消下面两行的注释：
        # if os.path.exists(self.log_file):
        #     os.remove(self.log_file)
        print(f"🚀 vLLM API 模式已激活，完整推理日志将保存至: {self.log_file}")

    def _encode_image(self, image_path):
        img = Image.open(image_path)
        img = img.resize((224, 448), Image.Resampling.LANCZOS)
        buffered = io.BytesIO()
        img.save(buffered, format="JPEG")
        return base64.b64encode(buffered.getvalue()).decode('utf-8')

    # 【关键修改 1】：增加 guidance 参数
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
            f.write(f"\n## 🔍 New Trial Analysis\n")
            # 把所有 Query 图片的名字拼起来打印
            q_names = [os.path.basename(p) for p in query_paths]
            f.write(f"- **Query Images**: {q_names}\n\n")
            
        if guidance and guidance != "无" and guidance.strip() != "":
            guidance_block = f"""
### 🚨 CRITICAL GUIDANCE FROM PAST FAILURES 🚨
{guidance}
*SYSTEM WARNING: You MUST strictly adhere to the above guidance during this analysis. Do NOT repeat past mistakes!*
"""
        else:
            guidance_block = ""

        for idx, g_path in enumerate(gallery_paths):
            g_base64 = self._encode_image(g_path)
            
            # 【关键修改 2】：将 guidance 注入到您的 Prompt 顶部
            prompt_text = textwrap.dedent(f"""\
                # SYSTEM INSTRUCTION: CRITICAL OUTPUT CONSTRAINT
                {guidance_block}
                
                You are a highly analytical AI vision expert specializing in Pedestrian Re-Identification (Person ReID). 
                Your task is to determine if the person shown in the [Query Images] (which all show the SAME target person from different views) and the Gallery Candidate image contain the EXACT SAME person.
               
                {SHARED_REID_RULES}
                
                INSTRUCTIONS:
                    You MUST format your analysis strictly using the steps below. Use extremely concise, bulleted notes (maximum 10 words per field) to avoid token truncation. Do NOT use long paragraphs or conversational filler (e.g., "Wait, let's check...").

                    ### Step 1: Synthesize Query Profile (Multi-View Integration)
                - Combined Query Profile: [Synthesize the clothing, colors, and accessories seen across ALL Query Images into ONE complete 3D profile. Note items visible in one view but hidden in others.]

                ### Step 2: Head & Gender Comparison
                - Synthesized Query: [Short description based on Step 1]
                - Gallery Candidate: [Short description]
                - Contradiction?: [Yes/No]

                ### Step 3: Upper Body Comparison
                - Synthesized Query: [Color, exact sleeve length, visible logos/patterns]
                - Gallery Candidate: [Color, exact sleeve length, visible logos/patterns]
                - Contradiction?: [Yes/No]

                ### Step 4: Lower Body & Accessories Comparison
                - Synthesized Query: [Exact color, length, carried items like bags/phones]
                - Gallery Candidate: [Exact color, length, carried items. Explicitly state 'None' if empty hands]
                - Contradiction?: [Yes/No]

                ### Step 5: Final Verdict
                Reasoning: [Provide exactly ONE concise sentence stating the strongest reason for match or mismatch based on the contradictions found above]
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
                    status_text = f"✅ 匹配 (True, Conf: {confidence})" if is_match else f"❌ 不匹配 (False, Conf: {confidence})"
                    results_list.append({"idx": idx, "is_match": is_match, "confidence": confidence})
                else:
                    print(f"⚠️ 解析异常 (Parse Error): 输出结尾为: {res_content[-50:]}")
                    status_text = "⚠️ 解析异常 (Parse Error)"
                    results_list.append({"idx": idx, "is_match": False, "confidence": 0})

                analyses.append(res_content)

                formatted_analysis = res_content.replace('\n', '<br>')

                with open(self.log_file, "a", encoding="utf-8") as f:
                    f.write(f"### 🖼️ Gallery Candidate {idx}\n")
                    f.write(f"- **Path**: `{os.path.basename(g_path)}` 判定结果: **{status_text}**\n")
                    f.write(f"#### LLM Analysis:\n> {formatted_analysis}\n\n")
                    f.write("---\n")

                print(f"   -> G-{idx} 分析已记录 (判定: {status_text})")

            except Exception as e:
                results_list.append({"idx": idx, "is_match": False, "confidence": 0}) 
                error_msg = str(e)
                analyses.append(f"Error: {error_msg}")
                
                print(f"   -> ❌ G-{idx} 发生网络/API错误: {error_msg}")
                
                with open(self.log_file, "a", encoding="utf-8") as f:
                    f.write(f"### 🖼️ Gallery Candidate {idx}\n")
                    f.write(f"- **Path**: `{os.path.basename(g_path)}` 判定结果: **⚠️ API 错误 (False)**\n")
                    f.write(f"#### 💥 错误详情:\n> {error_msg}\n\n")
                    f.write("---\n")

        true_candidates = [res for res in results_list if res["is_match"]]
        
        predicted_idx = -1
        if true_candidates:
            # 如果有 MATCH，选 MATCH 里分数最高的
            true_candidates.sort(key=lambda x: x["confidence"], reverse=True)
            predicted_idx = true_candidates[0]["idx"]
            best_conf = true_candidates[0]["confidence"]
        else:
            # 【新增保底逻辑】：如果全都是 MISMATCH，我们就选 MISMATCH 里面 Confidence 最高的！
            # 因为有时模型只是不敢确信，但它给出的相对分数依然是有价值的。
            valid_results = [res for res in results_list if res["confidence"] > 0]
            if valid_results:
                valid_results.sort(key=lambda x: x["confidence"], reverse=True)
                predicted_idx = valid_results[0]["idx"]
                with open(self.log_file, "a", encoding="utf-8") as f:
                    f.write(f"### ⚠️ Fallback Decision: Candidate Index {predicted_idx}\n")
                    f.write(f"All candidates were MISMATCH. Picked the one with highest relative confidence ({valid_results[0]['confidence']}%).\n\n")
            else:
                predicted_idx = -1
                
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
        """支持传入 PIL Image 对象或路径，转换为 Base64"""
        if isinstance(img_obj_or_path, str):
            img = Image.open(img_obj_or_path)
            img = img.resize((224, 448), Image.Resampling.LANCZOS)
        else:
            img = img_obj_or_path
            
        buffered = io.BytesIO()
        img.save(buffered, format="JPEG", quality=85)
        return base64.b64encode(buffered.getvalue()).decode('utf-8')

    def _create_gallery_grid(self, gallery_paths):
        """将多张 Gallery 图片拼成一张带有明显 Index 编号的大图"""
        # 参数设置
        img_w, img_h = 160, 320  # 单张子图大小 (适当缩小以控制总像素)
        padding = 10
        text_h = 40
        cols = 5 # 每行5张图
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
            
            # 加载并粘贴子图
            sub_img = Image.open(g_path).resize((img_w, img_h), Image.Resampling.LANCZOS)
            grid_img.paste(sub_img, (x, y + text_h))
            
            # 绘制显眼的红色编号背景和文字
            draw.rectangle([x, y, x + img_w, y + text_h], fill="red")
            text = f"Index {idx}"
            draw.text((x + 10, y + 2), text, fill="white", font=font)
            
        return grid_img

    def predict(self, query_paths, gallery_paths, guidance="无"):
        query_contents = []
        for i, q_path in enumerate(query_paths):
            q_base64 = self._encode_image(q_path)
            query_contents.extend([
                {"type": "text", "text": f"[Query Image {i+1}]:"},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{q_base64}"}}
            ])
            
        # 核心优化：生成网格图并转换为 Base64
        gallery_grid_img = self._create_gallery_grid(gallery_paths)
        grid_base64 = self._encode_image(gallery_grid_img)
        
        prompt_text = textwrap.dedent(f"""\
            # SYSTEM INSTRUCTION
            {guidance}
            
            You are a highly analytical AI vision expert specializing in Pedestrian Re-Identification.
            
            TASK: 
            I will provide you with [Query Image(s)] showing a specific target person.
            Then, I will provide a SINGLE composite image called [Gallery Grid]. This grid contains {len(gallery_paths)} candidate images, each clearly labeled with a red banner saying "Index X".
            
            YOUR OBJECTIVE:
            Find the exact same person from the [Gallery Grid] that matches the person in the [Query Image(s)].
            
            HOW TO THINK (in your <think> tags):
            1. Describe the EXACT color and style of the upper and lower clothing of the Query person.
            2. Visually scan the Gallery Grid. Eliminate candidates with obvious color or clothing type mismatches.
            3. For the remaining candidates, compare details like shoes, bags, or patterns.
            
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
                max_tokens=1500, # 给足空间让它生成 <think>
                temperature=0.0
            )
            
            if hasattr(response, 'usage') and response.usage:
                self.total_tokens_used += response.usage.total_tokens
                
            res_content = response.choices[0].message.content.strip()
            
            # 使用正则提取 [MATCH_INDEX]: 5 (兼容有无空格的情况)
            match = re.search(r'\[MATCH_INDEX\]:\s*(-?\d+)', res_content, re.IGNORECASE)
            pred_idx = int(match.group(1)) if match else -1
            
            # 记录日志
            with open(self.log_file, "a", encoding="utf-8") as f:
                f.write(f"\n## 🔍 List-wise Grid Trial (Gallery Size: {len(gallery_paths)})\n")
                f.write(f"Model decided Index: {pred_idx}\n")
                f.write(f"Token Used: {response.usage.total_tokens if response.usage else 'Unknown'}\n")
                f.write(f"> {res_content}\n\n")
                
            return pred_idx, [res_content]
            
        except Exception as e:
            print(f"List-wise API Error: {e}")
            return -1, [str(e)]


# ==========================================
# 策略三：Tournament (锦标赛分组淘汰)
# ==========================================
class TournamentLocalAgent(BaseReIDAgent):
    def __init__(self, api_key="local-test", model="/data/llm/AI-ModelScope/R-4B", base_url="http://localhost:8000/v1", group_size=5):
        # 内部挂载一个 List-wise 裁判来主持比赛
        self.referee = ListWiseLocalAgent(api_key, model, base_url)
        self.group_size = group_size # 每组最多几个人
        self.log_file = "reid_inference_log.md"

    @property
    def total_tokens_used(self):
        # 读取底层裁判的电表
        return self.referee.total_tokens_used

    def predict(self, query_paths, gallery_paths, guidance="无"):
        total_candidates = len(gallery_paths)
        
        # 如果图库本来就很小，直接打总决赛
        if total_candidates <= self.group_size:
            return self.referee.predict(query_paths, gallery_paths, guidance)
            
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(f"\n## 🏆 Tournament Start (Total: {total_candidates}, Group Size: {self.group_size})\n")
            
        # 1. 小组赛阶段
        winners_info = [] # 记录冠军的真实索引和路径
        
        for i in range(0, total_candidates, self.group_size):
            group_paths = gallery_paths[i : i + self.group_size]
            local_indices = list(range(i, i + len(group_paths))) # 真实的全局索引
            
            with open(self.log_file, "a", encoding="utf-8") as f:
                f.write(f"### ⚔️ Group Match: Candidates {local_indices}\n")
                
            # 裁判裁决这个小组
            local_winner_idx, _ = self.referee.predict(query_paths, group_paths, guidance)
            
            if local_winner_idx != -1 and local_winner_idx < len(group_paths):
                global_winner_idx = local_indices[local_winner_idx]
                winners_info.append({"global_idx": global_winner_idx, "path": group_paths[local_winner_idx]})
                with open(self.log_file, "a", encoding="utf-8") as f:
                    f.write(f"👑 Group Winner: Global Index {global_winner_idx}\n")
            else:
                with open(self.log_file, "a", encoding="utf-8") as f:
                    f.write(f"💀 No winner in this group.\n")

        # 2. 总决赛阶段
        if not winners_info:
            return -1, ["No winners from group stage."]
        if len(winners_info) == 1:
            return winners_info[0]["global_idx"], ["Won by default in final."]
            
        final_paths = [w["path"] for w in winners_info]
        final_global_indices = [w["global_idx"] for w in winners_info]
        
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(f"### 🏟️ GRAND FINAL: Candidates {final_global_indices}\n")
            
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
            return "目前还没有经验，请自由发挥。"
        return "\n".join([f"- {g}" for g in self.active_guidance_list[-3:]])

    def visual_reflect_wrong_match(self, query_paths, wrong_paths, correct_paths, wrong_reason, correct_reason):
        print("\n[Reflector] 🔍 触发多模态视觉针对wrong_match反思机制...")
        query_b64 = self.base_agent._encode_image(query_paths[0])
        wrong_b64 = self.base_agent._encode_image(wrong_paths[0])
        correct_b64 = self.base_agent._encode_image(correct_paths[0])

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
                temperature=0.6 
            )
            if hasattr(response, 'usage') and response.usage:
                self.base_agent.total_tokens_used += response.usage.total_tokens
            res_content = response.choices[0].message.content.strip()
            print(f"\n[Reflector] 导师痛骂:\n{res_content}\n")
            
            match = re.search(r'\[GUIDANCE\]:\s*(.*)', res_content, re.IGNORECASE)
            return match.group(1) if match else "交叉核对细节，严禁仅凭单一部位颜色下定论！"
            
        except Exception as e:
            print(f"[Reflector] 反思器宕机: {e}")
            return "仔细检查所有配件，拒绝视觉幻觉！"
        
    def visual_reflect_miss_hit(self, query_paths, correct_paths, missed_reason):
        print("\n[Reflector] 🔍 触发【漏认(Miss)】反思机制...")
        query_b64 = self.base_agent._encode_image(query_paths[0])
        correct_b64 = self.base_agent._encode_image(correct_paths[0])
        
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
                temperature=0.6 
            )
            if hasattr(response, 'usage') and response.usage:
                self.base_agent.total_tokens_used += response.usage.total_tokens
            res_content = response.choices[0].message.content.strip()
            print(f"\n[Reflector] 导师痛骂:\n{res_content}\n")
            
            match = re.search(r'\[GUIDANCE\]:\s*(.*)', res_content, re.IGNORECASE)
            return match.group(1) if match else "交叉核对细节，严禁仅凭单一部位颜色下定论！"
            
        except Exception as e:
            print(f"[Reflector] 反思器宕机: {e}")
            return "仔细检查所有配件，拒绝视觉幻觉！"

    def predict(self, query_paths, gallery_paths, ground_truth_idx=None):
        current_guidance = self._get_current_guidance()
        print(f"\n[Predictor] 正在执行单目标 Rank-1 识别...")
        
        pred_idx, analyses_list = self.base_agent.predict(query_paths, gallery_paths, guidance=current_guidance)
        
        if ground_truth_idx is not None:
            is_correct = (pred_idx == ground_truth_idx)

            if not is_correct:
                new_guidance = None
                if pred_idx != -1:
                    wrong_paths = [gallery_paths[pred_idx]]
                    correct_paths = [gallery_paths[ground_truth_idx]]
                    wrong_reason = analyses_list[pred_idx]
                    correct_reason = analyses_list[ground_truth_idx]
                    
                    print("\n[Reflector] 检测到错认(FP)，触发排雷反思...")
                    new_guidance = self.visual_reflect_wrong_match(
                        query_paths, wrong_paths, correct_paths, wrong_reason, correct_reason
                    )
                else:
                    correct_paths = [gallery_paths[ground_truth_idx]]
                    missed_reason = analyses_list[ground_truth_idx]
                    
                    print("\n[Reflector] 检测到漏认(FN)，触发召回反思...")
                    new_guidance = self.visual_reflect_miss_hit(
                        query_paths, correct_paths, missed_reason
                    )
                
                if new_guidance:
                    print(f"✨ [Memory] 习得新规矩: {new_guidance}")
                    self.active_guidance_list.append(new_guidance)
            
            log_analysis = f"Pred: {pred_idx}, GT: {ground_truth_idx}"
            self.persistent_memory.add_trial(prompt=current_guidance, is_correct=is_correct, analysis=log_analysis)
            
        return pred_idx