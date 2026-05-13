import base64
import random
import os
from PIL import Image
import io
import dashscope
import time
import re
from functools import wraps
from openai import OpenAI
import textwrap


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
            {"role": "system", "content": "You are a ReID expert. Return ONLY the index number."},
            {"role": "user", "content": [
                {"type": "text", "text": "Query Image:"},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_query}"}},
                *gallery_content,
                {"type": "text", "text": "Match index?"}
            ]}
        ]

        response = self.client.chat.completions.create(
            model=self.model, messages=messages, max_tokens=128, temperature=0.0
        )
        content = response.choices[0].message.content.strip()
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


# HYL 调用本地LLM
class LocalReIDAgent:
    def __init__(self, api_key="local-test", model="/data/llm/AI-ModelScope/R-4B", base_url="http://localhost:8000/v1"):
        # 建立与 vLLM 的连接
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        self.log_file = "reid_inference_log.md"
        if os.path.exists(self.log_file):
            os.remove(self.log_file)
        print(f"🚀 vLLM API 模式已激活，完整日志将保存至: {self.log_file}")

    def _encode_image(self, image_path):
        img = Image.open(image_path)
        img = img.resize((224, 448), Image.Resampling.LANCZOS)
        buffered = io.BytesIO()
        img.save(buffered, format="JPEG")
        return base64.b64encode(buffered.getvalue()).decode('utf-8')

    def predict(self, query_path, gallery_paths):
        query_base64 = self._encode_image(query_path)
        matches = []   
        analyses = []
        
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(f"\n## 🔍 New Trial Analysis\n")
            f.write(f"- **Query Image**: `{os.path.basename(query_path)}`\n\n")

        for idx, g_path in enumerate(gallery_paths):
            g_base64 = self._encode_image(g_path)
            
            prompt_text = textwrap.dedent("""\
                # SYSTEM INSTRUCTION: CRITICAL OUTPUT CONSTRAINT
                You are using a limited-token environment. Your internal thought process MUST be extremely brief. You MUST output the final [VERDICT] before reaching token limits, or you will fail the task. Keep reasoning to absolute minimum.
                
                You are a highly analytical AI vision expert specializing in Pedestrian Re-Identification (Person ReID). 
                Your task is to determine if the Query image and the Gallery Candidate image contain the EXACT SAME person captured across different camera views.

                CRITICAL RULES FOR REID TASK (STRICT ENFORCEMENT):
                    1. ANTI-HALLUCINATION CONSTRAINT: NEVER invent visual features. If a Gallery image is heavily occluded, extremely blurry, completely cropped (e.g., showing only an arm), or lacks a visible human body, you MUST reject it immediately.
                    2. HARD NEGATIVE REJECTION: Actively search for irreconcilable structural differences. Just ONE definitive mismatch (e.g., long vs. short hair, denim jeans vs. suit trousers, carrying a bag vs. empty hands) means they are NOT the same person, regardless of other similarities.
                    3. FINE-GRAINED GRANULARITY: Do NOT use generic terms like "dark" or "light". Specify the exact color shade (e.g., navy blue, black, light grey) and infer the material/texture (e.g., stiff cloth, denim, soft cotton).
                    4. VIEWPOINT INVARIANCE: Ignore background environments, lighting condition shifts, and pedestrian postures. Focus ONLY on identity-intrinsic attributes.

                INSTRUCTIONS:
                    You MUST format your analysis strictly using the steps below. Use extremely concise, bulleted notes (maximum 10 words per field) to avoid token truncation. Do NOT use long paragraphs or conversational filler (e.g., "Wait, let's check...").

                    ### Step 0: Visibility & Validity Check
                    - Query: [Is a full human figure clearly visible? Yes/No]
                    - Gallery: [Is a full human figure clearly visible? Yes/No]
                    *If Gallery is 'No', STOP all further analysis and immediately output [VERDICT]: MISMATCH.*

                    ### Step 1: Head & Gender Presentation
                    - Query Hair/Gender: [Short description]
                    - Gallery Hair/Gender: [Short description]
                    - Contradiction?: [Yes/No]

                    ### Step 2: Upper Body Features
                    - Query Upper: [Color, exact sleeve length, visible logos/patterns]
                    - Gallery Upper: [Color, exact sleeve length, visible logos/patterns]
                    - Contradiction?: [Yes/No]

                    ### Step 3: Lower Body & Carried Objects
                    - Query Lower: [Exact color, inferred material (e.g. denim), length]
                    - Gallery Lower: [Exact color, inferred material (e.g. denim), length]
                    - Carried/Accessories: [Query items vs. Gallery items. Explicitly state 'None' if empty hands]
                    - Contradiction?: [Yes/No]

                    ### Step 4: Final Verdict
                    Reasoning: [Provide exactly ONE concise sentence stating the strongest reason for match or mismatch based on the contradictions found above]
                    [VERDICT]: MATCH  (or)  [VERDICT]: MISMATCH
            """)
            
            content = [
                {"type": "text", "text": prompt_text},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{query_base64}"}},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{g_base64}"}}
            ]

            try:
                # 向 vLLM 发送请求
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": content}],
                    max_tokens=4096, 
                    temperature=0.0
                )
                
                res_content = response.choices[0].message.content.strip()

                # 匹配最终格式 [VERDICT]: MATCH 或 [VERDICT]: MISMATCH
                # 在 MATCH|MISMATCH 外面加上 \[? 和 \]?，表示括号可有可无
                match_result = re.search(r'\[VERDICT\]:\s*\[?(MATCH|MISMATCH)\]?', res_content, re.IGNORECASE)

                if match_result:
                    verdict_str = match_result.group(1).upper()
                    is_match = (verdict_str == 'MATCH')
                    status_text = "✅ 匹配 (True)" if is_match else "❌ 不匹配 (False)"
                    matches.append(is_match)
                else:
                # 拦截 Token 截断或未按格式输出的情况
                    print(f"⚠️ 解析异常 (Parse Error): 疑似 Token 截断或格式丢失。输出结尾为: {res_content[-50:]}")
                    is_match = None 
                    status_text = "⚠️ 解析异常 (Parse Error)"
                    matches.append(None)

                analyses.append(res_content)

                formatted_analysis = res_content.replace('\n', '<br>')
                status_text = "✅ 匹配 (True)" if is_match else "❌ 不匹配 (False)"

                with open(self.log_file, "a", encoding="utf-8") as f:
                    f.write(f"### 🖼️ Gallery Candidate {idx}\n")
                    f.write(f"- **Path**: `{os.path.basename(g_path)}` 判定结果: **{status_text}**\n")
                    f.write(f"#### LLM Analysis:\n> {formatted_analysis}\n\n")
                    f.write("---\n")

                print(f"   -> G-{idx} 分析已记录 (判定: {status_text})")

            except Exception as e:
                # 核心：保留报错机制！如果 vLLM 挂了，这里会大声告诉你
                matches.append(False) 
                error_msg = str(e)
                analyses.append(f"Error: {error_msg}")
                
                print(f"   -> ❌ G-{idx} 发生网络/API错误: {error_msg}")

                with open(self.log_file, "a", encoding="utf-8") as f:
                    f.write(f"### 🖼️ Gallery Candidate {idx}\n")
                    f.write(f"- **Path**: `{os.path.basename(g_path)}` 判定结果: **⚠️ API 错误 (False)**\n")
                    f.write(f"#### 💥 错误详情:\n> {error_msg}\n\n")
                    f.write("---\n")

        predicted_indices = [i for i, is_match in enumerate(matches) if is_match]
        
        with open(self.log_file, "a", encoding="utf-8") as f:
            if predicted_indices:
                f.write(f"### 🏆 Final Decision: Candidates {predicted_indices}\n")
                f.write(f"Accuracy Reason: LLM determined these as Match: True\n\n")
            else:
                f.write(f"### 🏆 Final Decision: None\n")
                f.write(f"Accuracy Reason: LLM determined all candidates as Match: False (or API Failed)\n\n")

        return predicted_indices