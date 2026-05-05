## 1. Task Introduction
**Person Re-Identification (ReID)** is a task within computer vision focused on image retrieval and fine-grained recognition. The objective is to identify the **"same individual"** across surveillance footage captured by different cameras, at different times, and from different viewpoints. This typically occurs in multi-camera environments—such as shopping malls, campuses, subway stations, or urban security systems. When a person exits the field of view of Camera A, the system must "re-identify" them in the feeds of Cameras B, C, or D.

Given that modern Multimodal Large Language Models (MLLMs) possess enhanced image recognition capabilities, this task explores whether MLLMs can be used directly to solve ReID. Which modalities can current models handle effectively, and which remain difficult to process?

Due to the constraints of token limits in standard chat interfaces, the use of **API interfaces** is required to complete this task. The test datasets also differ from traditional methods; you are expected to perform the evaluation on the provided test set. The primary evaluation metric is **Rank-1 accuracy** (whether the sample most similar to the query image belongs to the same ID).

### Task Definition:
* **Input:** A single query image or a set of query images, along with a gallery (database) of candidate images.
* **Output:** Images identified as the same person as the query (presented as one or more images sorted by similarity).

---

## 2. Recommended Methods
* It is recommended to use APIs for **Qwen3** or **GPT-4o**, or to deploy lightweight models locally. For API implementation tutorials, refer to frameworks like **LangChain** or **CrewAI**. An API Key is required for connection.
* Optimize model performance by **adjusting prompts**. Additionally, flexibly adjust the number of input images per dialogue round based on token limits to determine which parameter configurations yield the best results.
* Attempt to utilize **image retrieval toolsets** integrated with LangChain to handle batch data processing.

---

## 3. Evaluation Datasets
* **For Testing & Debugging:** [Market-1501](https://www.kaggle.com/datasets/pengcw1/market-1501)
* **Validation Dataset:** Contains a multimodal dataset. Each modality includes 4 Query IDs and 20 Gallery images across 5 distinct modalities.

---

## 4. Evaluation Examples
* **Intra-Modality Evaluation:** When the query size is ( ) and the gallery size is ( ), the Rank-1 success rate of the MLLM is ( ).
* **Cross-Modality Mixed Evaluation:** When the query size is ( ) and the gallery size is ( ), the Rank-1 success rate of the MLLM is ( ).

---

## 5. Scoring Rubric
* **API Execution (50%):** Successfully utilize the API to return image IDs and obtain final results.
* **Analysis Report (50%):** Identify which types of images the model processes correctly versus incorrectly, and provide a detailed report.
    * *Note: This task focuses on evaluating the visual capabilities of LLMs. Accuracy is not the sole priority; logical rigor in the report and the fairness of the experimental setup are critical.*

---

## 6. Bonus Points
* Explore tools or techniques that can enhance the recognition accuracy of the LLM.
* Given the existence of mature, traditional ReID methods, investigate whether the LLM can be prompted to select and utilize the correct method to complete the task.