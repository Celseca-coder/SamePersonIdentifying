在“光语启智”这类高性能算力平台（通常基于 SLURM 或 Kubernetes 容器管理）上部署完整的实验流水线，核心在于**环境隔离、后台常驻（防止断网导致中断）以及路径配置**。

你需要完成 63 组参数的网格搜索，时间跨度可能较长。以下是为你梳理的端到端标准部署流程：

### 1. 登录节点与代码拉取

首先，申请并进入你的计算节点（单卡 RTX 4090 或 A100），打开终端。

**克隆指定分支的代码：**

```bash
# 克隆仓库并直接切换到 heyalan 分支
git clone -b heyalan https://github.com/Celseca-coder/SamePersonIdentifying.git
cd SamePersonIdentifying/EvolveWithMemory

```

### 2. 环境配置与依赖安装

建议使用 Conda 创建独立的虚拟环境，避免与平台默认环境冲突。

```bash
# 创建并激活虚拟环境
conda create -n reid_env python=3.10 -y
conda activate reid_env

# 安装 vLLM 及相关依赖
pip install vllm openai transformers
pip install -r requirements.txt  # 如果代码库中有此文件

```

### 3. 模型与数据集准备

你需要确保模型和数据集路径在计算节点上是可读的。

* **模型下载**：如果你还没有下载 R-4B，可以使用 ModelScope 命令行工具下载：
```bash
pip install modelscope
python -c "from modelscope import snapshot_download; snapshot_download('YannQi/R-4B', local_dir='/data/llm/AI-ModelScope/R-4B')"

```


* **数据集验证**：检查 `Market-1501-v15.09.15` 是否已经解压在 `/home/user/GSK/heyalan/Reid/data/` 目录下。

### 4. 启动 vLLM 后端服务 (终端 A)

**注意：** 你提供的 vLLM 命令中包含了行内注释（如 `\ #注释`），这会导致 Bash 解析报错。同时，因为 40k 的上下文在 24G 的 4090 上极度消耗显存（KV Cache），设置 `gpu-memory-utilization 0.9` 是必要的，如果依然 OOM（显存溢出），你可能需要降低 `max-model-len` 或换用 A100。

为了防止 SSH 断开导致服务停止，强烈建议使用 `tmux` 或 `screen` 开启后台会话。

```bash
# 创建名为 vllm_server 的 tmux 会话
tmux new -s vllm_server

# 确保激活环境
conda activate reid_env

# 运行 vLLM（去除了会引起报错的中文行内注释）
python -m vllm.entrypoints.openai.api_server \
  --model /inspire/ssd/tenant_predefaa-9a1b-4522-bb10-8850f313be13/global_user/0053-zoutongjin/Baguette/workspaces/data/llm/AI-ModelScope/R-4B \
  --trust-remote-code \
  --port 8000 \
  --max-model-len 40960 \
  --gpu-memory-utilization 0.9

```
```bash
nohup python -m vllm.entrypoints.openai.api_server \
  --model /inspire/ssd/tenant_predefaa-9a1b-4522-bb10-8850f313be13/global_user/0053-zoutongjin/Baguette/workspaces/data/llm/AI-ModelScope/R-4B \
  --trust-remote-code \
  --port 8000 \
  --max-model-len 32768 \
  --gpu-memory-utilization 0.9 > vllm_server.log 2>&1 &
  ```

*服务启动成功后，按下 `Ctrl+b` 然后按 `d`，即可将该终端挂起在后台运行。*

### 5. 配置并运行实验网格 (终端 B)

开启另一个终端（或在平台网页 UI 中打开新终端），修改你的网格搜索脚本，开启所有的策略开关以跑满 63 个实验。

使用文本编辑器（如 `vim search_grid.py`）修改以下参数：

```python
# search_grid.py 核心配置检查
DATA_DIR = "/inspire/ssd/tenant_predefaa-9a1b-4522-bb10-8850f313be13/global_user/0053-zoutongjin/Baguette/workspaces/data/Market-1501/Market-1501-v15.09.15"  # 确保路径准确
MODEL_NAME = "/inspire/ssd/tenant_predefaa-9a1b-4522-bb10-8850f313be13/global_user/0053-zoutongjin/Baguette/workspaces/data/llm/AI-ModelScope/R-4B"
BASE_URL = "http://localhost:8000/v1"

# === 将这三个改为 True 以执行所有 63 组实验 ===
RUN_PHASE_1_LIST = False        # 阶段一：List-wise
RUN_PHASE_2_TOUR = True         # 阶段二 A：Tournament 
RUN_PHASE_2_PAIR = False         # 阶段二 B：Pair-wise

```

**运行评测：**
同样建议使用 `tmux` 将评测任务也挂在后台，防止长时间运行导致终端断开。

```bash
tmux new -s eval_client
conda activate reid_env

# 运行网格搜索，并将终端输出保存到日志文件中便于后续分析
python search_grid.py | tee experiment_logs.txt

```

---

### 💡 部署避坑指南

1. **4090 的显存预警**：R-4B 虽然只有 4B 参数（约占用 8GB 显存），但 `max-model-len 40960` 的 KV Cache 会吃掉大量显存。如果 vLLM 启动时直接报错 OOM，请将参数调为 `--max-model-len 32768` 甚至更低，或者直接申请 **A100 (40GB/80GB)** 节点进行这批实验。
2. **端口冲突**：如果光语启智的节点上 8000 端口被占用，可以改为 `--port 8080`，记得同步修改 `search_grid.py` 中的 `BASE_URL = "http://localhost:8080/v1"`。
3. **并发请求导致拥堵**：`search_grid.py` 向 vLLM 发起请求时，如果并发量过大可能导致超时。vLLM 默认能很好地处理队列，但请确保你的 `eval_script.py` 中有针对请求失败的重试机制（Retry logic）。

```

```