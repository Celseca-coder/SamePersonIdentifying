from modelscope import snapshot_download

# 指定模型 ID 和下载路径
model_dir = snapshot_download(
    'AI-ModelScope/R-4B', 
    cache_dir='/data/llm', 
    revision='master'
)

print(f"模型已成功下载到: {model_dir}")