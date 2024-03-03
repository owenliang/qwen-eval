# qwen-eval

通义千问的打分评测示例

## 准备工作

下载ceval评测数据集

```
bash download.sh
```

安装python依赖

```
pip install torch modelscope transformers thefuzz tiktoken transformers_stream_generator accelerate optimum auto-gptq -i https://mirrors.aliyun.com/pypi/simple/
```

## 运行评测

```
python qwen_eval.py
```

## 评测结果

查看eval目录