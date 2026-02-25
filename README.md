# TensorFlow 版 GPT-2 训练示例（可导入 Hugging Face）

这个仓库提供一个**从零开始的 TensorFlow GPT-2 训练脚本**，并使用开源数据集 `tiny_shakespeare` 作为示例。
训练完成后会输出标准 Hugging Face 目录结构，支持直接 `from_pretrained` 加载。

## 1. 环境准备

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

## 2. 开始训练

默认使用：
- 数据集：`tiny_shakespeare`
- 基础 tokenizer：`gpt2`
- 序列长度：`128`
- batch size：`8`

```bash
python train_gpt2_tf.py \
  --dataset-name tiny_shakespeare \
  --text-column text \
  --output-dir artifacts/tf-gpt2-tinyshakespeare \
  --epochs 1 \
  --seq-length 128 \
  --batch-size 8
```

> 说明：这是教学示例，默认参数偏向“可快速跑通”。
> 想提升质量请增加 `--epochs`、`--max-train-samples`、`--seq-length` 并换更大数据集。

## 3. 验证模型可从 Hugging Face Transformers 导入

```python
from transformers import GPT2TokenizerFast, TFGPT2LMHeadModel

model_dir = "artifacts/tf-gpt2-tinyshakespeare"
tokenizer = GPT2TokenizerFast.from_pretrained(model_dir)
model = TFGPT2LMHeadModel.from_pretrained(model_dir)

prompt = "To be, or not to be"
inputs = tokenizer(prompt, return_tensors="tf")
outputs = model.generate(**inputs, max_length=40)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

## 4. （可选）上传到 Hugging Face Hub

先登录：

```bash
huggingface-cli login
```

上传目录（示例仓库名）：

```bash
python - <<'PY'
from huggingface_hub import HfApi

api = HfApi()
api.upload_folder(
    folder_path="artifacts/tf-gpt2-tinyshakespeare",
    repo_id="<your-username>/tf-gpt2-tinyshakespeare",
    repo_type="model",
)
print("Uploaded!")
PY
```

上传后即可通过：

```python
from transformers import TFGPT2LMHeadModel
model = TFGPT2LMHeadModel.from_pretrained("<your-username>/tf-gpt2-tinyshakespeare")
```

## 5. 关键实现说明

- 通过 `datasets.load_dataset("tiny_shakespeare")` 读取开源文本数据。
- 使用 `GPT2TokenizerFast` 将文本 tokenization 后拼接切块。
- 构造 `tf.data.Dataset`，输入是 `input_ids + attention_mask`，标签是同一序列（Causal LM）。
- 使用 `TFGPT2LMHeadModel` + `model.fit(...)` 训练。
- 最终 `model.save_pretrained()` 与 `tokenizer.save_pretrained()` 导出标准 Hugging Face 结构。
