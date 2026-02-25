# 基于 TensorFlow 的 GPT-2（独立实现）+ Hugging Face 导出

这次实现包含一个**单独的 GPT-2 模型文件**：`gpt2_tf.py`。

- `gpt2_tf.py`：纯 TensorFlow/Keras 的 GPT-2 结构实现（Embedding、Masked Self-Attention、MLP、Block、LM Head）。
- `train_gpt2_tf.py`：训练脚本，使用开源数据集 `tiny_shakespeare` 训练上述自实现模型，并导出 Hugging Face 可加载模型。

## 1. 安装依赖

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

## 2. 训练（基于 TensorFlow 自实现 GPT-2）

```bash
python train_gpt2_tf.py \
  --dataset-name tiny_shakespeare \
  --text-column text \
  --output-dir artifacts/tf-gpt2-tinyshakespeare \
  --epochs 1 \
  --seq-length 128 \
  --batch-size 8 \
  --n-layer 4 \
  --n-head 4 \
  --n-embd 256
```

## 3. 输出文件

训练后输出目录包含：

- `custom_model.weights.h5`：自实现 TensorFlow GPT-2 权重。
- Hugging Face 兼容文件（`config.json`, `tf_model.h5`/分片, tokenizer 文件等）。
- `train_metrics.json`：训练指标与参数。

## 4. Hugging Face 导入验证

```python
from transformers import GPT2TokenizerFast, TFGPT2LMHeadModel

model_dir = "artifacts/tf-gpt2-tinyshakespeare"
tokenizer = GPT2TokenizerFast.from_pretrained(model_dir)
model = TFGPT2LMHeadModel.from_pretrained(model_dir)

prompt = "To be, or not to be"
inputs = tokenizer(prompt, return_tensors="tf")
out = model.generate(**inputs, max_length=40)
print(tokenizer.decode(out[0], skip_special_tokens=True))
```

## 5. 关键说明

- 本仓库满足“单独实现 GPT-2 文件”的要求：`gpt2_tf.py`。
- 训练使用 TensorFlow 自实现模型，而不是直接拿 HF 的 TF GPT-2 训练。
- 训练结束后会把自实现模型权重映射导出到 Hugging Face `TFGPT2LMHeadModel` 格式，以保证可 `from_pretrained`。
