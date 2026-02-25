#!/usr/bin/env python3
"""Train standalone TensorFlow GPT-2 and export to Hugging Face format."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class TrainConfig:
    dataset_name: str = "tiny_shakespeare"
    dataset_config: str | None = None
    text_column: str = "text"
    tokenizer_name: str = "gpt2"
    output_dir: str = "artifacts/tf-gpt2-tinyshakespeare"
    seq_length: int = 128
    batch_size: int = 8
    learning_rate: float = 3e-4
    epochs: int = 1
    max_train_samples: int = 20000
    val_split: float = 0.02
    random_seed: int = 42
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 256
    dropout: float = 0.1


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(description="训练 TensorFlow 自实现 GPT-2，并导出 Hugging Face 模型")
    parser.add_argument("--dataset-name", default="tiny_shakespeare")
    parser.add_argument("--dataset-config", default=None)
    parser.add_argument("--text-column", default="text")
    parser.add_argument("--tokenizer-name", default="gpt2")
    parser.add_argument("--output-dir", default="artifacts/tf-gpt2-tinyshakespeare")
    parser.add_argument("--seq-length", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max-train-samples", type=int, default=20000)
    parser.add_argument("--val-split", type=float, default=0.02)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--n-layer", type=int, default=4)
    parser.add_argument("--n-head", type=int, default=4)
    parser.add_argument("--n-embd", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)
    args = parser.parse_args()

    return TrainConfig(
        dataset_name=args.dataset_name,
        dataset_config=args.dataset_config,
        text_column=args.text_column,
        tokenizer_name=args.tokenizer_name,
        output_dir=args.output_dir,
        seq_length=args.seq_length,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        epochs=args.epochs,
        max_train_samples=args.max_train_samples,
        val_split=args.val_split,
        random_seed=args.random_seed,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        dropout=args.dropout,
    )


def _import_deps():
    import numpy as np
    import tensorflow as tf
    from datasets import load_dataset
    from transformers import GPT2Config as HFGPT2Config
    from transformers import GPT2TokenizerFast, TFGPT2LMHeadModel

    from gpt2_tf import GPT2Config, GPT2LMHeadModel, causal_lm_loss

    return (
        np,
        tf,
        load_dataset,
        GPT2TokenizerFast,
        HFGPT2Config,
        TFGPT2LMHeadModel,
        GPT2Config,
        GPT2LMHeadModel,
        causal_lm_loss,
    )


def make_lm_examples(np, token_ids: list[int], seq_length: int):
    if len(token_ids) <= seq_length:
        raise ValueError(f"Need token count > seq_length, got {len(token_ids)} <= {seq_length}")

    total = (len(token_ids) - 1) // seq_length
    trimmed = token_ids[: total * seq_length]
    input_ids = np.asarray(trimmed, dtype=np.int32).reshape(total, seq_length)
    attention_mask = np.ones_like(input_ids, dtype=np.int32)
    labels = input_ids.copy()
    return input_ids, attention_mask, labels


def build_tf_dataset(tf, input_ids, attention_mask, labels, batch_size: int, shuffle: bool):
    ds = tf.data.Dataset.from_tensor_slices(
        ({"input_ids": input_ids, "attention_mask": attention_mask}, labels)
    )
    if shuffle:
        ds = ds.shuffle(buffer_size=max(len(input_ids), 1), reshuffle_each_iteration=True)
    return ds.batch(batch_size, drop_remainder=False).prefetch(tf.data.AUTOTUNE)


def copy_custom_to_hf(custom_model, hf_model):
    hf_model.transformer.wte.set_weights(custom_model.wte.get_weights())
    hf_model.transformer.wpe.set_weights(custom_model.wpe.get_weights())
    hf_model.transformer.ln_f.set_weights(custom_model.ln_f.get_weights())

    for i, custom_block in enumerate(custom_model.h):
        hf_block = hf_model.transformer.h[i]
        hf_block.ln_1.set_weights(custom_block.ln_1.get_weights())
        hf_block.attn.c_attn.set_weights(custom_block.attn.c_attn.get_weights())
        hf_block.attn.c_proj.set_weights(custom_block.attn.c_proj.get_weights())
        hf_block.ln_2.set_weights(custom_block.ln_2.get_weights())
        hf_block.mlp.c_fc.set_weights(custom_block.mlp.c_fc.get_weights())
        hf_block.mlp.c_proj.set_weights(custom_block.mlp.c_proj.get_weights())


def main() -> None:
    cfg = parse_args()
    (
        np,
        tf,
        load_dataset,
        GPT2TokenizerFast,
        HFGPT2Config,
        TFGPT2LMHeadModel,
        GPT2Config,
        GPT2LMHeadModel,
        causal_lm_loss,
    ) = _import_deps()

    tf.keras.utils.set_random_seed(cfg.random_seed)

    tokenizer = GPT2TokenizerFast.from_pretrained(cfg.tokenizer_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = load_dataset(cfg.dataset_name, cfg.dataset_config)
    split = "train" if "train" in dataset else list(dataset.keys())[0]
    raw_texts = dataset[split][cfg.text_column][: cfg.max_train_samples]
    merged_text = "\n".join(raw_texts)
    token_ids = tokenizer(merged_text, add_special_tokens=False)["input_ids"]

    input_ids, attention_mask, labels = make_lm_examples(np, token_ids, cfg.seq_length)
    num_examples = len(input_ids)
    val_size = max(1, int(num_examples * cfg.val_split))
    train_size = num_examples - val_size
    if train_size <= 0:
        raise ValueError("Not enough train samples after split; increase max_train_samples.")

    train_ds = build_tf_dataset(
        tf,
        input_ids[:train_size],
        attention_mask[:train_size],
        labels[:train_size],
        cfg.batch_size,
        shuffle=True,
    )
    val_ds = build_tf_dataset(
        tf,
        input_ids[train_size:],
        attention_mask[train_size:],
        labels[train_size:],
        cfg.batch_size,
        shuffle=False,
    )

    model_cfg = GPT2Config(
        vocab_size=tokenizer.vocab_size,
        n_positions=cfg.seq_length,
        n_embd=cfg.n_embd,
        n_layer=cfg.n_layer,
        n_head=cfg.n_head,
        dropout=cfg.dropout,
    )
    model = GPT2LMHeadModel(model_cfg)
    optimizer = tf.keras.optimizers.Adam(learning_rate=cfg.learning_rate)
    model.compile(optimizer=optimizer, loss=causal_lm_loss)

    print(
        "Train TensorFlow GPT-2: "
        f"train={train_size}, val={val_size}, layers={cfg.n_layer}, heads={cfg.n_head}, embd={cfg.n_embd}"
    )
    history = model.fit(train_ds, validation_data=val_ds, epochs=cfg.epochs)

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save native TensorFlow checkpoint for standalone implementation
    model.save_weights(output_dir / "custom_model.weights.h5")

    # Export Hugging Face-compatible model weights
    hf_cfg = HFGPT2Config(
        vocab_size=tokenizer.vocab_size,
        n_positions=cfg.seq_length,
        n_ctx=cfg.seq_length,
        n_embd=cfg.n_embd,
        n_layer=cfg.n_layer,
        n_head=cfg.n_head,
        resid_pdrop=cfg.dropout,
        embd_pdrop=cfg.dropout,
        attn_pdrop=cfg.dropout,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    hf_model = TFGPT2LMHeadModel(hf_cfg)
    dummy = tf.constant([[tokenizer.eos_token_id] * min(cfg.seq_length, 2)], dtype=tf.int32)
    _ = hf_model(input_ids=dummy)
    _ = model({"input_ids": dummy, "attention_mask": tf.ones_like(dummy)})
    copy_custom_to_hf(model, hf_model)

    hf_model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    metrics = {
        "train_config": asdict(cfg),
        "final_loss": float(history.history["loss"][-1]),
        "final_val_loss": float(history.history.get("val_loss", [0.0])[-1]),
    }
    (output_dir / "train_metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"Saved custom TensorFlow GPT-2 + Hugging Face export to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
