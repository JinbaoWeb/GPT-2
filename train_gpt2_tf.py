#!/usr/bin/env python3
"""TensorFlow GPT-2 training example using an open-source dataset.

This script trains a GPT-2 style causal language model with TensorFlow/Keras,
then exports artifacts in Hugging Face format via `save_pretrained`.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tensorflow as tf
from datasets import load_dataset
from transformers import GPT2Config, GPT2TokenizerFast, TFGPT2LMHeadModel


@dataclass
class TrainConfig:
    dataset_name: str = "tiny_shakespeare"
    dataset_config: str | None = None
    text_column: str = "text"
    model_name_or_path: str = "gpt2"
    output_dir: str = "artifacts/tf-gpt2-tinyshakespeare"
    seq_length: int = 128
    batch_size: int = 8
    learning_rate: float = 5e-5
    epochs: int = 1
    max_train_samples: int = 20000
    val_split: float = 0.02
    random_seed: int = 42


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(description="Train GPT-2 with TensorFlow")
    parser.add_argument("--dataset-name", default="tiny_shakespeare")
    parser.add_argument("--dataset-config", default=None)
    parser.add_argument("--text-column", default="text")
    parser.add_argument("--model-name-or-path", default="gpt2")
    parser.add_argument("--output-dir", default="artifacts/tf-gpt2-tinyshakespeare")
    parser.add_argument("--seq-length", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max-train-samples", type=int, default=20000)
    parser.add_argument("--val-split", type=float, default=0.02)
    parser.add_argument("--random-seed", type=int, default=42)

    args = parser.parse_args()
    return TrainConfig(
        dataset_name=args.dataset_name,
        dataset_config=args.dataset_config,
        text_column=args.text_column,
        model_name_or_path=args.model_name_or_path,
        output_dir=args.output_dir,
        seq_length=args.seq_length,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        epochs=args.epochs,
        max_train_samples=args.max_train_samples,
        val_split=args.val_split,
        random_seed=args.random_seed,
    )


def make_lm_examples(token_ids: list[int], seq_length: int) -> tuple[np.ndarray, np.ndarray]:
    """Create fixed-length language-modeling sequences.

    Returns arrays with shape [num_examples, seq_length].
    Labels are identical to inputs for causal LM training.
    """
    if len(token_ids) <= seq_length:
        raise ValueError(
            f"Token count ({len(token_ids)}) must be greater than seq_length ({seq_length})."
        )

    chunks = []
    for start in range(0, len(token_ids) - seq_length, seq_length):
        chunks.append(token_ids[start : start + seq_length])

    input_ids = np.asarray(chunks, dtype=np.int32)
    attention_mask = np.ones_like(input_ids, dtype=np.int32)
    return input_ids, attention_mask


def build_tf_dataset(
    input_ids: np.ndarray,
    attention_mask: np.ndarray,
    batch_size: int,
    shuffle: bool,
) -> tf.data.Dataset:
    labels = np.copy(input_ids)
    ds = tf.data.Dataset.from_tensor_slices(
        ({"input_ids": input_ids, "attention_mask": attention_mask}, labels)
    )
    if shuffle:
        ds = ds.shuffle(buffer_size=len(input_ids), reshuffle_each_iteration=True)
    ds = ds.batch(batch_size, drop_remainder=True).prefetch(tf.data.AUTOTUNE)
    return ds


def main() -> None:
    cfg = parse_args()
    tf.keras.utils.set_random_seed(cfg.random_seed)

    tokenizer = GPT2TokenizerFast.from_pretrained(cfg.model_name_or_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading dataset: {cfg.dataset_name}")
    dataset = load_dataset(cfg.dataset_name, cfg.dataset_config)

    train_split_name = "train" if "train" in dataset else list(dataset.keys())[0]
    raw_texts = dataset[train_split_name][cfg.text_column][: cfg.max_train_samples]
    merged_text = "\n".join(raw_texts)

    encoded = tokenizer(
        merged_text,
        add_special_tokens=False,
        return_attention_mask=False,
    )
    token_ids = encoded["input_ids"]
    input_ids, attention_mask = make_lm_examples(token_ids, cfg.seq_length)

    num_examples = len(input_ids)
    val_size = max(1, int(num_examples * cfg.val_split))
    train_size = num_examples - val_size

    train_input_ids = input_ids[:train_size]
    train_attention_mask = attention_mask[:train_size]
    val_input_ids = input_ids[train_size:]
    val_attention_mask = attention_mask[train_size:]

    train_ds = build_tf_dataset(train_input_ids, train_attention_mask, cfg.batch_size, shuffle=True)
    val_ds = build_tf_dataset(val_input_ids, val_attention_mask, cfg.batch_size, shuffle=False)

    model_config = GPT2Config.from_pretrained(cfg.model_name_or_path)
    model_config.vocab_size = tokenizer.vocab_size
    model_config.n_positions = max(model_config.n_positions, cfg.seq_length)
    model = TFGPT2LMHeadModel(model_config)

    optimizer = tf.keras.optimizers.Adam(learning_rate=cfg.learning_rate)
    model.compile(optimizer=optimizer)

    print(
        f"Start training: train_examples={train_size}, val_examples={val_size}, "
        f"seq_length={cfg.seq_length}, batch_size={cfg.batch_size}"
    )
    history = model.fit(train_ds, validation_data=val_ds, epochs=cfg.epochs)
    final_loss = history.history["loss"][-1]
    print(f"Training done. final_loss={final_loss:.4f}")

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    print(f"Saved model/tokenizer to: {output_dir.resolve()}")
    print("Now you can load it via transformers: TFGPT2LMHeadModel.from_pretrained(output_dir)")


if __name__ == "__main__":
    main()
