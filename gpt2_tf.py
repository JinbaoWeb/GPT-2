#!/usr/bin/env python3
"""Standalone GPT-2 implementation in TensorFlow/Keras."""

from __future__ import annotations

from dataclasses import dataclass

import tensorflow as tf


@dataclass
class GPT2Config:
    vocab_size: int
    n_positions: int = 1024
    n_embd: int = 256
    n_layer: int = 4
    n_head: int = 4
    dropout: float = 0.1
    layer_norm_epsilon: float = 1e-5


class CausalSelfAttention(tf.keras.layers.Layer):
    def __init__(self, config: GPT2Config, **kwargs):
        super().__init__(**kwargs)
        if config.n_embd % config.n_head != 0:
            raise ValueError("n_embd must be divisible by n_head")
        self.n_head = config.n_head
        self.head_dim = config.n_embd // config.n_head
        self.dropout_rate = config.dropout

        self.c_attn = tf.keras.layers.Dense(3 * config.n_embd, name="c_attn")
        self.c_proj = tf.keras.layers.Dense(config.n_embd, name="c_proj")
        self.attn_dropout = tf.keras.layers.Dropout(config.dropout)
        self.resid_dropout = tf.keras.layers.Dropout(config.dropout)

    def _split_heads(self, x: tf.Tensor) -> tf.Tensor:
        b, t, c = tf.unstack(tf.shape(x))
        x = tf.reshape(x, [b, t, self.n_head, self.head_dim])
        return tf.transpose(x, [0, 2, 1, 3])

    def _merge_heads(self, x: tf.Tensor) -> tf.Tensor:
        b, h, t, d = tf.unstack(tf.shape(x))
        x = tf.transpose(x, [0, 2, 1, 3])
        return tf.reshape(x, [b, t, h * d])

    def _causal_mask(self, seq_len: tf.Tensor) -> tf.Tensor:
        i = tf.range(seq_len)[:, None]
        j = tf.range(seq_len)[None, :]
        return i >= j

    def call(self, x: tf.Tensor, attention_mask: tf.Tensor | None = None, training: bool = False):
        qkv = self.c_attn(x)
        q, k, v = tf.split(qkv, num_or_size_splits=3, axis=-1)

        q = self._split_heads(q)
        k = self._split_heads(k)
        v = self._split_heads(v)

        scale = tf.cast(self.head_dim, q.dtype) ** -0.5
        attn_scores = tf.matmul(q, k, transpose_b=True) * scale  # [B,H,T,T]

        t = tf.shape(attn_scores)[-1]
        causal = self._causal_mask(t)
        causal = tf.reshape(causal, [1, 1, t, t])
        neg_inf = tf.cast(-1e9, attn_scores.dtype)
        attn_scores = tf.where(causal, attn_scores, neg_inf)

        if attention_mask is not None:
            # attention_mask: [B,T], 1 for valid tokens.
            mask = tf.cast(attention_mask[:, None, None, :], attn_scores.dtype)
            attn_scores += (1.0 - mask) * neg_inf

        attn_weights = tf.nn.softmax(attn_scores, axis=-1)
        attn_weights = self.attn_dropout(attn_weights, training=training)

        y = tf.matmul(attn_weights, v)
        y = self._merge_heads(y)
        y = self.c_proj(y)
        y = self.resid_dropout(y, training=training)
        return y


class MLP(tf.keras.layers.Layer):
    def __init__(self, config: GPT2Config, **kwargs):
        super().__init__(**kwargs)
        hidden_dim = 4 * config.n_embd
        self.c_fc = tf.keras.layers.Dense(hidden_dim, name="c_fc")
        self.c_proj = tf.keras.layers.Dense(config.n_embd, name="c_proj")
        self.dropout = tf.keras.layers.Dropout(config.dropout)

    def call(self, x: tf.Tensor, training: bool = False):
        x = self.c_fc(x)
        x = tf.nn.gelu(x, approximate=True)
        x = self.c_proj(x)
        x = self.dropout(x, training=training)
        return x


class GPT2Block(tf.keras.layers.Layer):
    def __init__(self, config: GPT2Config, **kwargs):
        super().__init__(**kwargs)
        self.ln_1 = tf.keras.layers.LayerNormalization(epsilon=config.layer_norm_epsilon, name="ln_1")
        self.attn = CausalSelfAttention(config, name="attn")
        self.ln_2 = tf.keras.layers.LayerNormalization(epsilon=config.layer_norm_epsilon, name="ln_2")
        self.mlp = MLP(config, name="mlp")

    def call(self, x: tf.Tensor, attention_mask: tf.Tensor | None = None, training: bool = False):
        x = x + self.attn(self.ln_1(x), attention_mask=attention_mask, training=training)
        x = x + self.mlp(self.ln_2(x), training=training)
        return x


class GPT2LMHeadModel(tf.keras.Model):
    """TensorFlow GPT-2 LM head model (from-scratch implementation)."""

    def __init__(self, config: GPT2Config, **kwargs):
        super().__init__(**kwargs)
        self.config = config
        self.wte = tf.keras.layers.Embedding(config.vocab_size, config.n_embd, name="wte")
        self.wpe = tf.keras.layers.Embedding(config.n_positions, config.n_embd, name="wpe")
        self.drop = tf.keras.layers.Dropout(config.dropout)
        self.h = [GPT2Block(config, name=f"h_{i}") for i in range(config.n_layer)]
        self.ln_f = tf.keras.layers.LayerNormalization(epsilon=config.layer_norm_epsilon, name="ln_f")

    def call(self, inputs, training: bool = False):
        input_ids = inputs["input_ids"] if isinstance(inputs, dict) else inputs
        attention_mask = inputs.get("attention_mask") if isinstance(inputs, dict) else None

        seq_len = tf.shape(input_ids)[1]
        pos = tf.range(seq_len)[None, :]

        x = self.wte(input_ids) + self.wpe(pos)
        x = self.drop(x, training=training)

        for block in self.h:
            x = block(x, attention_mask=attention_mask, training=training)

        x = self.ln_f(x)
        # Weight tying with token embeddings
        logits = tf.matmul(x, self.wte.embeddings, transpose_b=True)
        return logits


def causal_lm_loss(labels: tf.Tensor, logits: tf.Tensor) -> tf.Tensor:
    """Standard next-token loss for causal LM."""
    shift_labels = labels[:, 1:]
    shift_logits = logits[:, :-1, :]
    loss = tf.keras.losses.sparse_categorical_crossentropy(
        shift_labels, shift_logits, from_logits=True
    )
    return tf.reduce_mean(loss)
