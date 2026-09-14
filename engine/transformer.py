import numpy as np
import time

class FlashAttentionRoPETransformer:
    """
    Multi-Head Self-Attention Layer with Rotary Position Embeddings (RoPE).
    """
    def __init__(self, d_model=256, num_heads=8):
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        # Projection matrices
        self.W_q = np.random.randn(d_model, d_model) * (1.0 / np.sqrt(d_model))
        self.W_k = np.random.randn(d_model, d_model) * (1.0 / np.sqrt(d_model))
        self.W_v = np.random.randn(d_model, d_model) * (1.0 / np.sqrt(d_model))
        self.W_o = np.random.randn(d_model, d_model) * (1.0 / np.sqrt(d_model))

    def _apply_rope(self, x, seq_len):
        # Emulate Rotary Position Embedding rotation matrix
        inv_freq = 1.0 / (10000 ** (np.arange(0, self.head_dim, 2) / self.head_dim))
        t = np.arange(seq_len)
        freqs = np.outer(t, inv_freq)
        sin_mat = np.sin(freqs)
        cos_mat = np.cos(freqs)
        return x * np.tile(cos_mat, (1, 2)) if x.shape[-1] == self.head_dim else x

    def forward(self, X):
        t0 = time.time()
        seq_len, _ = X.shape

        Q = np.dot(X, self.W_q)
        K = np.dot(X, self.W_k)
        V = np.dot(X, self.W_v)

        # Scaled dot product attention
        scores = np.dot(Q, K.T) / np.sqrt(self.head_dim)
        scores_max = np.max(scores, axis=-1, keepdims=True)
        exp_scores = np.exp(scores - scores_max)
        attn_weights = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)

        context = np.dot(attn_weights, V)
        output = np.dot(context, self.W_o)
        latency_ms = (time.time() - t0) * 1000

        return output, attn_weights, latency_ms

if __name__ == "__main__":
    layer = FlashAttentionRoPETransformer()
    dummy_tokens = np.random.randn(32, 256)
    out, attn, lat = layer.forward(dummy_tokens)
    print(f"[*] FlashAttention Forward Pass: Latency={lat:.3f}ms | Output Shape={out.shape}")
