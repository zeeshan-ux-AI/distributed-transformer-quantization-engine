import numpy as np
import time

class INT8Quantizer:
    """
    Symmetric 8-Bit Quantization Aware Training (QAT) Module.
    Scales FP32 tensors to INT8 signed integer representation [-128, 127].
    """
    def __init__(self, num_bits=8):
        self.num_bits = num_bits
        self.qmin = -128
        self.qmax = 127

    def quantize(self, tensor: np.ndarray):
        max_val = np.max(np.abs(tensor))
        if max_val == 0:
            scale = 1.0
        else:
            scale = max_val / self.qmax
        
        quantized = np.clip(np.round(tensor / scale), self.qmin, self.qmax).astype(np.int8)
        return quantized, scale

    def dequantize(self, quantized_tensor: np.ndarray, scale: float):
        return (quantized_tensor.astype(np.float32) * scale)

if __name__ == "__main__":
    quantizer = INT8Quantizer()
    fp32_weights = np.random.randn(512, 512).astype(np.float32)
    q_weights, scale = quantizer.quantize(fp32_weights)
    reconstructed = quantizer.dequantize(q_weights, scale)
    error = np.mean(np.abs(fp32_weights - reconstructed))
    print(f"[*] INT8 Quantization Test Completed. Mean Abs Error: {error:.6f} | Scale: {scale:.6f}")
