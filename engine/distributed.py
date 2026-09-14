import numpy as np

class RingAllReduceSynchronizer:
    """
    Ring-AllReduce Gradient Accumulation Simulator for Parallel Worker Nodes.
    """
    def __init__(self, num_nodes=4):
        self.num_nodes = num_nodes

    def synchronize_gradients(self, gradient_tensors):
        accumulated = np.mean(gradient_tensors, axis=0)
        return accumulated

if __name__ == "__main__":
    nodes_grads = [np.random.randn(128, 128) for _ in range(4)]
    syncer = RingAllReduceSynchronizer()
    synced = syncer.synchronize_gradients(nodes_grads)
    print(f"[*] Ring-AllReduce Synchronized Gradient Tensor across 4 Nodes. Shape: {synced.shape}")
