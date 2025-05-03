import numpy as np

def weighted_cross_entropy(probs, target_indices, weights, eps=1e-12):
    """
    Weighted cross-entropy loss for sequence prediction.

    Args:
        probs: np.ndarray of shape (seq_len, vocab_size), predicted probabilities.
        target_indices: np.ndarray of shape (seq_len,), ground truth token indices.
        weights: np.ndarray of shape (seq_len,), weights per position.
        eps: float, small value to avoid log(0).

    Returns:
        Scalar weighted loss.
    """
    # make sure probabilities stay in certain range
    probs = np.clip(probs, eps, 1. - eps)
    correct_probs = probs[np.arange(len(target_indices)), target_indices]
    log_losses = -np.log(correct_probs)
    
    weighted_loss = np.sum(weights * log_losses) / np.sum(weights)
    return weighted_loss

# Example setup
vocab = ['A', 'C', 'G', 'T']
token_to_id = {ch: i for i, ch in enumerate(vocab)}
target_seq = "ACGTA"
target_indices = np.array([token_to_id[ch] for ch in target_seq])

# Simulated model probabilities: these would usually be given by the transformer
probs = np.array([
    [0.9, 0.05, 0.03, 0.02],
    [0.1, 0.8, 0.05, 0.05],
    [0.1, 0.1, 0.75, 0.05],
    [0.25, 0.25, 0.25, 0.25],
    [0.05, 0.05, 0.05, 0.85]
])

# Weight vector: higher weights in conserved regions (e.g., positions 1 and 3)
weights = np.array([2.0, 1.0, 2.0, 1.0, 1.0])

loss = weighted_cross_entropy(probs, target_indices, weights)
print("Weighted cross-entropy loss:", loss)
