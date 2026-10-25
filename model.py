"""
Why Selection Matters: Mamba's Synthetic Tasks in tinygrad

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - selective_copying_batch
import numpy as np

def selective_copying_batch(n, seed, n_ctx=16, n_data=4, n_vocab=16):
    # Initialize a seeded random number generator.
    rng = np.random.default_rng(seed)

    # Draw random scores and select n_data distinct positions per row.
    # Sort the selected positions so the data tokens appear in their
    # original order when copied to the output.
    pos = np.sort(
        rng.random((n, n_ctx)).argsort(axis=1)[:, :n_data],
        axis=1
    )

    # Generate data tokens in the required range [2, n_vocab + 1].
    # This draw occurs after generating the positions to preserve draw order.
    toks = rng.integers(2, 2 + n_vocab, size=(n, n_data))

    # Initialize the input sequence with zeros (noise/blank tokens).
    x = np.zeros((n, n_ctx + n_data), dtype=np.int64)

    # Write the data tokens at their selected context positions.
    x[np.arange(n)[:, None], pos] = toks

    # Append copy cues (token 1) after the context.
    x[:, n_ctx:] = 1

    # Initialize targets to zero and place the expected copied tokens
    # at the positions corresponding to the copy cues.
    y = np.zeros((n, n_ctx + n_data), dtype=np.int64)
    y[:, n_ctx:] = toks

    # Only the final n_data positions contribute to the loss and accuracy.
    mask = np.zeros((n, n_ctx + n_data), dtype=bool)
    mask[:, n_ctx:] = True

    return x, y, mask

