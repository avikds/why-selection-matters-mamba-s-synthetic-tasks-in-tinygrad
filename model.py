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

# Step 2 - induction_heads_batch
def induction_heads_batch(n, seed, seq_len=20, n_vocab=16):
    # Initialize a seeded random number generator.
    rng = np.random.default_rng(seed)

    # Draw regular tokens first, following the specified RNG draw order.
    x = rng.integers(0, n_vocab, size=(n, seq_len))

    # Choose one trigger position per sequence.
    # The upper bound is exclusive, so p + 1 <= seq_len - 2.
    p = rng.integers(0, seq_len - 2, size=n)

    # Draw the value that must be recalled for each sequence.
    b = rng.integers(0, n_vocab, size=n)

    # Plant the trigger and its associated value in each sequence.
    rows = np.arange(n)
    x[rows, p] = n_vocab
    x[rows, p + 1] = b

    # Place the trigger again at the final position.
    x[:, -1] = n_vocab

    # Initialize targets to zero; the final target is the value
    # that appeared immediately after the earlier trigger.
    y = np.zeros((n, seq_len), dtype=np.int64)
    y[:, -1] = b

    # Only the final position contributes to loss and accuracy.
    mask = np.zeros((n, seq_len), dtype=bool)
    mask[:, -1] = True

    return (
        x.astype(np.int64, copy=False),
        y,
        mask,
    )

# Step 3 - encode_batch
def encode_batch(x, y, mask, vocab):
    # Convert token IDs into one-hot input vectors.
    # Shape: (n, L, vocab), dtype: float32.
    inputs = np.eye(vocab, dtype=np.float32)[x]

    # Convert target token IDs into one-hot vectors.
    # Shape: (n, L, vocab), dtype: float32.
    targets = np.eye(vocab, dtype=np.float32)[y]

    # Zero out target vectors at positions that are not answers.
    # Expanding mask adds a final dimension for broadcasting.
    targets *= mask[..., None].astype(np.float32)

    return inputs, targets

# Step 4 - masked_accuracy
def masked_accuracy(logits, y, mask):
    # Convert a tinygrad Tensor to a NumPy array when necessary.
    if hasattr(logits, "numpy"):
        logits = logits.numpy()

    # Predict the token with the highest logit at each sequence position.
    predictions = np.argmax(logits, axis=-1)

    # Ensure the mask is Boolean.
    mask = np.asarray(mask, dtype=bool)

    # Return 0.0 when there are no positions to evaluate.
    if not np.any(mask):
        return 0.0

    # Compare predictions with the expected token IDs only at
    # positions where the task requires an answer.
    correct = predictions[mask] == np.asarray(y)[mask]

    # Return the fraction of correctly predicted answer positions
    # as a standard Python float.
    return float(np.mean(correct))

# Step 5 - scan_reference
def scan_reference(a, b):
    # Convert inputs to float64 to ensure the recurrence and output
    # use the required precision.
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)

    # Initialize the hidden state to zero for every batch item and
    # every trailing dimension, excluding the time axis (axis 1).
    h = np.zeros(a.shape[:1] + a.shape[2:], dtype=np.float64)

    # Allocate the output array to store every hidden state.
    out = np.empty(a.shape, dtype=np.float64)

    # Sequentially evaluate h_t = a_t * h_(t-1) + b_t.
    # Broadcasting handles all dimensions other than the time axis.
    for t in range(a.shape[1]):
        h = a[:, t, ...] * h + b[:, t, ...]
        out[:, t, ...] = h

    return out

# Step 6 - scan_combine
SCAN_IDENTITY = (1.0, 0.0)


def scan_combine(left, right):
    # Compose two affine transformations:
    # left:  h -> a1 * h + b1
    # right: h -> a2 * h + b2
    #
    # Applying left first and then right gives:
    # h -> a2 * (a1 * h + b1) + b2
    #   = (a1 * a2) * h + (a2 * b1 + b2)

    a1, b1 = left
    a2, b2 = right

    # Use only multiplication and addition so the operation supports
    # Python scalars, NumPy arrays, and tinygrad Tensors.
    return a1 * a2, a2 * b1 + b2

# Step 7 - associative_scan
import math
from tinygrad import Tensor


def associative_scan(a, b):
    # Time is along axis 1.
    seq_len = a.shape[1]
    d = 1

    # Hillis-Steele scan: double the distance at each level,
    # giving logarithmic scan depth.
    while d < seq_len:
        # Shift a right by d positions, padding with the identity
        # multiplier 1.0 at the beginning.
        a_prev = Tensor.cat(
            a[:, :d] * 0.0 + 1.0,
            a[:, :-d],
            dim=1,
        )

        # Shift b right by d positions, padding with 0.0.
        b_prev = Tensor.cat(
            b[:, :d] * 0.0,
            b[:, :-d],
            dim=1,
        )

        # Compose the earlier affine pair with the current pair:
        # a_new[t] = a_prev[t] * a[t]
        # b_new[t] = a[t] * b_prev[t] + b[t]
        #
        # Keep the current a and b unchanged until both updates
        # have been computed.
        a_new = (a_prev * a).contiguous()
        b_new = (a * b_prev + b).contiguous()

        # Advance to the next scan level.
        a, b = a_new, b_new
        d *= 2

    # b now contains every state h_t, with h_(-1) = 0.
    return b

# Step 8 - Dense
class Dense:
    def __init__(self, d_in, d_out, rng, bias=True):
        # Use a fan-in-scaled uniform initialization for stable training.
        bound = 1.0 / math.sqrt(d_in)

        # Initialize weights with shape (d_in, d_out) and dtype float32.
        self.weight = Tensor(
            rng.uniform(-bound, bound, size=(d_in, d_out)).astype(np.float32)
        )

        # Initialize the bias with shape (d_out,) when requested.
        if bias:
            self.bias = Tensor(
                rng.uniform(-bound, bound, size=(d_out,)).astype(np.float32)
            )
        else:
            self.bias = None

    def __call__(self, x):
        # Apply the linear transformation without transposing the weights.
        out = x @ self.weight

        # Add the bias when this layer was configured to use one.
        if self.bias is not None:
            out = out + self.bias

        return out

# Step 9 - ssm_init
def ssm_init(rng, d_inner, d_state, dt_min=0.001, dt_max=0.1):
    # S4D-Real initialization:
    # A_n = -(n + 1), stored as log(n + 1) so A = -exp(A_log).
    # Use the same sequence of values for every inner channel.
    A_log = np.broadcast_to(
        np.log(np.arange(1, d_state + 1, dtype=np.float32)),
        (d_inner, d_state),
    ).copy().astype(np.float32)

    # Draw positive step sizes from a log-uniform distribution.
    # This is the only random draw in this initializer.
    dt = np.exp(
        rng.uniform(
            math.log(dt_min),
            math.log(dt_max),
            size=d_inner,
        )
    )

    # Apply the inverse softplus so softplus(dt_bias) approximately
    # recovers the sampled step sizes:
    # inverse_softplus(dt) = dt + log(1 - exp(-dt)).
    dt_bias = dt + np.log(-np.expm1(-dt))
    dt_bias = dt_bias.astype(np.float32)

    # D is the learnable skip-connection parameter, initialized to ones.
    D = np.ones((d_inner,), dtype=np.float32)

    return {
        "A_log": A_log,
        "dt_bias": dt_bias,
        "D": D,
    }

# Step 10 - discretize
def discretize(delta, A, B, method="euler"):
    # Zero-order hold for the state-transition coefficient:
    # a_bar = exp(delta * A).
    a_bar = (delta * A).exp()

    # Discretize B using the selected method.
    if method == "euler":
        # First-order Euler discretization: b_bar = delta * B.
        b_bar = delta * B

    elif method == "zoh":
        # Exact zero-order-hold discretization:
        # b_bar = ((exp(delta * A) - 1) / A) * B.
        b_bar = ((a_bar - 1.0) / A) * B

    else:
        raise ValueError(
            f"Unsupported discretization method: {method!r}. "
            "Use 'euler' or 'zoh'."
        )

    return a_bar, b_bar

