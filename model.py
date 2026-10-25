"""
Why Selection Matters: Mamba's Synthetic Tasks in tinygrad

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - selective_copying_batch
import math
import numpy as np
from tinygrad import Tensor, TinyJit, nn
from tinygrad.nn.state import get_parameters

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

# Step 11 - LTISSM
class LTISSM:
    def __init__(self, d, n, rng, dt_min=0.001, dt_max=0.1):
        # Initialize the SSM parameters first, preserving the required RNG draw order.
        p = ssm_init(rng, d, n, dt_min, dt_max)

        # Draw B and C from the seeded NumPy generator.
        B = rng.normal(0.0, 1.0, size=(d, n)).astype(np.float32)
        C = (
            rng.normal(0.0, 1.0, size=(d, n)) / math.sqrt(n)
        ).astype(np.float32)

        # Store all model parameters as float32 tinygrad Tensors.
        self.A_log = Tensor(p["A_log"].astype(np.float32))
        self.dt = Tensor(p["dt_bias"].astype(np.float32))
        self.B = Tensor(B)
        self.C = Tensor(C)
        self.D = Tensor(p["D"].astype(np.float32))

        self.d = d
        self.n = n

    def discrete(self):
        # Recover the continuous-time diagonal state matrix:
        # A = -exp(A_log).
        A = -self.A_log.exp()

        # Recover positive step sizes using softplus.
        # Shape: (d,).
        delta = self.dt.softplus()

        # Apply Euler discretization with broadcasting over state dimensions.
        # Both returned tensors have shape (d, n).
        a_bar, b_bar = discretize(delta[:, None], A, self.B)

        return a_bar, b_bar

    def __call__(self, x):
        # Input x has shape (batch, L, d).
        batch, seq_len, d = x.shape

        # Obtain the time-invariant discrete recurrence coefficients.
        a_bar, b_bar = self.discrete()

        # Expand a_bar across batch and sequence dimensions.
        # Shape: (batch, L, d, n).
        a = a_bar.reshape(1, 1, d, self.n).expand(
            batch, seq_len, d, self.n
        )

        # The input-dependent forcing term is b_bar * x_t.
        # Shape: (batch, L, d, n).
        b = b_bar * x.reshape(batch, seq_len, d, 1)

        # Compute every recurrent state with the differentiable
        # associative scan implemented in Part 2.
        h = associative_scan(a, b)

        # Read out the states and add the per-channel skip connection:
        # y_t = sum_n(C * h_t) + D * x_t.
        y = (h * self.C).sum(axis=-1) + self.D * x

        return y

# Step 12 - lti_kernel
def lti_kernel(ssm, L):
    # Obtain the time-invariant discrete recurrence coefficients.
    a_bar, b_bar = ssm.discrete()

    # Convert tinygrad Tensors to NumPy float32 arrays.
    a_bar = a_bar.numpy().astype(np.float32)
    b_bar = b_bar.numpy().astype(np.float32)
    C = ssm.C.numpy().astype(np.float32)

    # K[c, k] = sum_n C[c, n] * a_bar[c, n]**k * b_bar[c, n].
    # Each column represents the contribution from one time lag.
    K = np.empty((ssm.d, L), dtype=np.float32)

    for k in range(L):
        K[:, k] = np.sum(
            C * np.power(a_bar, k) * b_bar,
            axis=1,
        )

    return K


def lti_conv(ssm, x):
    # Convert the input to a NumPy float32 array.
    # Expected shape: (batch, L, d).
    x = np.asarray(x, dtype=np.float32)
    batch, L, d = x.shape

    # Compute the fixed convolution kernel and skip parameter.
    K = lti_kernel(ssm, L)
    D = ssm.D.numpy().astype(np.float32)

    # Initialize the convolution output.
    y = np.zeros((batch, L, d), dtype=np.float32)

    # Apply causal convolution:
    # y_t = sum_{k=0}^t K[:, k] * x_{t-k} + D * x_t.
    for t in range(L):
        for k in range(t + 1):
            y[:, t, :] += x[:, t - k, :] * K[:, k]

    # Add the channel-wise skip connection.
    y += x * D.reshape(1, 1, d)

    return y

# Step 13 - S6
class S6:
    def __init__(self, d, n, dt_rank, rng, dt_min=0.001, dt_max=0.1):
        # Initialize the SSM parameters first, preserving the specified RNG draw order.
        p = ssm_init(rng, d, n, dt_min, dt_max)

        # Project each input token into:
        # dt_rank components for the step-size projection,
        # n components for the input-dependent B,
        # n components for the input-dependent C.
        self.x_proj = Dense(d, dt_rank + 2 * n, rng, bias=False)

        # Map the low-rank step-size representation to d channels.
        self.dt_proj = Dense(dt_rank, d, rng)

        # Replace the randomly initialized bias with the SSM step-size bias.
        self.dt_proj.bias = Tensor(p["dt_bias"])

        # Store the continuous-time dynamics and skip connection.
        self.A_log = Tensor(p["A_log"])
        self.D = Tensor(p["D"])

        self.d = d
        self.n = n
        self.dt_rank = dt_rank

    def selection(self, x):
        # Project the input sequence.
        projected = self.x_proj(x)

        # Split the projection into step-size, input, and readout components.
        dt_part = projected[..., :self.dt_rank]
        Bt = projected[..., self.dt_rank:self.dt_rank + self.n]
        Ct = projected[..., self.dt_rank + self.n:]

        # Produce positive, input-dependent step sizes.
        # Shape: (batch, L, d).
        delta = self.dt_proj(dt_part).softplus()

        # Bt and Ct have shape (batch, L, n) and are shared across channels.
        return delta, Bt, Ct

    def __call__(self, x):
        # Input x has shape (batch, L, d).
        batch, seq_len, d = x.shape

        # Compute the input-dependent SSM parameters.
        delta, Bt, Ct = self.selection(x)

        # Recover the continuous-time diagonal state matrix.
        # Shape: (d, n).
        A = -self.A_log.exp()

        # Euler discretization, broadcasting across batch, time,
        # channel, and state dimensions.
        # delta[..., None]: (batch, L, d, 1)
        # Bt[:, :, None, :]: (batch, L, 1, n)
        # a_bar and b_bar: (batch, L, d, n)
        a_bar, b_bar = discretize(
            delta[..., None],
            A,
            Bt[:, :, None, :],
            method="euler",
        )

        # Construct the input-dependent recurrence:
        # h_t = a_bar_t * h_(t-1) + b_bar_t * x_t,
        # with the initial state h_(-1) = 0.
        b = b_bar * x[..., None]

        # Compute all recurrent states using the differentiable
        # Hillis-Steele associative scan from Part 2.
        h = associative_scan(a_bar, b)

        # Input-dependent readout followed by the skip connection:
        # y_t = sum_n(C_t,n * h_t,d,n) + D * x_t.
        y = (h * Ct[:, :, None, :]).sum(axis=-1) + self.D * x

        return y

# Step 14 - CausalSelfAttention
class CausalSelfAttention:
    def __init__(self, d, rng):
        self.q = Dense(d, d, rng)
        self.k = Dense(d, d, rng)
        self.v = Dense(d, d, rng)
        self.o = Dense(d, d, rng)
        self.d = d

    def __call__(self, x):
        # Input shape: (batch, L, d).
        batch, L, d = x.shape

        # Query, key, and value projections.
        q = self.q(x)
        k = self.k(x)
        v = self.v(x)

        # Scaled dot-product attention scores.
        scores = (q @ k.transpose(1, 2)) / math.sqrt(self.d)

        # Lower-triangular causal mask, explicitly converted to Boolean.
        keep = Tensor.ones((L, L)).tril().cast(
            __import__("tinygrad").dtypes.bool
        )

        # Mask future positions.
        scores = keep.where(scores, -1e9)

        # Normalize attention scores over the key-position axis.
        weights = scores.softmax(axis=-1)

        # Weighted value aggregation and output projection.
        out = self.o(weights @ v)

        return out

# Step 15 - MixerBlock
def rms_norm(x, eps=1e-5):
    # Root-mean-square normalization over the final feature dimension.
    mean_square = (x * x).mean(axis=-1, keepdim=True)
    return x / (mean_square + eps).sqrt()


class MixerBlock:
    def __init__(self, d_model, kind, rng, d_state=8, expand=2, dt_rank=4):
        # Store the mixer type and model dimension.
        self.kind = kind
        self.d_model = d_model

        if kind in ("lti", "s6"):
            # Expand the model dimension before applying the SSM.
            d_inner = expand * d_model

            # Build components in the required initialization order.
            self.in_proj = Dense(d_model, d_inner, rng)

            if kind == "lti":
                self.mixer = LTISSM(d_inner, d_state, rng)
            else:
                self.mixer = S6(d_inner, d_state, dt_rank, rng)

            self.out_proj = Dense(d_inner, d_model, rng)

        elif kind == "attn":
            # Attention has its own query, key, value, and output projections.
            self.mixer = CausalSelfAttention(d_model, rng)

        else:
            raise ValueError(
                f"Unsupported mixer kind: {kind!r}. "
                "Expected 'lti', 's6', or 'attn'."
            )

    def __call__(self, x):
        # Pre-normalize the input before applying the mixer.
        x_norm = rms_norm(x)

        if self.kind in ("lti", "s6"):
            # Project to the inner dimension, then apply SiLU:
            # SiLU(z) = z * sigmoid(z).
            z = self.in_proj(x_norm)
            z = z * z.sigmoid()

            # Apply the state space mixer and project back to d_model.
            z = self.mixer(z)
            z = self.out_proj(z)

            # Residual connection.
            return x + z

        # Attention operates directly at the model dimension.
        return x + self.mixer(x_norm)

# Step 16 - SelectionModel
class SelectionModel:
    def __init__(
        self,
        vocab,
        d_model,
        n_layers,
        kind,
        seed,
        max_len=256,
        d_state=8,
        expand=2,
        dt_rank=4,
    ):
        # Initialize the seeded NumPy random number generator.
        rng = np.random.default_rng(seed)

        self.kind = kind
        self.vocab = vocab

        # Token embedding table: (vocab, d_model), float32.
        self.embed = Tensor(
            rng.normal(0.0, 1.0, size=(vocab, d_model)).astype(np.float32)
        )

        # Learned absolute positional embeddings are used only by attention.
        if kind == "attn":
            self.pos = Tensor(
                rng.normal(
                    0.0, 0.02, size=(max_len, d_model)
                ).astype(np.float32)
            )
        else:
            self.pos = None

        # Construct the mixer blocks in order, sharing the seeded generator.
        self.blocks = [
            MixerBlock(d_model, kind, rng, d_state, expand, dt_rank)
            for _ in range(n_layers)
        ]

        # Output projection from model dimension to vocabulary logits.
        self.head = Dense(d_model, vocab, rng)

    def __call__(self, inputs):
        # Convert one-hot token inputs to model representations.
        # inputs: (batch, L, vocab)
        x = inputs @ self.embed

        # Add learned absolute positional embeddings for attention only.
        if self.pos is not None:
            L = inputs.shape[1]
            x = x + self.pos[:L]

        # Apply each sequence-mixing residual block.
        for block in self.blocks:
            x = block(x)

        # Normalize and project to vocabulary logits.
        return self.head(rms_norm(x))

# Step 17 - masked_cross_entropy
def masked_cross_entropy(logits, targets):
    # Compute log probabilities over the vocabulary dimension.
    log_probs = logits.log_softmax(axis=-1)

    # Only answer positions contribute because targets are all zeros
    # at positions that should be ignored.
    negative_log_likelihood = -(targets * log_probs).sum()

    # Normalize by the total number of answer positions.
    n_answers = targets.sum()

    return negative_log_likelihood / n_answers

# Step 18 - train_model
def train_model(model, make_batch, vocab, steps=300, lr=3e-3, batch_size=64, seed=0):
    from contextlib import nullcontext

    # Initialize Adam using all trainable parameters in the model.
    opt = nn.optim.Adam(get_parameters(model), lr=lr)

    # Use the training API supported by the installed tinygrad version.
    if hasattr(Tensor, "train"):
        # Legacy API, as specified in the Deep-ML instructions.
        training_context = nullcontext()

        @TinyJit
        def step(inputs, targets):
            with Tensor.train():
                opt.zero_grad()
                loss = masked_cross_entropy(model(inputs), targets)
                loss.backward()
                opt.step()
                return loss.realize()

    else:
        # Newer API: use Context to enable training mode.
        from tinygrad import Context
        training_context = Context(TRAINING=1)

        @TinyJit
        def step(inputs, targets):
            opt.zero_grad()
            loss = masked_cross_entropy(model(inputs), targets)
            loss.backward()
            opt.step()
            return loss.realize()

    losses = []

    # In newer tinygrad versions, keep training mode active around
    # the TinyJit calls, including the initial tracing calls.
    with training_context:
        for s in range(steps):
            # Generate a reproducible batch.
            x, y, mask = make_batch(
                batch_size,
                seed * 100_000 + s,
            )

            # Encode inputs and masked targets.
            inputs, targets = encode_batch(x, y, mask, vocab)

            inputs_tensor = Tensor(inputs).realize()
            targets_tensor = Tensor(targets).realize()

            # Run the training step and store a Python float.
            loss = step(inputs_tensor, targets_tensor)
            losses.append(float(loss.item()))

    return losses

# Step 19 - evaluate
def evaluate(model, x, y, mask, vocab, chunk=64):
    # Track the number of correct predictions and total answer positions.
    correct = 0
    total = 0

    # Process the dataset in consecutive chunks to bound memory usage.
    for start in range(0, len(x), chunk):
        end = min(start + chunk, len(x))

        # Encode this chunk into float32 one-hot input and target arrays.
        inputs, _ = encode_batch(
            x[start:end],
            y[start:end],
            mask[start:end],
            vocab,
        )

        # Run the model without JIT compilation or explicit gradient computation.
        logits = model(Tensor(inputs))

        # Convert logits to NumPy and select the most likely token at each position.
        predictions = np.argmax(logits.numpy(), axis=-1)

        # Only positions marked True contribute to accuracy.
        chunk_y = y[start:end]
        chunk_mask = mask[start:end].astype(bool)

        correct += int(np.sum(
            (predictions == chunk_y) & chunk_mask
        ))
        total += int(np.sum(chunk_mask))

    # Return answer-position accuracy as a Python float.
    return float(correct / total) if total > 0 else 0.0

# Step 20 - selective_copying_study
STUDY_VOCAB = 18
STUDY_MAX_LEN = 160


def selective_copying_study(
    kinds=("lti", "s6", "attn"),
    seeds=(0,),
    steps=300,
    batch_size=64,
    lr=3e-3,
    n_ctx=16,
    n_data=4,
    n_vocab=16,
    d_model=32,
    n_eval=256,
):
    # Store evaluation accuracies and trained models for each mixer.
    acc = {kind: [] for kind in kinds}
    models = {kind: [] for kind in kinds}

    # Preserve the required iteration order: mixer kind, then seed.
    for kind in kinds:
        for seed in seeds:
            # Use the shared vocabulary and maximum sequence length
            # for every model in this experiment.
            model = SelectionModel(
                vocab=STUDY_VOCAB,
                d_model=d_model,
                n_layers=2,
                kind=kind,
                seed=seed,
                max_len=STUDY_MAX_LEN,
            )

            # Generate Selective Copying batches using the supplied seed.
            def make(n, s):
                return selective_copying_batch(
                    n,
                    s,
                    n_ctx=n_ctx,
                    n_data=n_data,
                    n_vocab=n_vocab,
                )

            # Train every model with the same training budget.
            train_model(
                model,
                make,
                STUDY_VOCAB,
                steps=steps,
                lr=lr,
                batch_size=batch_size,
                seed=seed,
            )

            # Generate a separate, reproducible held-out evaluation set.
            x_eval, y_eval, mask_eval = make(
                n_eval,
                10**7 + seed,
            )

            # Measure accuracy only at answer positions.
            score = evaluate(
                model,
                x_eval,
                y_eval,
                mask_eval,
                STUDY_VOCAB,
            )

            acc[kind].append(round(float(score), 4))
            models[kind].append(model)

    return acc, models

# Step 21 - induction_extrapolation
STUDY_VOCAB = 18
STUDY_MAX_LEN = 160


def induction_extrapolation(
    kinds=("s6", "attn"),
    seeds=(0,),
    steps=300,
    batch_size=64,
    lr=3e-3,
    train_len=20,
    mults=(1, 2, 4, 8),
    n_vocab=16,
    d_model=32,
    n_eval=128,
):
    # Store accuracy lists indexed by mixer kind and length multiplier.
    acc = {
        kind: {int(m): [] for m in mults}
        for kind in kinds
    }
    models = {kind: [] for kind in kinds}

    # Allocate enough positions for the longest evaluation sequence.
    max_eval_len = train_len * max(mults, default=1)
    max_len = max(STUDY_MAX_LEN, max_eval_len)

    # Train each mixer at the same sequence length and training budget.
    for kind in kinds:
        for seed in seeds:
            model = SelectionModel(
                STUDY_VOCAB,
                d_model,
                2,
                kind,
                seed,
                max_len=max_len,
            )

            # The model is trained only on sequences of length train_len.
            def make(n, s):
                return induction_heads_batch(
                    n,
                    s,
                    seq_len=train_len,
                    n_vocab=n_vocab,
                )

            train_model(
                model,
                make,
                STUDY_VOCAB,
                steps=steps,
                lr=lr,
                batch_size=batch_size,
                seed=seed,
            )

            # Preserve each trained model for subsequent analysis.
            models[kind].append(model)

            # Test generalization at each requested length multiplier.
            for m in mults:
                eval_len = train_len * m

                x_eval, y_eval, mask_eval = induction_heads_batch(
                    n_eval,
                    10**7 + seed,
                    seq_len=eval_len,
                    n_vocab=n_vocab,
                )

                score = evaluate(
                    model,
                    x_eval,
                    y_eval,
                    mask_eval,
                    STUDY_VOCAB,
                )

                # Store accuracy rounded to four decimal places.
                acc[kind][int(m)].append(round(float(score), 4))

    return acc, models

# Step 22 - delta_profile
def delta_profile(model, x, trigger):
    # This analysis is defined only for the selective S6 model.
    if model.kind != "s6":
        raise ValueError("delta_profile requires a model with kind='s6'")

    # Convert the input sequences to a NumPy array.
    x = np.asarray(x)

    # Identify every trigger occurrence in each sequence.
    trigger_mask = x == trigger

    # Mark the position immediately after the first trigger in each row.
    value_mask = np.zeros(x.shape, dtype=bool)

    for i in range(x.shape[0]):
        positions = np.flatnonzero(trigger_mask[i])

        if positions.size > 0:
            value_pos = positions[0] + 1
            if value_pos < x.shape[1]:
                value_mask[i, value_pos] = True

    # Every remaining position belongs to the "other" category.
    other_mask = ~(trigger_mask | value_mask)

    # Encode the input tokens as one-hot vectors.
    inputs, _ = encode_batch(
        x,
        x,
        np.zeros(x.shape, dtype=bool),
        model.vocab,
    )

    # Initialize the model's hidden representation from token embeddings.
    h = Tensor(inputs) @ model.embed

    profiles = []

    # Inspect the input-dependent step sizes in each S6 layer.
    for block in model.blocks:
        # Reproduce the SSM input path from MixerBlock:
        # RMSNorm -> input projection -> SiLU.
        u = block.in_proj(rms_norm(h))
        u = u * u.sigmoid()

        # Selection returns delta with shape (batch, L, d_inner).
        # Average across channels to obtain one value per position.
        delta = block.mixer.selection(u)[0]
        delta = delta.mean(axis=-1).numpy()

        # Calculate the mean step size for each position category.
        layer_profile = {}

        for name, mask in (
            ("trigger", trigger_mask),
            ("value", value_mask),
            ("other", other_mask),
        ):
            selected = delta[mask]

            # Return 0.0 when a category has no positions.
            mean_delta = float(selected.mean()) if selected.size else 0.0
            layer_profile[name] = round(mean_delta, 4)

        profiles.append(layer_profile)

        # Advance the hidden representation for analysis of the next layer.
        h = block(h)

    return profiles

# Step 23 - s6_decode_step
def s6_decode_step(ssm, x_t, h):
    # x_t has shape (batch, d).
    # h has shape (batch, d, n).

    batch, d = x_t.shape

    # Compute input-dependent selection parameters for this single position.
    # selection expects (batch, L, d), so use L = 1.
    delta, Bt, Ct = ssm.selection(x_t.reshape(batch, 1, d))

    # Remove the length-one sequence dimension.
    # delta: (batch, d)
    # Bt, Ct: (batch, n)
    delta = delta[:, 0, :]
    Bt = Bt[:, 0, :]
    Ct = Ct[:, 0, :]

    # Continuous-time diagonal state matrix: A = -exp(A_log).
    A = -ssm.A_log.exp()

    # Zero-order hold for A:
    # a_bar = exp(delta * A).
    a_bar = (delta[..., None] * A).exp()

    # Euler discretization for B, combined with the current input:
    # b_bar * x_t = delta * Bt * x_t.
    b = delta[..., None] * Bt[:, None, :] * x_t[..., None]

    # Update the recurrent state:
    # h_new = a_bar * h + b.
    h_new = a_bar * h + b

    # Read out the new state using input-dependent C and add the skip:
    # y_t = sum_n(C_t * h_new) + D * x_t.
    y_t = (Ct[:, None, :] * h_new).sum(axis=-1) + ssm.D * x_t

    return y_t, h_new

# Step 24 - model_decode
def model_decode(model, inputs):
    # Recurrent decoding is implemented only for selective S6 models.
    if model.kind != "s6":
        raise ValueError("model_decode requires a model with kind='s6'")

    # Ensure inputs are a float32 NumPy array of shape (batch, L, vocab).
    inputs = np.asarray(inputs, dtype=np.float32)
    batch, seq_len, _ = inputs.shape

    # Initialize one fixed-size recurrent state for every S6 block.
    # Each state has shape (batch, d_inner, d_state).
    states = []
    for block in model.blocks:
        d_inner, d_state = block.mixer.A_log.shape
        state = Tensor.zeros((batch, d_inner, d_state)).realize()
        states.append(state)

    # The per-token vocabulary logits will be collected here.
    logits_list = []

    # Process one token position at a time.
    for t in range(seq_len):
        # Embed the current one-hot token.
        # Shape: (batch, d_model).
        x_t = Tensor(inputs[:, t, :]) @ model.embed

        # Advance through each S6 residual block in recurrent mode.
        for i, block in enumerate(model.blocks):
            # Match the parallel block's pre-norm, projection, and SiLU.
            u = block.in_proj(rms_norm(x_t))
            u = u * u.sigmoid()

            # Advance this layer's state by one token.
            y_t, new_state = s6_decode_step(
                block.mixer,
                u,
                states[i],
            )

            # Realize and retain the updated state for the next position.
            states[i] = new_state.realize()

            # Apply the output projection and residual connection.
            x_t = x_t + block.out_proj(y_t)

        # Compute vocabulary logits for this token position.
        token_logits = model.head(rms_norm(x_t))
        logits_list.append(token_logits.numpy())

    # Stack per-token logits into shape (batch, L, vocab).
    if logits_list:
        logits = np.stack(logits_list, axis=1).astype(np.float32, copy=False)
    else:
        logits = np.empty(
            (batch, 0, model.vocab),
            dtype=np.float32,
        )

    # Count the floats retained by all recurrent states.
    # This depends on batch size and model dimensions, not sequence length.
    state_floats = sum(
        int(np.prod(state.shape))
        for state in states
    )

    return logits, state_floats

# Step 25 - cache_bytes
def cache_bytes(
    kind,
    seq_len,
    d_model=32,
    n_layers=2,
    d_state=8,
    expand=2,
    bytes_per=4,
):
    # SSM models retain a fixed-size recurrent state per sequence.
    if kind in ("lti", "s6"):
        return int(
            n_layers * expand * d_model * d_state * bytes_per
        )

    # Attention retains keys and values for every past token,
    # for every layer.
    elif kind == "attn":
        return int(
            n_layers * 2 * seq_len * d_model * bytes_per
        )

    # Reject unsupported model kinds.
    else:
        raise ValueError(
            f"Unsupported model kind: {kind!r}. "
            "Expected 'lti', 's6', or 'attn'."
        )

# Step 26 - selection_report
STUDY_VOCAB = 18
STUDY_MAX_LEN = 160


def _s6_decode_step_rank_preserving(ssm, x_t, h):
    """
    Advance one S6 token while retaining a length-one sequence axis.

    x_t: (batch, 1, d)
    h:   (batch, d, n)
    """
    batch, seq_len, d = x_t.shape

    if seq_len != 1:
        raise ValueError("Expected exactly one token per decode step")

    # Use the same selection and discretization shapes as S6.__call__.
    delta, Bt, Ct = ssm.selection(x_t)
    A = -ssm.A_log.exp()

    a_bar, b_bar = discretize(
        delta[..., None],
        A,
        Bt[:, :, None, :],
        method="euler",
    )

    # Retain the singleton sequence axis during the state update.
    h_new = (
        a_bar * h[:, None, :, :]
        + b_bar * x_t[..., None]
    )

    # Input-dependent readout plus the skip connection.
    y_t = (
        h_new * Ct[:, :, None, :]
    ).sum(axis=-1) + ssm.D * x_t

    # Return the output with its sequence axis and the updated state.
    return y_t, h_new[:, 0, :, :]


def _model_decode_rank_preserving(model, inputs):
    """Run an S6 SelectionModel recurrently, one token at a time."""
    if model.kind != "s6":
        raise ValueError("Recurrent decoding requires kind='s6'")

    inputs = np.asarray(inputs, dtype=np.float32)
    batch, seq_len, vocab = inputs.shape

    # Allocate one fixed-size recurrent state per S6 layer.
    states = []
    for block in model.blocks:
        d_inner, d_state = block.mixer.A_log.shape
        states.append(
            Tensor.zeros((batch, d_inner, d_state)).realize()
        )

    logits_by_position = []

    for t in range(seq_len):
        # Embed one token, retaining shape (batch, 1, d_model).
        x_t = Tensor(inputs[:, t:t + 1, :]) @ model.embed

        for i, block in enumerate(model.blocks):
            # Match the block's parallel pre-norm input path.
            u = block.in_proj(rms_norm(x_t))
            u = u * u.sigmoid()

            # Advance the S6 state by one position.
            y_t, new_state = _s6_decode_step_rank_preserving(
                block.mixer,
                u,
                states[i],
            )

            # Realize the updated state before the next token.
            states[i] = new_state.realize()

            # Apply the outer residual connection.
            x_t = x_t + block.out_proj(y_t)

        # Compute this position's vocabulary logits.
        token_logits = model.head(rms_norm(x_t))
        logits_by_position.append(token_logits.numpy()[:, 0, :])

    if seq_len:
        logits = np.stack(logits_by_position, axis=1).astype(
            np.float32, copy=False
        )
    else:
        logits = np.empty((batch, 0, model.vocab), dtype=np.float32)

    state_floats = sum(
        int(np.prod(state.shape))
        for state in states
    )

    return logits, int(state_floats)


def selection_report(
    seed=0,
    steps=300,
    sc_kinds=("lti", "s6", "attn"),
    mults=(1, 2, 4, 8),
    n_eval=128,
):
    # Use three consecutive seeds for the experiment.
    seeds = (seed, seed + 1, seed + 2)

    # ---------------------------------------------------------
    # 1. Selective Copying experiment.
    # ---------------------------------------------------------
    sc_acc, _ = selective_copying_study(
        kinds=sc_kinds,
        seeds=seeds,
        steps=steps,
        batch_size=64,
        lr=3e-3,
        n_ctx=16,
        n_data=4,
        n_vocab=16,
        d_model=32,
        n_eval=n_eval,
    )

    selective_copying = {
        kind: float(np.mean(scores))
        for kind, scores in sc_acc.items()
    }

    # ---------------------------------------------------------
    # 2. Induction Heads length extrapolation.
    # ---------------------------------------------------------
    ind_acc, ind_models = induction_extrapolation(
        kinds=("s6", "attn"),
        seeds=seeds,
        steps=steps,
        batch_size=64,
        lr=3e-3,
        train_len=20,
        mults=mults,
        n_vocab=16,
        d_model=32,
        n_eval=n_eval,
    )

    induction = {
        kind: {
            int(mult): float(np.mean(scores))
            for mult, scores in values.items()
        }
        for kind, values in ind_acc.items()
    }

    # ---------------------------------------------------------
    # 3. Measure where the trained S6 layer selects.
    # ---------------------------------------------------------
    s6_model = ind_models["s6"][0]

    probe_x, _, _ = induction_heads_batch(
        max(1, n_eval),
        10**7 + seed,
        seq_len=20,
        n_vocab=16,
    )

    profile = delta_profile(
        s6_model,
        probe_x,
        trigger=16,
    )

    delta_profile_result = [
        {
            name: float(value)
            for name, value in layer.items()
        }
        for layer in profile
    ]

    # ---------------------------------------------------------
    # 4. Compare parallel and recurrent S6 inference.
    # ---------------------------------------------------------
    decode_batch = min(8, max(1, n_eval))

    decode_x, decode_y, decode_mask = induction_heads_batch(
        decode_batch,
        10**7 + seed + 1,
        seq_len=20,
        n_vocab=16,
    )

    decode_inputs, _ = encode_batch(
        decode_x,
        decode_y,
        decode_mask,
        STUDY_VOCAB,
    )

    # Parallel evaluation of the full sequence.
    parallel_logits = s6_model(Tensor(decode_inputs)).numpy()

    # Recurrent evaluation, one token at a time.
    recurrent_logits, state_floats = _model_decode_rank_preserving(
        s6_model,
        decode_inputs,
    )

    max_abs_diff = float(
        np.max(np.abs(parallel_logits - recurrent_logits))
    )

    decode = {
        "state_floats": int(state_floats),
        "max_abs_diff": max_abs_diff,
    }

    # ---------------------------------------------------------
    # 5. Compare per-sequence S6 state memory with attention KV.
    # ---------------------------------------------------------
    cache_lengths = {
        20,
        160,
        1280,
        10240,
        *(20 * int(mult) for mult in mults),
    }

    cache_table = {
        length: {
            "s6": int(cache_bytes("s6", length)),
            "attn": int(cache_bytes("attn", length)),
        }
        for length in sorted(cache_lengths)
    }

    return {
        "cache_bytes": cache_table,
        "decode": decode,
        "delta_profile": delta_profile_result,
        "induction": induction,
        "selective_copying": selective_copying,
    }

