# Why Selection Matters: Mamba's Synthetic Tasks in tinygrad

The Mamba paper's central argument is not about speed. It is that a state space model whose dynamics are fixed in time cannot do content-based reasoning, and that making the step size and the B and C matrices functions of the input fixes this without giving up a recurrent, constant-memory inference mode. This project tests that argument yourself, in tinygrad, on the paper's two synthetic tasks. You write seeded generators for Selective Copying (tokens to copy hidden at random positions among noise) and Induction Heads (recall the token that followed a trigger), then a log-depth Hillis-Steele associative scan over the operator (a1, b1) o (a2, b2) = (a1 a2, a2 b1 + b2), checked against a sequential loop and differentiated through. On top of it you build three sequence mixers: an LTI diagonal SSM with S4D-Real initialisation (and the fixed convolution kernel it is equivalent to), the selective S6 layer with softplus step sizes, zero-order hold for A and the Euler rule for B, and causal attention with a learned absolute position table. A shared pre-norm block, a masked cross-entropy over answer positions and a TinyJit-compiled Adam loop train two-layer models in seconds on a CPU. Then you run the experiments: under one training budget the LTI model stays at about 0.37 on Selective Copying while S6 reaches 0.62 to 0.88 and attention 0.83 to 0.90 (chance 0.06); trained on Induction Heads at length 20, S6 stays above 0.8 at 2x and 4x the length while attention falls to 0.36-0.50 at 2x and toward chance beyond; and inside the trained S6 the second layer takes a step 3.7 to 8.9 times larger on the value right after the trigger than on noise. You finish by decoding the S6 model one token at a time, matching its parallel forward pass to float precision with a state that never grows, and by setting that state against attention's KV cache. Every claim was checked across three seeds, and every step prints the numbers it measures.

## How to run

```bash
python scaffold.py
```

## Steps

- [x] **1.** selective_copying_batch
- [x] **2.** induction_heads_batch
- [x] **3.** encode_batch
- [x] **4.** masked_accuracy
- [x] **5.** scan_reference
- [x] **6.** scan_combine
- [x] **7.** associative_scan
- [x] **8.** Dense
- [x] **9.** ssm_init
- [x] **10.** discretize
- [x] **11.** LTISSM
- [x] **12.** lti_kernel
- [x] **13.** S6
- [x] **14.** CausalSelfAttention
- [x] **15.** MixerBlock
- [x] **16.** SelectionModel
- [x] **17.** masked_cross_entropy
- [x] **18.** train_model
- [x] **19.** evaluate
- [x] **20.** selective_copying_study
- [x] **21.** induction_extrapolation
- [x] **22.** delta_profile
- [x] **23.** s6_decode_step
- [x] **24.** model_decode
- [x] **25.** cache_bytes
- [x] **26.** selection_report

## Results

```
1. Selective Copying (16 noise positions, 4 tokens to copy from 16 symbols; chance 0.0625)
   LTI diagonal SSM (S4D-Real)    answer accuracy 0.372
   selective SSM (S6)             answer accuracy 0.751
   causal attention               answer accuracy 0.870
   Same block, width, data and 300 Adam steps: the time-invariant model is far behind.
   Over seeds 0-2 at this budget: LTI 0.37-0.38, S6 0.62-0.88, attention 0.83-0.90.

2. Induction Heads, trained at length 20, tested at 1x / 2x / 4x / 8x (chance 0.0625)
   selective SSM (S6)             1x 0.953  2x 0.927  4x 0.812  8x 0.396
   attention + learned positions  1x 0.857  2x 0.419  4x 0.229  8x 0.161
   S6 stays above 0.8 to 4x; attention breaks where its position table was never trained.
   Over seeds 0-2: S6 0.82-1.0 at 2x, 0.83-0.91 at 4x, 0.25-0.63 at 8x; attention 0.36-0.50, 0.16-0.27, 0.11-0.19.

3. Where the trained S6 writes (mean step size delta by position class)
   layer 1: trigger 0.0814  value 0.0247  other 0.0248
   layer 2: trigger 0.0360  value 0.0989  other 0.0268
   The second layer steps 3.7x harder on the value right after the trigger than on noise (3.7x-8.9x over seeds 0-2).

4. Recurrent inference
   token-by-token decoding vs parallel scan: max |logit difference| = 1.03e-05
   recurrent state for 8 sequences: 8192 floats, the same at any length
       20 tokens: S6 state     4096 bytes | attention KV cache     10240 bytes
       40 tokens: S6 state     4096 bytes | attention KV cache     20480 bytes
       80 tokens: S6 state     4096 bytes | attention KV cache     40960 bytes
      160 tokens: S6 state     4096 bytes | attention KV cache     81920 bytes
     1280 tokens: S6 state     4096 bytes | attention KV cache    655360 bytes
    10240 tokens: S6 state     4096 bytes | attention KV cache   5242880 bytes

study finished in 207 s
```
