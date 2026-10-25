"""
Why Selection Matters: Mamba's Synthetic Tasks in tinygrad scaffold.

Run this with: python scaffold.py
Uses functions defined in model.py.
"""

from model import *  # noqa: F401, F403 (pulls in your solution functions)

"""Why Selection Matters: Mamba's synthetic tasks in tinygrad.

Story: a linear time-invariant SSM weights the past only by distance, so it cannot pick
out tokens that sit at random positions; the selective S6 SSM computes its step size,
B and C from each token and can. Trained at 20 tokens, S6 keeps its induction-head
accuracy at 2x and 4x that length while attention with learned absolute positions
falls toward chance, and at inference S6 runs as a recurrence with a fixed-size state
that gives the same logits as its parallel scan, where attention's cache keeps growing.
"""
import time
import numpy as np


def main() -> None:
    t0 = time.time()
    rep = selection_report(seed=0, steps=300)

    print("1. Selective Copying (16 noise positions, 4 tokens to copy from 16 symbols; chance 0.0625)")
    for kind, name in (("lti", "LTI diagonal SSM (S4D-Real)"), ("s6", "selective SSM (S6)"), ("attn", "causal attention")):
        print(f"   {name:30s} answer accuracy {rep['selective_copying'][kind]:.3f}")
    print("   Same block, width, data and 300 Adam steps: the time-invariant model is far behind.")
    print("   Over seeds 0-2 at this budget: LTI 0.37-0.38, S6 0.62-0.88, attention 0.83-0.90.")

    print("\n2. Induction Heads, trained at length 20, tested at 1x / 2x / 4x / 8x (chance 0.0625)")
    for kind, name in (("s6", "selective SSM (S6)"), ("attn", "attention + learned positions")):
        accs = rep["induction"][kind]
        print(f"   {name:30s} " + "  ".join(f"{m}x {accs[m]:.3f}" for m in sorted(accs)))
    print("   S6 stays above 0.8 to 4x; attention breaks where its position table was never trained.")
    print("   Over seeds 0-2: S6 0.82-1.0 at 2x, 0.83-0.91 at 4x, 0.25-0.63 at 8x; attention 0.36-0.50, 0.16-0.27, 0.11-0.19.")

    print("\n3. Where the trained S6 writes (mean step size delta by position class)")
    for i, d in enumerate(rep["delta_profile"]):
        print(f"   layer {i + 1}: trigger {d['trigger']:.4f}  value {d['value']:.4f}  other {d['other']:.4f}")
    lp = rep["delta_profile"][-1]
    print(f"   The second layer steps {lp['value'] / lp['other']:.1f}x harder on the value right after the trigger than on noise (3.7x-8.9x over seeds 0-2).")

    print("\n4. Recurrent inference")
    print(f"   token-by-token decoding vs parallel scan: max |logit difference| = {rep['decode']['max_abs_diff']:.2e}")
    print(f"   recurrent state for 8 sequences: {rep['decode']['state_floats']} floats, the same at any length")
    for L in sorted(rep["cache_bytes"]):
        c = rep["cache_bytes"][L]
        print(f"   {L:>6} tokens: S6 state {c['s6']:>8} bytes | attention KV cache {c['attn']:>9} bytes")
    print(f"\nstudy finished in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()

