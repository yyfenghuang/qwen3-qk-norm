"""From-scratch RMSNorm, proven equivalent to Qwen3's own implementation.

This module derives RMSNorm explicitly rather than importing it, because the
point is to understand the operation q_norm/k_norm perform. 
But a from-scratch reimplementation is only trustworthy if it is
checked against the real thing, so `assert_matches_qwen3` proves the manual
version is numerically identical to the upstream `Qwen3RMSNorm` on random
inputs and weights. No model weights are downloaded: a `Qwen3RMSNorm` module
is instantiated directly and given random parameters.

Two subtleties this makes explicit, both of which are easy to get wrong and
both of which change the output if missed:

  1. The variance is computed in float32 and the normalization is applied in
     float32, then cast back to the input dtype. A manual version that stays
     in the input dtype (e.g. float16) diverges. This is the reason the
     equivalence check uses allclose, not equal, when dtypes differ.

  2. q_norm/k_norm normalize over head_dim (128), not hidden_size (1024). The
     normalized axis is the last one, and for QK-Norm that axis is a single
     head's vector. This is what makes the normalization per-head, and it is
     the mechanism behind the depth-wise norm compression measured elsewhere
     in this repo.
"""

import inspect

import torch

from transformers.models.qwen3.modeling_qwen3 import Qwen3RMSNorm

__all__ = [
    "rms_norm_from_scratch",
    "assert_rms_matches_qwen3",
    "assert_qwen3_rmsnorm_unchanged",
    "demo_per_head_axis",
]

HEAD_DIM = 128  # Qwen3-0.6B q_norm/k_norm normalize over this


def rms_norm_from_scratch(x, weight, eps=1e-6):
    """RMSNorm over the last axis, written out explicitly.

    y = x / sqrt(mean(x^2, last axis) + eps) * weight

    The cast to float32 before computing the statistic, and back to the input
    dtype after scaling, mirrors what Qwen3RMSNorm does. Removing it would make
    this diverge from upstream on half precision.
    """
    in_dtype = x.dtype
    x = x.to(torch.float32)
    variance = x.pow(2).mean(dim=-1, keepdim=True)
    x = x * torch.rsqrt(variance + eps)
    # Weight is applied after casting back, matching Qwen3RMSNorm exactly.
    return (weight * x.to(in_dtype))


def assert_qwen3_rmsnorm_unchanged():
    """Fail loudly if upstream Qwen3RMSNorm no longer matches this derivation.

    Keys on the structural operations the from-scratch version replicates
    rather than an exact-text hash: an exact hash breaks on whitespace or
    comment changes, while these checks break only if the actual computation
    changes.
    """
    src = inspect.getsource(Qwen3RMSNorm.forward)
    flat = " ".join(src.split())

    assert "pow(2).mean(-1" in flat or "pow(2).mean(dim=-1" in flat, (
        "Qwen3RMSNorm no longer computes mean of squares over the last axis. "
        "The from-scratch derivation here may be stale. Re-read the source."
    )
    assert "rsqrt(" in flat, (
        "Qwen3RMSNorm no longer uses rsqrt. Re-read the source."
    )
    assert "float32" in flat, (
        "Qwen3RMSNorm no longer casts to float32 for the statistic. The "
        "dtype-handling note in this module may be stale."
    )
    return True


def assert_rms_matches_qwen3(
    n_tokens=32, dim=HEAD_DIM, eps=1e-6, dtype=torch.float32, seed=0
):
    """Prove the manual RMSNorm equals Qwen3RMSNorm on random weights.

    A Qwen3RMSNorm is built directly (no model download) and given random
    weights, then both implementations run on the same random input. Equality
    is checked with allclose: the two paths do identical float32 arithmetic,
    but allclose is used rather than equal to stay robust to any harmless
    reordering upstream and to non-float32 dtypes.
    """
    assert_qwen3_rmsnorm_unchanged()

    torch.manual_seed(seed)
    x = torch.randn(1, n_tokens, dim, dtype=dtype)

    ref = Qwen3RMSNorm(dim, eps=eps).to(dtype)
    with torch.no_grad():
        ref.weight.copy_(torch.randn(dim, dtype=dtype))

    with torch.no_grad():
        mine = rms_norm_from_scratch(x, ref.weight, eps=eps)
        theirs = ref(x)

    max_abs_diff = (mine - theirs).abs().max().item()
    assert torch.allclose(mine, theirs, atol=1e-6, rtol=1e-5), (
        f"from-scratch RMSNorm diverges from Qwen3RMSNorm "
        f"(max abs diff {max_abs_diff:.2e})"
    )
    return {"max_abs_diff": max_abs_diff, "dim": dim, "n_tokens": n_tokens}


def demo_per_head_axis():
    """Show the normalized axis is head_dim, and normalization is independent
    per head.

    Build a (batch, heads, tokens, head_dim) tensor, apply RMSNorm over the
    last axis, and confirm each head's output depends only on that head's
    input: perturbing one head's values leaves the other heads' normalized
    output untouched. This is the concrete meaning of "per-head" normalization.

    Note on the perturbation: RMSNorm is scale-invariant, so multiplying a
    head's vector by a constant is a no-op (the scalar normalizes away). That
    is itself a real property worth knowing, but it makes multiplication the
    wrong probe for per-head independence. An additive perturbation changes the
    direction, not just the scale, so it actually alters the normalized output.
    """
    torch.manual_seed(0)
    n_heads, n_tokens = 4, 8
    x = torch.randn(1, n_heads, n_tokens, HEAD_DIM)
    weight = torch.ones(HEAD_DIM)

    out = rms_norm_from_scratch(x, weight)

    # Additive perturbation on head 0 only. Multiplying would be a no-op:
    # RMSNorm normalizes away any constant scale, so a scaled vector maps to
    # the same output. Adding noise changes direction and does change it.
    x2 = x.clone()
    x2[:, 0] += torch.randn_like(x2[:, 0])
    out2 = rms_norm_from_scratch(x2, weight)

    head0_changed = not torch.allclose(out[:, 0], out2[:, 0])
    others_unchanged = torch.allclose(out[:, 1:], out2[:, 1:])

    assert head0_changed, "perturbing head 0's input should change its output"
    assert others_unchanged, (
        "perturbing head 0 changed another head's output; normalization is "
        "not actually per-head"
    )

    # Separately, demonstrate the scale-invariance itself, since it is the
    # reason the multiplication probe would have failed.
    x_scaled = x.clone()
    x_scaled[:, 0] *= 10.0
    out_scaled = rms_norm_from_scratch(x_scaled, weight)
    scale_invariant = torch.allclose(out[:, 0], out_scaled[:, 0], atol=1e-6)

    print(f"input shape           : {tuple(x.shape)}  (batch, heads, tokens, head_dim)")
    print(f"normalized axis       : last ({HEAD_DIM}) = one head's vector")
    print(f"perturbing head 0 changes head 0 output   : {head0_changed}")
    print(f"perturbing head 0 leaves others untouched : {others_unchanged}")
    print(f"scaling head 0 by 10x is a no-op (RMSNorm scale-invariant): {scale_invariant}")
    print("Normalization is independent per head, and invariant to per-head scale.")

    return {
        "head0_changed": head0_changed,
        "others_unchanged": others_unchanged,
        "scale_invariant": scale_invariant,
    }


def main():
    print("From-scratch RMSNorm source:")
    print()
    print(inspect.getsource(rms_norm_from_scratch))
    print("Upstream Qwen3RMSNorm.forward (imported live):")
    print()
    print(inspect.getsource(Qwen3RMSNorm.forward))

    print("Drift guard:", "PASS" if assert_qwen3_rmsnorm_unchanged() else "FAIL")
    print()

    result = assert_rms_matches_qwen3()
    print(
        f"Equivalence: max abs diff {result['max_abs_diff']:.2e} over "
        f"{result['n_tokens']} tokens x {result['dim']} dims  (PASS)"
    )
    print()
    demo_per_head_axis()


if __name__ == "__main__":
    main()