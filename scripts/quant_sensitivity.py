"""Quantization sensitivity of q_norm / k_norm weights in Qwen3-0.6B.

Two measurements, chained:

  A. Weight rounding error. q_norm.weight and k_norm.weight are single
     head_dim-length vectors (128), shared across all heads. Quantizing them to
     INT8 and INT4 (symmetric per-tensor) introduces a rounding error per
     element. This is a pure property of the weight, needs no activations.

  B. Per-head output error. The same quantized weight, applied through RMSNorm
     to real per-head query activations, produces a different output error for
     each head, because each head's pre-norm q vector is different. This is the
     measurement that answers "which head degrades first at low precision",
     and it needs real activations, captured here via a short forward pass.

Symmetric per-tensor was chosen deliberately over per-channel: it is what edge
runtimes actually use for a vector this small, and per-channel would hide the
head sensitivity by fitting each element independently. The point of B is to
surface that sensitivity, not to minimize it.

The shape guard from capture_qk_stats is reused so the two scripts cannot
disagree about what q_norm/k_norm are.
"""

import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from transformers import AutoModelForCausalLM, AutoTokenizer

import capture_qk_stats as cap  # for assert_qknorm_module_shape and MMLU loader

MODEL_ID = "Qwen/Qwen3-0.6B"
OUTPUT_PATH = REPO_ROOT / "results" / "quant_sensitivity.json"

# A handful of short prompts is enough to get representative per-head q
# activations for measurement B; this is not a statistical sweep.
N_PROMPTS = 8
BITS = [8, 4]


# ---------------------------------------------------------------------------
# Quantization
# ---------------------------------------------------------------------------


def quantize_symmetric_per_tensor(w, bits):
    """Symmetric per-tensor quantize-dequantize of a weight vector.

    Returns the dequantized tensor (same shape and dtype as input), i.e. the
    weight as it would be seen after INT{bits} round-trip. Symmetric: zero maps
    to zero, range is [-qmax, qmax]. Per-tensor: one scale for the whole vector.
    """
    qmax = 2 ** (bits - 1) - 1              # 127 for int8, 7 for int4
    amax = w.abs().max()
    if amax == 0:
        return w.clone()
    scale = amax / qmax
    q = torch.clamp(torch.round(w / scale), -qmax, qmax)
    return q * scale


def assert_quant_roundtrip_correct():
    """Verify quantize_symmetric_per_tensor obeys the guarantees it relies on.

    Two properties that hold for any correct symmetric per-tensor round-trip,
    checked without a model:
      - Bounded error: every element's rounding error is at most half a
        quantization step. This is the defining property of round-to-nearest.
      - Coarser bits, larger error: INT4 error strictly exceeds INT8 error on a
        non-trivial vector. If this inverted, the bit-depth wiring is wrong.
    """
    torch.manual_seed(0)
    w = torch.randn(256)

    prev_mean_err = -1.0
    for bits in (8, 4):
        deq = quantize_symmetric_per_tensor(w, bits)
        err = (deq - w).abs()
        qmax = 2 ** (bits - 1) - 1
        step = w.abs().max() / qmax
        assert err.max().item() <= step / 2 + 1e-6, (
            f"INT{bits} rounding error {err.max().item()} exceeds half-step "
            f"{step / 2}"
        )
        # error grows as bits shrink; 8 is checked before 4
        assert err.mean().item() > prev_mean_err, (
            "INT4 error did not exceed INT8 error; bit-depth wiring may be wrong"
        )
        prev_mean_err = err.mean().item()

    # Zero vector must round-trip to zero (no division by zero).
    z = quantize_symmetric_per_tensor(torch.zeros(16), 4)
    assert torch.equal(z, torch.zeros(16)), "zero vector did not round-trip to zero"
    return True


# ---------------------------------------------------------------------------
# Measurement A: weight rounding error
# ---------------------------------------------------------------------------


def measure_weight_error(weight, bits):
    """Per-element and summary rounding error for one weight vector at `bits`."""
    deq = quantize_symmetric_per_tensor(weight, bits)
    err = (deq - weight).abs()
    rel = err / weight.abs().clamp_min(1e-9)
    return {
        "abs_error_mean": err.mean().item(),
        "abs_error_max": err.max().item(),
        "rel_error_mean": rel.mean().item(),
        "rel_error_max": rel.max().item(),
        # The element that rounds worst is worth keeping: it is the dimension
        # most exposed at this precision.
        "worst_element_index": int(err.argmax().item()),
    }


# ---------------------------------------------------------------------------
# Measurement B: per-head output error from real activations
# ---------------------------------------------------------------------------


def rms_normalize(x, weight, eps=1e-6):
    """RMSNorm over the last axis, matching Qwen3RMSNorm (float32 statistic)."""
    in_dtype = x.dtype
    x = x.to(torch.float32)
    var = x.pow(2).mean(dim=-1, keepdim=True)
    x = x * torch.rsqrt(var + eps)
    return weight * x.to(in_dtype)


def measure_output_error_per_head(q_pre, weight, bits, eps):
    """Per-head RMSNorm output error induced by quantizing `weight`.

    q_pre is (n_rows, H, head_dim) of real pre-norm query vectors. The full-
    precision and quantized-weight outputs are compared per head; the error is
    reduced to a per-head mean over rows and head_dim.
    """
    deq = quantize_symmetric_per_tensor(weight, bits)
    out_fp = rms_normalize(q_pre, weight, eps)     # (n_rows, H, D)
    out_q = rms_normalize(q_pre, deq, eps)
    # Per-head error: mean absolute difference over rows and head_dim.
    per_head = (out_q - out_fp).abs().mean(dim=(0, 2))   # (H,)
    return per_head.tolist()


# ---------------------------------------------------------------------------
# Activation capture (short forward pass)
# ---------------------------------------------------------------------------


class QPreCapture:
    """Capture pre-norm q per layer via a forward pre-hook on q_norm.

    The pre-hook input to q_norm is exactly the pre-norm, post-reshape q, shaped
    (B, T, H, head_dim). Rows (B*T) are flattened per layer.
    """

    def __init__(self, model):
        self.model = model
        self.handles = []
        self.buf = {}

    def _hook(self, idx):
        def hook(module, args):
            self.buf.setdefault(idx, []).append(args[0].detach())
        return hook

    def attach(self):
        for idx, layer in enumerate(self.model.model.layers):
            h = layer.self_attn.q_norm.register_forward_pre_hook(self._hook(idx))
            self.handles.append(h)
        return self

    def detach(self):
        for h in self.handles:
            h.remove()
        self.handles.clear()

    def stacked(self):
        """Return {layer_idx: (n_rows, H, head_dim)} concatenated over prompts."""
        out = {}
        for idx, chunks in self.buf.items():
            # each chunk is (B, T, H, D); flatten B,T into rows
            flat = [c.reshape(-1, c.shape[-2], c.shape[-1]) for c in chunks]
            out[idx] = torch.cat(flat, dim=0)
        return out


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


@torch.no_grad()
def main():
    cap.assert_qknorm_module_shape()
    print("Shape guard: PASS")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.float32, attn_implementation="eager"
    ).eval()

    eps = model.config.rms_norm_eps
    n_layers = model.config.num_hidden_layers

    prompts = cap._load_mmlu_prompts(N_PROMPTS)

    qpre = QPreCapture(model).attach()
    for p in prompts:
        ids = tokenizer(p, return_tensors="pt").input_ids
        model(ids)
    activations = qpre.stacked()
    qpre.detach()

    result = {"model_id": MODEL_ID, "bits": BITS, "layers": {}}

    for idx in range(n_layers):
        attn = model.model.layers[idx].self_attn
        q_w = attn.q_norm.weight.detach().float()
        k_w = attn.k_norm.weight.detach().float()
        q_pre = activations[idx].float()

        entry = {"q_norm": {}, "k_norm": {}}
        for bits in BITS:
            entry["q_norm"][f"int{bits}"] = {
                "weight_error": measure_weight_error(q_w, bits),
                "output_error_per_head": measure_output_error_per_head(
                    q_pre, q_w, bits, eps
                ),
            }
            # k_norm weight error is measurable; k activations are not captured
            # here (k_norm has its own pre-hook we did not attach), so only the
            # weight-level measurement A is recorded for k.
            entry["k_norm"][f"int{bits}"] = {
                "weight_error": measure_weight_error(k_w, bits),
            }
        result["layers"][str(idx)] = entry
        if (idx + 1) % 7 == 0:
            print(f"  {idx + 1}/{n_layers} layers")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(result))
    print(f"Wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()