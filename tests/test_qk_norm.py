"""Verify the repo's claims still hold against the installed transformers.

Silent on success, loud on failure, following the Unix convention: a passing
run prints nothing and exits 0; a failing check raises and exits non-zero. This
does not reimplement any logic. It calls the guards and proofs that already
live in scripts/, so the test and the code it checks cannot drift apart.

What each check answers:

  rms equivalence   : does the from-scratch RMSNorm still equal Qwen3RMSNorm?
  rms properties    : is normalization still per-head and scale-invariant?
  attention shape   : are q_norm/k_norm still Qwen3RMSNorm sized to head_dim?
  attention source  : does Qwen3Attention.forward still order things the way
                      the capture script assumes (norm before RoPE, etc.)?

Run:  python tests/test_qk_norm.py
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import rmsnorm_mechanism as rms
import capture_qk_stats as cap


def test_rms_matches_qwen3():
    result = rms.assert_rms_matches_qwen3()
    assert result["max_abs_diff"] < 1e-5, result


def test_rms_properties():
    result = rms.demo_per_head_axis()
    assert result["head0_changed"], "perturbing one head should change its output"
    assert result["others_unchanged"], "other heads should be untouched (per-head)"
    assert result["scale_invariant"], "RMSNorm should be invariant to per-head scale"


def test_attention_shape():
    cap.assert_qknorm_module_shape()


def test_attention_source():
    cap.assert_qknorm_unchanged()


def main():
    # demo_per_head_axis and the source inspections print to stdout; silence
    # them so a passing run says nothing.
    import contextlib
    import io

    checks = [
        test_rms_matches_qwen3,
        test_rms_properties,
        test_attention_shape,
        test_attention_source,
    ]
    with contextlib.redirect_stdout(io.StringIO()):
        for check in checks:
            check()


if __name__ == "__main__":
    main()