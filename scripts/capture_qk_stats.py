"""Capture q/k magnitude and attention-logit statistics from Qwen3-0.6B.

Records pre-norm and post-norm q/k statistics per layer, plus post-RoPE
attention logits, under two regimes kept deliberately separate:

    aggregate : 200 MMLU-Redux samples. Answers "what is the typical
                magnitude", giving distribution shape across inputs.
    long_run  : one continuous 4096-token document. Answers "does magnitude
                grow with position", which is the question the FP16 overflow
                claim actually depends on.

Merging the two would blur both signals, so they are stored under separate
keys and never pooled.

Statistics are captured by forward hooks on the `q_norm` and `k_norm`
submodules: the hook input is the pre-norm tensor, the hook output is the
post-norm tensor, so both sides are read from the same hook without
monkeypatching `forward`.

One caveat on measurement point. Hooks on q_norm/k_norm see pre-RoPE
tensors. RoPE is orthogonal, so ||q|| and ||k|| are identical before and
after it, and the norm statistics are valid as captured. The dot product is
not: rotation changes alignment, so q@k differs pre- and post-RoPE. The
attention logits recorded here therefore re-apply RoPE inside the capture
step, replicating one line of Qwen3Attention.forward. A drift guard pins
that replication to the upstream source.

Long-run logits are reduced per position to max and mean rather than stored
in full. Storing every logit would be 28 layers x 16 heads x 4096 positions
x up to 4096 keys. The reduction is lossy by design: it preserves the
extremes the overflow claim depends on and discards the rest.

The model is loaded in float32 on purpose. Measuring whether magnitudes
would overflow FP16 has to happen in a precision that does not overflow:
loading in FP16 would turn any offending value into inf before it could be
recorded.
"""

import inspect
import json
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.qwen3.configuration_qwen3 import Qwen3Config
from transformers.models.qwen3.modeling_qwen3 import (
    Qwen3Attention,
    Qwen3RMSNorm,
    apply_rotary_pos_emb,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
MODEL_ID = "Qwen/Qwen3-0.6B"
OUTPUT_PATH = REPO_ROOT / "results" / "qk_stats.json"

N_SAMPLES = 200
LONG_SEQ_LEN = 4096
MMLU_ID = "edinburgh-dawg/mmlu-redux-2.0"
MMLU_SUBJECTS = [
    "abstract_algebra",
    "anatomy",
    "astronomy",
    "college_computer_science",
    "college_physics",
    "electrical_engineering",
    "high_school_mathematics",
    "machine_learning",
    "philosophy",
    "professional_law",
]
LONG_DOC_ID = "Salesforce/wikitext"
LONG_DOC_CONFIG = "wikitext-103-raw-v1"

FP16_MAX = 65504.0

# Entropy is only counted for query rows with at least this many attendable
# keys. Rows earlier than this attend to too few keys for a low entropy to mean
# anything other than causal short-context. See _logit_stats.
ENTROPY_MIN_CONTEXT = 32


# ---------------------------------------------------------------------------
# Drift guard
# ---------------------------------------------------------------------------


def assert_qknorm_module_shape():
    """Verify q_norm/k_norm are RMSNorm sized to head_dim, on a real module.

    Reads the constructed module rather than `__init__` source. Transformers
    wraps `Qwen3Attention.__init__` with a kernel decorator, so
    `inspect.getsource` on it returns the wrapper body, not the real
    constructor. Checking the built object sidesteps that entirely and has
    the better property anyway: it tests what actually exists at runtime
    rather than what the source appears to say.

    Built on the meta device, so no parameter memory is allocated and no
    weights are downloaded. Same pattern as `gqa_shapes.py`.
    """
    cfg = Qwen3Config(
        hidden_size=1024,
        num_attention_heads=16,
        num_key_value_heads=8,
        head_dim=128,
        num_hidden_layers=28,
    )
    with torch.device("meta"):
        attn = Qwen3Attention(cfg, layer_idx=0)

    for name in ("q_norm", "k_norm"):
        mod = getattr(attn, name, None)
        assert mod is not None, (
            f"Qwen3Attention has no {name}. The per-head normalization claim "
            "in this repo is stale."
        )
        assert isinstance(mod, Qwen3RMSNorm), (
            f"{name} is {type(mod).__name__}, not Qwen3RMSNorm. This repo "
            "documents RMSNorm specifically, not LayerNorm."
        )
        assert tuple(mod.weight.shape) == (attn.head_dim,), (
            f"{name}.weight has shape {tuple(mod.weight.shape)}, expected "
            f"({attn.head_dim},). Normalization is no longer sized to "
            "head_dim, so the per-head claim may be stale."
        )
    return True


def assert_qknorm_unchanged():
    """Fail loudly if Qwen3Attention no longer matches this repo's description.

    Keys on structural operations rather than an exact-text hash: an exact
    hash breaks on whitespace or comment changes, while these checks break
    only if the mechanism itself changes.

    The ordering check deliberately looks for q_norm and the view call inside
    a single statement rather than comparing global indices. If upstream
    splits that nested expression into two statements the guard fails, which
    is the correct outcome: the layout assumptions here would need re-reading
    even though the mechanism is arguably unchanged.
    """
    src = inspect.getsource(Qwen3Attention.forward)
    flat = " ".join(src.split())

    assert "self.q_norm(" in flat and "self.k_norm(" in flat, (
        "Qwen3Attention.forward no longer applies q_norm/k_norm. The pre/post "
        "norm capture in this script assumes they exist. Re-read the source."
    )
    assert "apply_rotary_pos_emb(" in flat, (
        "Qwen3Attention.forward no longer calls apply_rotary_pos_emb. The "
        "logit replication here would no longer match. Re-read the source."
    )

    # q_norm must wrap the reshape in one statement: norm sees (B, T, H, D).
    statements = [s.strip() for s in src.splitlines() if "self.q_norm(" in s]
    assert statements, "could not isolate the q_norm statement"
    q_stmt = " ".join(statements[0].split())
    assert ".view(" in q_stmt, (
        "q_norm no longer wraps a .view(...) call in a single statement. The "
        "captured tensor layout assumption (B, T, H, D) may be wrong. "
        "Re-read Qwen3Attention.forward."
    )
    assert ".transpose(" in q_stmt, (
        "the transpose no longer sits in the same statement as q_norm. This "
        "repo documents the order view -> norm -> transpose -> RoPE; verify "
        "it still holds."
    )

    # RoPE must come after the norm block.
    assert flat.index("self.q_norm(") < flat.index("apply_rotary_pos_emb("), (
        "apply_rotary_pos_emb now precedes q_norm. The norm-before-RoPE "
        "ordering this repo documents is stale."
    )

    assert_qknorm_module_shape()
    return True


# ---------------------------------------------------------------------------
# Statistic reductions
# ---------------------------------------------------------------------------


def _norm_stats(t):
    """Reduce a (B, T, H, D) tensor to per-head vector-norm statistics.

    Head-dim is the last axis, matching the layout q_norm and k_norm operate
    on. Returns per-head mean and max of the head-dim vector norms, plus the
    global max absolute element.
    """
    vec_norms = t.float().norm(dim=-1)  # (B, T, H)
    return {
        "norm_mean_per_head": vec_norms.mean(dim=(0, 1)).tolist(),
        "norm_max_per_head": vec_norms.amax(dim=(0, 1)).tolist(),
        "abs_max": t.float().abs().max().item(),
    }


def _row_entropy(logits, keep_b):
    """Per-row attention entropy in nats, for one chunk of query rows.

    logits is (B, H, c, T); keep_b is the causal boolean mask (1, 1, c, T).
    Softmax runs over the attendable keys of each row (masked keys get -inf and
    so receive exactly zero probability), then entropy is S = -sum p log p.
    torch.special.entr applies the 0 log 0 = 0 convention without producing
    NaNs. Returns (B, H, c).
    """
    masked_logits = logits.masked_fill(~keep_b, float("-inf"))
    probs = torch.softmax(masked_logits, dim=-1)
    return torch.special.entr(probs).sum(dim=-1)


def assert_row_entropy_correct():
    """Verify _row_entropy against closed-form values on known distributions.

    Two checks that pin the entropy convention without a model:
      - Flat logits over n attendable keys give a uniform softmax, whose entropy
        is exactly log(n). Row i (0-indexed, causal) attends to i+1 keys, so its
        entropy must be log(i+1).
      - A logit spike on one key drives the softmax to near-one-hot, whose
        entropy must be ~0. This also exercises the 0 log 0 = 0 path, which must
        not produce NaN.
    """
    import math

    T = 6
    rows = torch.arange(T).unsqueeze(1)
    cols = torch.arange(T).unsqueeze(0)
    keep_b = (cols <= rows).view(1, 1, T, T)

    flat = _row_entropy(torch.zeros(1, 1, T, T), keep_b)[0, 0]
    for i in range(T):
        expected = math.log(i + 1)
        assert abs(flat[i].item() - expected) < 1e-5, (
            f"flat-logit entropy at row {i} was {flat[i].item()}, "
            f"expected log({i + 1}) = {expected}"
        )

    peaked = torch.zeros(1, 1, T, T)
    peaked[0, 0, :, 0] = 50.0
    ent = _row_entropy(peaked, keep_b)[0, 0]
    assert not torch.isnan(ent).any(), "entropy produced NaN (0 log 0 path)"
    assert ent.max().item() < 1e-3, (
        f"near-one-hot entropy should be ~0, got max {ent.max().item()}"
    )
    return True


def _logit_stats(q, k, scaling, want_per_position):
    """Post-RoPE attention logits, reduced.

    q arrives as (B, H, T, D) and k as (B, H_kv, T, D). KV heads are expanded
    to match q via repeat_interleave, which is equivalent to what repeat_kv
    does downstream (see qwen3_gqa_mechanism.ipynb for the storage-pointer
    proof of that equivalence).

    Only the causal lower triangle is counted. The upper triangle is never
    realized in a real forward pass, so including it would report statistics
    over logits the model never actually computes.

    The full (H, T, T) logit matrix is never materialized at once. It is built
    in query-position chunks so peak memory stays at chunk_q x T rather than
    T x T. At T=4096 the full matrix is ~1GB per head-batch in float32; the
    chunked form keeps it to a fraction of that.
    """
    B, H, T, _ = q.shape
    n_rep = H // k.shape[1]
    k_exp = torch.repeat_interleave(k, dim=1, repeats=n_rep).float()

    chunk = 512
    running_absmax = 0.0
    running_sum = 0.0
    running_count = 0
    abs_samples = []
    per_pos_max = [] if want_per_position else None
    per_pos_mean = [] if want_per_position else None

    # Per-head attention entropy accumulation. Entropy is a property of the
    # softmax over each query row, so it must be computed row by row inside the
    # chunk loop, on the causally masked logits, before the full matrix is
    # discarded. Collected per head, then reduced to mean and min across query
    # rows: the mean is the head's typical sharpness, the min catches a head
    # that is healthy on average but collapses to near-one-hot at some rows,
    # which a mean alone would hide.
    #
    # Only rows with at least ENTROPY_MIN_CONTEXT attendable keys are counted.
    # Early rows can attend to only a handful of keys (row 0 to exactly one), so
    # their entropy is near zero by causal structure, not by collapse. Including
    # them would make entropy_min trivially zero for every head and destroy its
    # diagnostic value. The floor restricts the statistic to positions where a
    # low entropy actually means the head chose to be sharp.
    ent_sum = torch.zeros(H, dtype=torch.float64)
    ent_count = 0
    ent_min = torch.full((H,), float("inf"), dtype=torch.float64)

    for start in range(0, T, chunk):
        end = min(start + chunk, T)
        q_chunk = q[:, :, start:end, :].float()          # (B, H, c, D)
        logits = torch.matmul(q_chunk, k_exp.transpose(-1, -2)) * scaling  # (B,H,c,T)

        # Causal mask for these query rows: query at absolute position p may
        # attend to keys 0..p inclusive.
        rows = torch.arange(start, end).unsqueeze(1)
        cols = torch.arange(T).unsqueeze(0)
        keep = (cols <= rows)                            # (c, T) bool

        keep_b = keep.view(1, 1, end - start, T)
        finite = logits[keep_b.expand_as(logits)]

        running_absmax = max(running_absmax, finite.abs().max().item())
        running_sum += finite.sum().item()
        running_count += finite.numel()
        # Subsample for the p99 estimate to bound memory; exact p99 over ~34M
        # values is unnecessary for the overflow claim.
        abs_samples.append(finite.abs()[:: max(1, finite.numel() // 100_000)])

        # Per-head entropy for this chunk. See _row_entropy; restricted below
        # to rows with enough attendable context.
        row_entropy = _row_entropy(logits, keep_b)  # (B, H, c), nats

        # Restrict to rows with enough attendable context (absolute position
        # >= ENTROPY_MIN_CONTEXT). row_ctx counts attendable keys per row.
        row_ctx = keep.sum(dim=-1)                       # (c,)
        enough = row_ctx >= ENTROPY_MIN_CONTEXT          # (c,)
        if enough.any():
            re = row_entropy[:, :, enough]               # (B, H, c')
            ent_sum += re.sum(dim=(0, 2)).double()
            ent_count += re.shape[0] * re.shape[2]
            ent_min = torch.minimum(ent_min, re.amin(dim=(0, 2)).double())

        if want_per_position:
            masked = logits.masked_fill(~keep_b, float("nan"))
            per_pos_max.extend(
                torch.nan_to_num(masked, nan=-float("inf"))
                .amax(dim=(0, 1, 3))
                .tolist()
            )
            per_pos_mean.extend(masked.nanmean(dim=3).mean(dim=(0, 1)).tolist())

        del logits

    abs_all = torch.cat(abs_samples)

    # If no row met the context floor (short prompts in the aggregate regime),
    # entropy is undefined for this pass. Emit null rather than inf/NaN, which
    # are not valid JSON, and let the consumer skip them.
    if ent_count > 0:
        entropy_mean = (ent_sum / ent_count).tolist()
        entropy_min = ent_min.tolist()
    else:
        entropy_mean = None
        entropy_min = None

    out = {
        "abs_max": running_absmax,
        "mean": running_sum / running_count,
        "p99_abs": torch.quantile(abs_all.float(), 0.99).item(),
        # Per-head entropy in nats over rows with >= ENTROPY_MIN_CONTEXT keys.
        # mean = typical sharpness, min = worst (most collapsed) qualifying row.
        # null when no row qualified (short-prompt aggregate samples).
        "entropy_mean_per_head": entropy_mean,
        "entropy_min_per_head": entropy_min,
    }
    if want_per_position:
        out["per_position_max"] = per_pos_max
        out["per_position_mean"] = per_pos_mean
    return out


# ---------------------------------------------------------------------------
# Capture
# ---------------------------------------------------------------------------


class QKCapture:
    """Per-layer q/k statistics computed inside the hooks, layer by layer.

    The naive approach, holding q and k for all 28 layers and reducing them
    afterward, peaks at the whole model's attention activations at once. For a
    4096-token forward in float32 that is tens of GB and OOMs a 16GB machine.

    Instead, each layer's statistics are computed the moment that layer's
    q_norm/k_norm fire, and the raw tensors are dropped before the next layer
    runs. Peak memory is therefore one layer's attention matrix, not all of
    them.

    RoPE needs the per-position cos/sin. Rather than recompute them (which
    would require position_ids the hook does not see, and a dependency on the
    exact location of rotary_emb in the model tree), a pre-hook on the
    attention module captures the `position_embeddings=(cos, sin)` tuple that
    Qwen3Attention.forward already receives as an argument. This is both
    cheaper and less fragile: it reads what the layer is actually given.
    """

    def __init__(self, model):
        self.model = model
        self.handles = []
        self.want_per_position = False
        self._pos_emb = {}   # layer_idx -> (cos, sin) for the current forward
        self._pending = {}   # layer_idx -> partial {q_pre, q_post, ...}
        self.results = {}    # layer_idx -> reduced statistics

    def _attn_pre_hook(self, layer_idx):
        def hook(module, args, kwargs):
            # position_embeddings is the 2nd positional arg in forward, but may
            # arrive by keyword. Handle both.
            if "position_embeddings" in kwargs:
                self._pos_emb[layer_idx] = kwargs["position_embeddings"]
            else:
                self._pos_emb[layer_idx] = args[1]
        return hook

    def _norm_hook(self, layer_idx, which):
        def hook(module, args, output):
            slot = self._pending.setdefault(layer_idx, {})
            slot[f"{which}_pre"] = args[0].detach()
            slot[f"{which}_post"] = output.detach()
            # k_norm fires after q_norm in forward, so once both q and k are
            # present this layer is complete and can be reduced immediately.
            if "q_post" in slot and "k_post" in slot:
                self._reduce_layer(layer_idx)
        return hook

    @torch.no_grad()
    def _reduce_layer(self, layer_idx):
        slot = self._pending.pop(layer_idx)
        attn = self.model.model.layers[layer_idx].self_attn
        cos, sin = self._pos_emb.pop(layer_idx)

        # (B, T, H, D) -> (B, H, T, D), matching the transpose in forward.
        q = slot["q_post"].transpose(1, 2)
        k = slot["k_post"].transpose(1, 2)
        q_rope, k_rope = apply_rotary_pos_emb(q, k, cos, sin)

        self.results[str(layer_idx)] = {
            "q_pre": _norm_stats(slot["q_pre"]),
            "q_post": _norm_stats(slot["q_post"]),
            "k_pre": _norm_stats(slot["k_pre"]),
            "k_post": _norm_stats(slot["k_post"]),
            "logits": _logit_stats(
                q_rope, k_rope, attn.scaling, self.want_per_position
            ),
        }
        # Raw tensors go out of scope here, before the next layer's forward.

    def attach(self):
        for idx, layer in enumerate(self.model.model.layers):
            attn = layer.self_attn
            self.handles.append(
                attn.register_forward_pre_hook(
                    self._attn_pre_hook(idx), with_kwargs=True
                )
            )
            self.handles.append(
                attn.q_norm.register_forward_hook(self._norm_hook(idx, "q"))
            )
            self.handles.append(
                attn.k_norm.register_forward_hook(self._norm_hook(idx, "k"))
            )
        return self

    def detach(self):
        for h in self.handles:
            h.remove()
        self.handles.clear()

    def collect(self, want_per_position):
        """Run-mode setup: call before the forward pass, read `results` after.

        Returns the results dict populated during the forward pass and resets
        internal state for the next call.
        """
        self.want_per_position = want_per_position
        self.results = {}
        self._pos_emb.clear()
        self._pending.clear()
        return self.results


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------


def _load_mmlu_prompts(n_samples):
    """MMLU-Redux questions pooled across subjects, clean entries only.

    MMLU-Redux has no combined "all" config, so subjects are loaded
    individually and interleaved round-robin rather than concatenated. Taking
    the first N from a concatenated list would draw almost entirely from the
    first subject or two, and prompt length and vocabulary correlate strongly
    with subject. Since the magnitudes measured here depend on hidden states,
    and hidden states depend on prompt content and length, a single-domain
    slice would confound domain with the magnitude trend being measured.

    error_type == "ok" excludes items the dataset authors flagged as
    malformed, matching the filtering convention used elsewhere in this
    project. Prompts are questions only, with no exemplars and no chat
    template, since this measures activation magnitude rather than task
    performance.
    """
    per_subject = []
    for subject in MMLU_SUBJECTS:
        ds = load_dataset(MMLU_ID, subject, split="test")
        ds = ds.filter(lambda r: r["error_type"] == "ok")
        rows = [r["question"] for r in ds]
        if rows:
            per_subject.append(rows)

    assert per_subject, f"no clean rows found across {len(MMLU_SUBJECTS)} subjects"

    pooled = []
    for i in range(max(len(s) for s in per_subject)):
        for subject_rows in per_subject:
            if i < len(subject_rows):
                pooled.append(subject_rows[i])
            if len(pooled) >= n_samples:
                return pooled
    return pooled[:n_samples]


def _load_long_document(tokenizer, target_len):
    """One continuous document of exactly target_len tokens.

    Continuity matters. Concatenating short samples would introduce document
    boundaries, and any position-wise trend measured across them would be an
    artifact of those boundaries rather than a property of the model.
    """
    ds = load_dataset(LONG_DOC_ID, LONG_DOC_CONFIG, split="train", streaming=True)
    buf = []
    ids = torch.empty(1, 0, dtype=torch.long)

    for row in ds:
        text = row["text"].strip()
        if not text:
            continue
        buf.append(text)
        if len(buf) % 16:
            continue
        ids = tokenizer("\n\n".join(buf), return_tensors="pt").input_ids
        if ids.shape[-1] >= target_len:
            break

    assert ids.shape[-1] >= target_len, (
        f"could not assemble {target_len} contiguous tokens from {LONG_DOC_ID}"
    )
    return ids[:, :target_len]


# ---------------------------------------------------------------------------
# Regimes
# ---------------------------------------------------------------------------


@torch.no_grad()
def run_aggregate(model, tokenizer, capture, n_samples):
    """Per-sample statistics across MMLU prompts.

    Every sample is stored separately rather than accumulated, so the
    notebook can plot the distribution shape per layer instead of a single
    summary point.
    """
    prompts = _load_mmlu_prompts(n_samples)
    per_sample = []

    for i, prompt in enumerate(prompts):
        ids = tokenizer(prompt, return_tensors="pt").input_ids
        results = capture.collect(want_per_position=False)
        model(ids)
        # collect() returns the live dict the hooks fill; copy it out before
        # the next sample resets it.
        per_sample.append(dict(results))
        if (i + 1) % 25 == 0:
            print(f"  aggregate: {i + 1}/{len(prompts)}")

    return {
        "n_samples": len(per_sample),
        "note": "one entry per sample, not pooled",
        "samples": per_sample,
    }


@torch.no_grad()
def run_long(model, tokenizer, capture, seq_len):
    """Position-wise statistics over one continuous document."""
    ids = _load_long_document(tokenizer, seq_len)
    print(f"  long_run: forward over {ids.shape[-1]} tokens")
    results = capture.collect(want_per_position=True)
    model(ids)
    return {
        "seq_len": int(ids.shape[-1]),
        "source": f"{LONG_DOC_ID}/{LONG_DOC_CONFIG}",
        "layers": dict(results),
    }


def _save(result):
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(result))
    print(f"Wrote {OUTPUT_PATH}")


def main():
    assert_qknorm_unchanged()
    print("Drift guard: PASS")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        dtype=torch.float32,
        attn_implementation="eager",
    ).eval()

    capture = QKCapture(model).attach()
    result = {"model_id": MODEL_ID, "fp16_max": FP16_MAX}

    try:
        result["aggregate"] = run_aggregate(model, tokenizer, capture, N_SAMPLES)
        # Written before the long run starts. The long run is the memory-heavy
        # half and the more likely one to fail, and there is no reason for its
        # failure to discard several minutes of completed aggregate work.
        _save(result)

        result["long_run"] = run_long(model, tokenizer, capture, LONG_SEQ_LEN)
        _save(result)
    finally:
        capture.detach()


if __name__ == "__main__":
    main()