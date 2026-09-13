# Private evaluation packages

The repository intentionally contains only schemas and SHA-256 manifests for
`router_blind_v1.json`, `top_chat_v3_blind_v1.json` and `jd_blind_v1.json`. It must not contain blind
questions, JD text, gold answers, or per-case evaluation output.

To provision a frozen set, place the independently authored JSON file below
`$OFFERCLAW_PRIVATE_EVAL_ROOT`, calculate its SHA-256, and update only the
matching manifest's `sha256`. Evaluation checks the digest and expected item
count before any model call. An unset root, unset digest, missing file, digest
mismatch, malformed JSON, or count mismatch produces an explicit non-passing
`skipped`/integrity error.

Blind-mode reports contain aggregate metrics only. Once an individual failed
question or label is disclosed for debugging, that corpus must be reclassified
as a development regression set and replaced by a new private blind version.

Commands:

```bash
# Repository development suites
.venv/bin/python eval_intelligent_router.py --set reviewed_regression
.venv/bin/python eval_intelligent_router.py --set surface_invariance
.venv/bin/python eval_jd_analysis.py --set phenomena_dev

# Private release gates (requires a pinned manifest digest)
OFFERCLAW_PRIVATE_EVAL_ROOT=/secure/evals \
  .venv/bin/python eval_intelligent_router.py --set blind --output /tmp/router-aggregate.json
OFFERCLAW_PRIVATE_EVAL_ROOT=/secure/evals \
  .venv/bin/python eval_jd_analysis.py --set blind --output /tmp/jd-aggregate.json
OFFERCLAW_PRIVATE_EVAL_ROOT=/secure/evals \
  .venv/bin/python eval_top_chat_v3.py --output /tmp/top-chat-v3-aggregate.json
```

The top-chat v3 package contains exactly 120 action/query contrasts, 60
multi-turn cases and 40 ambiguity cases. Its authors must attest that they did
not read the router implementation. An unset digest is a failing release gate,
not a skipped pass.
