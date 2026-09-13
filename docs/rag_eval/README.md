# RAG evaluation privacy boundary

Raw RAG evaluation datasets, traces, model responses, judge outputs, machine
metadata, and job-source captures are intentionally not published. They may
contain excerpts from a local knowledge base or enough context to reconstruct a
user profile.

The public repository contains only aggregate metrics in `metrics.json` and
explicitly synthetic fixtures under `tests/`. Artifact-based tests skip with a
clear reason when the private evaluation bundle is unavailable.

Keep generated artifacts in this directory locally. Do not force-add them to
Git. For portable private evaluation, use a repository-external directory and
pin its manifest by SHA-256.
