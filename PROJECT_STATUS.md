# OfferClaw project status

Last reviewed: 2026-09-13

## Current scope

OfferClaw is a local-first career workflow application. The repository includes
the matching engine, multi-agent workflow, RAG/query services, profile review,
resume drafting, application tracking, WeChat/OpenClaw integration, and two web
interfaces.

The public default profile and all committed job fixtures are synthetic. Real
profiles, applications, job captures, conversations, memory, logs, screenshots,
and generated evaluation traces are local runtime data and are excluded from
Git.

## Verification baseline

- API route declarations and documented route count are cross-checked by
  `verify_docs.py`.
- The public knowledge index count and aggregate metrics are cross-checked
  without publishing the underlying private corpus.
- Repository privacy checks reject credentials, non-loopback literal service
  endpoints, local machine identities, private workspace links, local profile
  facts, runtime data paths, and raw RAG evaluation artifacts.
- CI and the pre-push hook run the documentation/privacy gate, secret scanning,
  and the test suite.
- OpenClaw direct-reply behavior has a separate Node.js regression suite.

## Data boundary

Public:

- Source code and architecture documentation
- `.env.example` with placeholders only
- Explicitly synthetic profiles and test fixtures
- Aggregate, non-identifying quality metrics

Local only:

- `.env.local` and credentials
- `user_profile.md` and `profiles/private_*.json`
- Application/JD stores, daily records, memory, plans, logs, and resumes
- Knowledge-base source material and vector databases
- Screenshots and raw evaluation datasets, traces, prompts, and model replies

See `SECURITY.md` and `docs/rag_eval/README.md` for the publication rules.

## Known limitations

- Raw RAG benchmarks are intentionally unavailable in a fresh public clone, so
  artifact-reproducibility tests report an explicit skip without the private
  bundle.
- External LLM and messaging integrations require user-supplied local
  configuration.
- Matching output is decision support, not a hiring guarantee.
