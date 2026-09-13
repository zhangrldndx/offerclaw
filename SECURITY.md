# Security and privacy

OfferClaw is designed to keep credentials and personal career data outside Git.

## Never commit

- API keys, access tokens, passwords, gateway addresses, or private service endpoints
- `user_profile.md`, application records, daily logs, resumes, interview notes, or private JD copies
- screenshots, machine inventories, absolute user paths, or raw model/judge artifacts

Use `.env.local` and the ignored runtime directories for local data. Public examples must use
synthetic identities, loopback addresses, or reserved example domains.

## Before pushing

Run:

```bash
python verify_docs.py --privacy-revision HEAD
set -o pipefail
git log --no-ext-diff --full-history -p HEAD | gitleaks stdin --no-banner --redact
python -m pytest tests/ -q
```

Activate the fail-closed pre-push hook once per clone, and tell it where Gitleaks is installed
when the executable is not already on `PATH`:

```bash
git config core.hooksPath .githooks
git config --local offerclaw.gitleaksPath /absolute/path/to/gitleaks
```

The hook scans the complete history reachable from every revision being pushed, verifies the
Gitleaks rule engine with a synthetic canary, and refuses to push if the scanner is unavailable or
unhealthy. CI repeats the reachable-history privacy and secret scans. GitHub Secret Scanning and
Push Protection should remain enabled in repository settings.

## If sensitive data is exposed

Revoke or rotate the credential first. Then remove the data from every reachable commit and
force-update the affected refs. Existing forks, clones, caches, releases, and workflow artifacts
are separate copies and must be handled with their owners or the hosting provider.
