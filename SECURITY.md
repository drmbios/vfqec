# Security policy

Security fixes target the latest commit on `main`. Older snapshots, including
the recorded scientific benchmarks, are retained for provenance rather than
maintained as supported application releases.

## Report a vulnerability privately

Use [GitHub private vulnerability reporting](https://github.com/drmbios/vfqec/security/advisories/new).
Include the affected commit, reproduction steps, and impact. Do not post API
keys, credentials, or exploit details in public issues or pull requests.

## Deployment boundaries

The GitHub Pages demo only replays public, recorded data. It does not contain
API credentials or submit cloud jobs. The Python API and dashboard are intended
for local use: keep the Compose loopback bindings. They have no authentication
and must not be exposed directly to an untrusted network.

Redis and Postgres are internal services; do not publish their ports. Redis is
trusted infrastructure because RQ workers deserialize queued jobs. Set a private
Postgres password for shared installations. Configure provider credentials in
an ignored `.env` or environment variables; never commit them or store them in
Actions variables. Revoke and replace a credential if it becomes exposed.

## Automated checks

- GitHub secret scanning and push protection guard supported credential types.
- Gitleaks scans Git history, including a custom BlueQubit assignment rule.
- CodeQL scans Python, JavaScript, and Actions using the extended query suite.
- Dependabot alerts and security updates track vulnerable dependencies; weekly
  update proposals cover Python, Actions, and Docker manifests.
- Security CI audits the complete dependency lock and build tools with
  `pip-audit`, and checks Python for medium/high severity defects with Bandit.

Keep exact versions in `pyproject.toml` and `requirements.lock` synchronized
when updating dependencies. Validate with tests, a local E1 smoke run, and the
runtime Docker build. Action references and container images are pinned by
commit or digest; review updates before merging them.
