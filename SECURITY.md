# Security Policy

## Reporting Vulnerabilities

If you find a vulnerability, please do not open a public issue with exploit details. Report it privately to the project maintainers through the repository security advisory workflow or the maintainer contact channel listed by the publishing organization.

Include:

- Affected commit or release.
- Reproduction steps.
- Impact and reachable attack surface.
- Any logs or proof-of-concept code needed to validate the issue.

## Operational Expectations

MANTIS can execute agent-generated code and user-submitted strategy code. Run it with the same care you would apply to any system that launches containers and spends API credits.

- Never expose the GUI publicly without `MANTIS_AUTH_TOKEN`.
- Never run the Targon agent server without `MANTIS_SERVER_AUTH_KEY`.
- Never run the remote evaluation service without `MANTIS_EVAL_API_KEY`.
- Rotate any API key that was stored in a local file before publishing.
- Keep Docker, Python, Node.js, Claude Code CLI, and deployment images patched.

## Secret Handling

The repository ignores local key/config files, caches, agent workspaces, generated deployment files, and wallet artifacts. Before publishing or submitting changes, run:

```bash
git status --short --ignored
```

Confirm that no local secrets or generated runtime data are staged.
