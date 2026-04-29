# Contributing

Thanks for helping improve MANTIS model iteration tool. This project is aimed at developers who care about reproducible evaluation, secure agent execution, and clear research workflows.

## Local Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e ".[dev]"
```

Run checks before opening a pull request:

```bash
python -m pytest
ruff check .
```

## Development Guidelines

- Keep strategy evaluation causal. New data access helpers must not expose future rows to `Featurizer.compute`.
- Treat API keys, wallets, generated agents, cached data, and deployment configs as local runtime state. Do not commit them.
- Add focused tests for auth, evaluation metrics, challenge behavior, and any public API changes.
- Prefer small pull requests with clear motivation and reproduction steps.
- Update `README.md`, `GUIDE.md`, or `API_guide.md` when changing developer-facing behavior.

## Pull Request Checklist

- Tests pass locally.
- New public behavior is documented.
- No secrets, generated agent workspaces, caches, or wallet artifacts are included.
- Security-sensitive routes fail closed when configuration is missing.
