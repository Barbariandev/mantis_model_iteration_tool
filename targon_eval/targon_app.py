"""Targon SDK deployment for the MANTIS full agent server.

Deploys the complete agent management server — Claude Code CLI,
walk-forward evaluation, and all agent lifecycle management — to
Targon. No local Docker containers needed; everything runs remotely.

Deploy manually::

    targon deploy mantis_model_iteration_tool/targon_eval/targon_app.py --name mantis-server

Run ephemerally::

    targon run mantis_model_iteration_tool/targon_eval/targon_app.py

For automated deployment via the GUI, see targon_deploy.py which
generates a configured version of this file with credentials baked in.
"""

import targon
from pathlib import Path

PKG_DIR = Path(__file__).resolve().parent.parent

IGNORE_PATTERNS = [
    "agents", ".data", "__pycache__", "*.pyc",
    ".targon_config", ".anthropic_key", ".coinglass_key",
    "_targon_deploy_*", "*.tmp", "*.lock", "templates",
]

image = (
    targon.Image.debian_slim()
    .apt_install("curl", "git", "nodejs", "npm")
    .run_commands("npm install -g @anthropic-ai/claude-code")
    .pip_install(
        "numpy", "pandas", "scipy", "scikit-learn",
        "requests", "pyarrow", "fastapi", "uvicorn",
        "boto3", "bittensor", "cryptography",
    )
    .add_local_dir(str(PKG_DIR), "/app/mantis_model_iteration_tool",
                   ignore=IGNORE_PATTERNS)
    .env({"PYTHONPATH": "/app"})
    .workdir("/app")
)

app = targon.App(name="mantis-server", image=image)


@app.function(
    resource=targon.Compute.CPU_LARGE,
    min_replicas=1,
    max_replicas=2,
    timeout=3600,
)
@targon.web_server(port=8080, startup_timeout=600)
def serve():
    import subprocess
    subprocess.Popen(
        ["python", "-m", "mantis_model_iteration_tool.targon_server"],
        env={**__import__("os").environ, "PORT": "8080"},
    )


@app.local_entrypoint()
def main():
    serve.remote()


if __name__ == "__main__":
    print("Use the CLI to deploy:")
    print(f"  targon deploy {__file__} --name mantis-server")
