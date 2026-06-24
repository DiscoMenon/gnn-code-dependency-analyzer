import subprocess
import os
from pathlib import Path

REPOS = [
    "https://github.com/psf/requests",
    "https://github.com/pallets/flask",
    "https://github.com/tiangolo/fastapi",
    "https://github.com/httpie/cli",
    "https://github.com/Textualize/rich",
    "https://github.com/pallets/click",
    "https://github.com/encode/httpx",
    "https://github.com/tartley/colorama",
    "https://github.com/tqdm/tqdm",
]

def download_all(base_path: str = "data/repos"):
    Path(base_path).mkdir(parents=True, exist_ok=True)
    
    for url in REPOS:
        repo_name = url.split("/")[-1]
        target = os.path.join(base_path, repo_name)
        
        if Path(target).exists():
            print(f"  Already exists: {repo_name}")
            continue
        
        print(f"  Cloning {repo_name}...")
        result = subprocess.run(
            ["git", "clone", "--depth=1", url, target],
            capture_output=True, text=True
        )
        if result.returncode == 0:
            print(f"  Done: {repo_name}")
        else:
            print(f"  Failed: {repo_name} — {result.stderr[:100]}")

if __name__ == "__main__":
    download_all()