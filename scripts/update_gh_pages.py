#!/usr/bin/env python3
"""Script to update GitHub Pages with latest results."""
import shutil
from pathlib import Path

def main():
    repo_root = Path(__file__).parent.parent
    docs_dir = repo_root / "docs"
    results_dir = repo_root / "results"
    docs_results_dir = docs_dir / "results"
    
    # Create docs/results directory if it doesn't exist
    docs_results_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy results files to docs/results
    print("Copying results to docs/results...")
    
    copied = 0
    for file in results_dir.glob("*.json"):
        shutil.copy2(file, docs_results_dir)
        copied += 1
    
    for file in results_dir.glob("*.jsonl"):
        shutil.copy2(file, docs_results_dir)
        copied += 1
    
    print(f"GitHub Pages results updated successfully!")
    print(f"Copied {copied} files to: {docs_results_dir}")

if __name__ == "__main__":
    main()

