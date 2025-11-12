# GitHub Pages Dashboard

This directory contains the GitHub Pages interface for viewing experiment results.

## Setup

1. **Enable GitHub Pages** in your repository settings:
   - Go to Settings → Pages
   - Source: Deploy from a branch
   - Branch: `main` (or your default branch)
   - Folder: `/docs`

2. **Update results** after running experiments:
   ```bash
   ./scripts/update_gh_pages.sh
   ```

3. **Commit and push** the updated files:
   ```bash
   git add docs/
   git commit -m "Update experiment results"
   git push
   ```

The dashboard will be available at: `https://<username>.github.io/curator-benchmarking/`

## Features

- View all experiment results in a clean, modern interface
- Filter by task or model
- Sort by date, score, or task name
- See key metrics: average score, success rate, sample counts
- View experimental parameters (model, system instructions, prompt)

