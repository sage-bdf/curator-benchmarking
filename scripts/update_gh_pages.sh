#!/bin/bash
# Script to update GitHub Pages with latest results

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
DOCS_DIR="$REPO_ROOT/docs"
RESULTS_DIR="$REPO_ROOT/results"
DOCS_RESULTS_DIR="$DOCS_DIR/results"

# Create docs/results directory if it doesn't exist
mkdir -p "$DOCS_RESULTS_DIR"

# Copy results files to docs/results
echo "Copying results to docs/results..."
cp "$RESULTS_DIR"/*.json "$DOCS_RESULTS_DIR/" 2>/dev/null || true
cp "$RESULTS_DIR"/*.jsonl "$DOCS_RESULTS_DIR/" 2>/dev/null || true

echo "GitHub Pages results updated successfully!"
echo "Files copied to: $DOCS_RESULTS_DIR"

