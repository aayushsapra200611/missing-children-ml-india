#!/usr/bin/env bash
# ==============================================================================
# push_to_github.sh — One-Command GitHub Pusher
# Author: Aayush Sapra (Roll No. 24570096)
# ==============================================================================
set -e

REPO_URL=$1
if [ -z "$REPO_URL" ]; then
    read -p "Enter your GitHub repository URL (e.g. https://github.com/<username>/missing-children-ml-india.git): " REPO_URL
fi

if [ -n "$REPO_URL" ]; then
    git remote remove origin 2>/dev/null || true
    git remote add origin "$REPO_URL"
    echo "Pushing project to GitHub (main branch)..."
    git push -u origin main
    echo "Successfully pushed to: $REPO_URL"
else
    echo "No URL provided."
fi
