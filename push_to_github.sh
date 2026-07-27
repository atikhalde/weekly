#!/usr/bin/env bash
# ---------------------------------------------------------------------------
#  One-shot uploader for the weekly-breakout-alerts project.
#
#  Fixes the usual web-upload problem: browsers silently skip the hidden
#  .github folder, so the repo lands with no workflows and no automation.
#  Git does not care about hidden files, so pushing this way always works.
#
#  Usage:
#     bash push_to_github.sh https://github.com/YOUR_USERNAME/YOUR_REPO.git
# ---------------------------------------------------------------------------
set -euo pipefail

REMOTE="${1:-}"

if [[ -z "$REMOTE" ]]; then
  echo "ERROR: pass your repo URL."
  echo "   bash push_to_github.sh https://github.com/USERNAME/REPO.git"
  exit 1
fi

if ! command -v git >/dev/null 2>&1; then
  echo "ERROR: git is not installed."
  echo "   macOS:    xcode-select --install"
  echo "   Windows:  https://git-scm.com/download/win"
  echo "   Linux:    sudo apt install git"
  exit 1
fi

# must be run from inside the project folder
if [[ ! -f "config.yaml" || ! -f "scan.py" ]]; then
  echo "ERROR: run this from inside the weekly-breakout-flat folder."
  echo "   cd path/to/weekly-breakout-flat"
  echo "   bash push_to_github.sh $REMOTE"
  exit 1
fi

if [[ ! -f ".github/workflows/scan.yml" ]]; then
  echo "ERROR: .github/workflows/scan.yml is missing from this folder."
  echo "   Re-extract the zip - the workflows did not survive unzipping."
  exit 1
fi

echo "==> Preparing repository"
git init -q 2>/dev/null || true
git add -A

echo "==> Files staged (verify .github/workflows is listed):"
git diff --cached --name-only | sed 's/^/    /'

# fail loudly rather than pushing a repo with no automation
if ! git diff --cached --name-only | grep -q '^\.github/workflows/'; then
  echo
  echo "ERROR: the workflow files were not staged. Check .gitignore."
  exit 1
fi

git -c user.email="you@example.com" \
    -c user.name="Breakout Scanner" \
    commit -q -m "Weekly breakout scanner" 2>/dev/null || echo "==> Nothing new to commit"

git branch -M main
git remote remove origin 2>/dev/null || true
git remote add origin "$REMOTE"

echo "==> Pushing to $REMOTE"
echo "    If prompted for a password, use a Personal Access Token, not your"
echo "    GitHub password: https://github.com/settings/tokens"
echo

git push -u origin main --force

echo
echo "==> Done. Now in your repo:"
echo "    1. Confirm .github/workflows/ shows 3 files"
echo "    2. Settings > Secrets and variables > Actions  -> add the 4 secrets"
echo "    3. Settings > Actions > General -> Read and write permissions"
echo "    4. Actions > Weekly Snapshot > Run workflow (limit: 50)"
