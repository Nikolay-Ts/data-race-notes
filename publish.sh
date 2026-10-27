#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

# Initialize a local git repo in data-races if one doesn't exist right here
if [ ! -d ".git" ]; then
  echo "Initializing git repository in $(pwd)..."
  git init -b main
fi

DEFAULT_REMOTE="git@gitlabense.imtbs-tsp.eu:data-race/notes.git"

# Configure remote if passed as an argument or if missing
if [ "${1:-}" != "" ] && [[ "${1:-}" == http* || "${1:-}" == git@* ]]; then
  if git remote get-url origin >/dev/null 2>&1; then
    git remote set-url origin "$1"
  else
    git remote add origin "$1"
  fi
  shift
elif ! git remote get-url origin >/dev/null 2>&1; then
  git remote add origin "$DEFAULT_REMOTE"
fi

COMMIT_MSG="${1:-Update data-races notes ($(date +'%Y-%m-%d %H:%M'))}"

git add -A
# Ensure documentation markdown files are never staged even if previously tracked
git rm --cached -f documentation.md Documentation.md DOCUMENTATION.md 2>/dev/null || true

if git diff --cached --quiet; then
  echo "No new changes to publish."
else
  git commit -m "$COMMIT_MSG"
fi

BRANCH="$(git branch --show-current)"
git push -u origin "$BRANCH"
echo "Published! GitLab CI is now building and deploying your Quartz site."
