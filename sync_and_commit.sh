#!/bin/bash
cd "$(dirname "$0")"

git fetch --all
git add -A

msg="sync: automated commit on $(date '+%Y-%m-%d %H:%M:%S')"
git commit -m "$msg" --allow-empty
git push

echo "✓ Sync complete"
