#!/usr/bin/env bash
set -euo pipefail

git init
git add .
git commit -m "chore: initialize analytics project foundation"

echo
echo "Next:"
echo "1. Create an empty GitHub repository."
echo "2. Add it as origin."
echo "3. Push main."
