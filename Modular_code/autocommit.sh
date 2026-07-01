#!/bin/bash

# Go to the project root (parent of code_folder)
cd "$(dirname "$0")/.."

while true
do
    git add Results/

    if ! git diff --cached --quiet; then
        git commit -m "Autosave $(date '+%Y-%m-%d %H:%M:%S')"
    fi

    sleep 300    # 5 minutes
done