#!/bin/bash

# git_push.sh
# Usage: ./git_push.sh "Your commit message"

# Check if commit message is provided
if [ $# -eq 0 ]; then
    echo "Error: Commit message is required"
    echo "Usage: $0 \"Your commit message\""
    exit 1
fi

# Store the commit message
COMMIT_MSG="$1"

# Configure git credentials
git config --global user.name "your_user_name"
git config --global user.email <your_email>

# Add all files
echo "Adding files..."
git add .
git add -f tokenizer/*  # In case tokenizer is in .gitignore
echo "data/" > .gitignore
echo "logs/" >> .gitignore

# Commit with provided message
echo "Committing with message: $COMMIT_MSG"
git commit -m "$COMMIT_MSG"

# Push to main branch
echo "Pushing to main branch..."
git push -u origin main

# Check if any command failed
if [ $? -eq 0 ]; then
    echo "Successfully pushed to repository"
else
    echo "Error: Failed to push to repository"
    exit 1
fi
