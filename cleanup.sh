#!/bin/bash

# Remove macOS system files
find . -name ".DS_Store" -delete

# Remove IDE settings
rm -rf .vscode/
rm -rf .idea/

# Remove Jekyll-related files and directories
rm -rf _posts/
rm -rf _site/
rm -rf _sass/
rm -rf _includes/
rm -rf _layouts/
rm -rf _drafts/
rm -rf _authors/
rm -rf .jekyll-cache/
rm -f _config.yml
rm -f about.markdown
rm -f index.markdown
rm -f 404.html
rm -f Gemfile
rm -f Gemfile.lock

# Remove Python cache files
find . -type d -name "__pycache__" -exec rm -rf {} +
find . -type d -name ".ipynb_checkpoints" -exec rm -rf {} +

# Remove temporary or test files
rm -f bot.sh
rm -f func_test.py
rm -f code_used_google.md

# Remove backup files
rm -f *~
rm -f *.bak

# Clean up empty directories
find . -type d -empty -delete

echo "Repository cleanup completed!" 