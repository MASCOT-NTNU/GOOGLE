#!/bin/bash
# Build the Jekyll site
cd ../
bundle exec jekyll serve --livereload --draft --future --incremental --watch
