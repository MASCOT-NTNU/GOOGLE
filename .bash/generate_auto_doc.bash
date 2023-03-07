#!/bin/bash
cd ../Publish/docs
sphinx-apidoc -f -P -o ./source ../src
make html
echo "Documentation generated in docs/_build/html/index.html"

# Then run the following command to replace the folder name from _static to static to avoid 404 error. 
sed -i '' 's/_static/static/g' build/html/*.html
mv build/html/_static build/html/static

# To open the documentation in the browser
open build/html/index.html
