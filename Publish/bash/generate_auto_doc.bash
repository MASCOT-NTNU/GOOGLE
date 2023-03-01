cd ../docs
sphinx-apidoc -f -o ./source ../src
make html
echo "Documentation generated in docs/_build/html/index.html"
open build/html/index.html
