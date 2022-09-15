#!bash/bin

git add .
read -p "what to commit: " string
git commit -m $string
git push --all
