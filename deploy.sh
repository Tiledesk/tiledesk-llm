# npm version patch
version="0.1.20"
echo "version $version"

# Get curent branch name
current_branch=$(git rev-parse --abbrev-ref HEAD)
remote_name=$(git config --get branch.$current_branch.remote)

git add .
git commit -m "Created a new version: $version"
git push "$remote_name" "$current_branch"

if [ "$version" != "" ]; then
    git tag -a "$version" -m "`git log -1 --format=%s`"
    git push "$remote_name" --tags
    echo "Created a new tag, $version"
fi