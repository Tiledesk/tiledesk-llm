# npm version patch
version="0.1.18"
echo "version $version"

if [ "$version" != "" ]; then
    git add .
    git commit -m "Created a new version: $version"
    git push remote master --tags
    git tag -a "$version" -m "`git log -1 --format=%s`"
    echo "Created a new tag, $version"
fi