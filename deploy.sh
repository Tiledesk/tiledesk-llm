# npm version patch
version="0.1.18-rc12"
echo "version $version"

if [ "$version" != "" ]; then
    git add .
    git commit -m "Created a new version"
    git tag -a "$version" -m "`git log -1 --format=%s`"
    echo "Created a new tag, $version"
    git push remote --tags
fi