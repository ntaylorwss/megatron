#!/bin/bash

# docker hub info
username=ntaylor22
imagename=megatron

# ensure up to date
version=$(<VERSION)
if ! git diff-index --quiet HEAD -- # true means local changes
then
    echo "You have local changes not on remote. Please commit and push before bumping."
    exit
fi

# update sphinx docs
echo "Rebuilding docs..."
docker/start
docker exec megatron bash -c "sphinx-apidoc -f -o docs/source megatron && sphinx-build -b html docs/source docs/build"
docker/stop
git add docs
git commit -m "Documentation update in preparation for version bump"
git push origin master
echo "Docs rebuilt."

# rebuild images and bump docker versions
echo "Building docker images..."
docker/build --no-cache
echo "Docker images built."

# push images to hub
echo "Pushing docker images..."
echo "Password hint:"
cat creds/docker_hint
docker login -u ntaylor22
docker push ntaylor22/megatron:$new_version &
wait
docker push ntaylor22/megatron &
wait
echo "Docker images pushed."

# get new version
echo "Current version number: $version"
read -p "New version number: " new_version
sleep 1

# change version in setup.py
sed -i "s/$version/$new_version/g" setup.py

# bump version, push
echo "Committing version change..."
echo $new_version > VERSION
git add VERSION
git commit -m "version $new_version"
git tag -a "$new_version" -m "version $new_version"
git push origin master
git push origin master --tags
echo "Version change committed."

# re-run setup and push to pypi
python3 setup.py sdist
echo "Pushing to pypi..."
echo "Username is ntaylorwss"
echo "Password hint:"
cat creds/pypi_hint
twine upload dist/*
echo "Pushed to pypi."
echo "Successful release!"
