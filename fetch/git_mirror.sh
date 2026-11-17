#!/bin/sh
# Makes the commits a pins file names the only ones that `git clone https://github.com/...` can
# get, so that a build which clones a repository by a tag gets the commit pinned for it.
#
#     git_mirror.sh PINS MIRROR_DIRECTORY
#
# PINS has a line `<owner>/<repository> <tag> <commit>` for each repository. Each commit is
# fetched from GitHub, which names it by its full ID, into a bare repository at
# MIRROR_DIRECTORY/<owner>/<repository> with the tag on it, and git is then told, system-wide,
# to read `https://github.com/` from MIRROR_DIRECTORY.
set -eu
pins=$1
mirror=$2
while read -r repository tag commit; do
    git init --quiet --bare "$mirror/$repository"
    git -C "$mirror/$repository" fetch --quiet --depth 1 "https://github.com/$repository" "$commit"
    test "$(git -C "$mirror/$repository" rev-parse FETCH_HEAD)" = "$commit"
    git -C "$mirror/$repository" tag "$tag" "$commit"
done < "$pins"
git config --system "url.file://$mirror/.insteadOf" https://github.com/
