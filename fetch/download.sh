#!/bin/sh
# Downloads the files a pins file names, each checked against its SHA-256, so that a build which
# would fetch them itself finds them in place, and gets them only if they are these bytes.
#
#     download.sh PINS DIRECTORY
#
# PINS has a line `<sha256> <url> <path>` for each file, which is saved at DIRECTORY/<path>.
set -eu
pins=$1
directory=$2
while read -r sha256 url path; do
    mkdir -p "$directory/$(dirname "$path")"
    curl --fail --silent --show-error --location --retry 5 --output "$directory/$path" "$url"
    echo "$sha256  $directory/$path" | sha256sum --check --strict --quiet -
done < "$pins"
