#!/bin/sh
# Arguments name the executable files in the starter, as find -name patterns.
set -eu
find /frontier/task -type d -exec chmod 0555 {} +
find /frontier/task/bin -type f -exec chmod 0444 {} +
chmod 0555 /frontier/task/bin/*.sh
find /frontier/task/repo -type f -exec chmod 0444 {} +
for pattern do
    find /frontier/task/repo -type f -name "$pattern" -exec chmod 0555 {} +
done
chmod 0444 /frontier/task/README.md
