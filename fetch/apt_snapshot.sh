#!/bin/sh
# Points apt at one dated snapshot of Ubuntu's archive, and at nothing else, for a base image that
# sets none: its own archive and any other list in it (NVIDIA's CUDA repository, which no
# snapshot covers) are removed, and a failed request is tried up to 10 times.
#
#     apt_snapshot.sh CODENAME SNAPSHOT
#
# CODENAME is the base's Ubuntu release (`jammy`, `noble`), SNAPSHOT a snapshot.ubuntu.com
# timestamp such as `20261103T000000Z`, the one FrontierCS's shared bases are set to.
set -eu
codename=$1
snapshot=$2
rm --force /etc/apt/sources.list /etc/apt/sources.list.d/*
printf 'Types: deb\nURIs: https://snapshot.ubuntu.com/ubuntu/%s\nSuites: %s %s-updates %s-backports %s-security\nComponents: main restricted universe multiverse\nSigned-By: /usr/share/keyrings/ubuntu-archive-keyring.gpg\n' \
    "$snapshot" "$codename" "$codename" "$codename" "$codename" > /etc/apt/sources.list.d/ubuntu.sources
printf 'Acquire::Retries "10";\nAcquire::Retries::Delay::Maximum "60";\n' > /etc/apt/apt.conf.d/80-frontier-retries
