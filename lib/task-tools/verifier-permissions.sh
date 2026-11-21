#!/bin/sh
set -eu
chmod 0444 /opt/frontier/verifier.json /opt/frontier/*.py
chmod 0555 /opt/frontier
chmod --recursive a+rX,a-w /opt/frontier/datasets
chmod a-w /opt/frontier/task /opt/frontier/task/pristine
