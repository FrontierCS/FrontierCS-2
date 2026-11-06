# Shared GPU tools (#757): files every GPU task image copies, not a base to build FROM (each
# GPU task keeps its own CUDA/PyTorch stack). A GPU task's Dockerfile takes this image by build
# argument (the build injects tasks/shared/bases.json) and copies what it needs:
#
#   ARG GPU_TOOLS
#   FROM ${GPU_TOOLS} AS frontier_gpu_tools
#   COPY --from=frontier_gpu_tools /opt/frontier/bin/frontier-gpu-sshd-bootstrap /opt/frontier/bin/
#
# A Verifier GPU image whose judge runs submitted code takes the isolate launcher too (the
# userns backend on the runner, landlock on an EKS control image):
#
#   COPY --from=frontier_gpu_tools /opt/frontier/shared-gpu/isolate.py /opt/frontier/
#
# and puts /opt/frontier on the default import path of each interpreter that runs judge code
# (frontier.pth in its site-packages), so the judge and every worker process it starts import
# isolate by name, with no sys.path edits:
#
#   RUN --mount=type=bind,from=frontier_gpu_tools,source=/opt/frontier/shared-gpu/install_frontier_pth.py,target=/tmp/install_frontier_pth.py \
#       /opt/frontier/runtime-python /tmp/install_frontier_pth.py /opt/frontier
#
# Context: tasks/shared plus .root/, the image tree with its modes, which
# scripts/build_base_images.py stages: the bootstrap from gpu-tools/bootstrap.json (one pinned
# layer, hash checked), verifier-gpu/fcs2_verifier_gpu.py, lib/isolate.py and
# lib/install_frontier_pth.py.
FROM scratch
COPY .root/ /
LABEL org.opencontainers.image.title="FrontierCS2 shared GPU tools"
