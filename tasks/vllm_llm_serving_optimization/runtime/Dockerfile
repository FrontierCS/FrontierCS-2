# The task's H100 runtime, which the judges' parent (`parent/Dockerfile`) and the agent's GPU runner
# (`gpu/Dockerfile`) start from: CUDA 12.9, Python 3.12.11, the vLLM 0.11.0 wheel with every package
# it ran with, vLLM's clean source at /opt/vllm-clean, and the launch helper /opt/frontier/vllm-runtime
# with /opt/frontier/task-runtime. The Qwen3-Coder-30B-A3B-Instruct model is not in the image: each
# runner holds its files read-only at /opt/frontier/model, as `images.star`'s assets. The task's
# baselines were measured on an image that the source published without its recipe. This recipe
# follows the steps that the image's history records, with every fetch pinned, and leaves out the
# source's SSH bootstrap and the `openssh` packages, which only that bootstrap used.

FROM docker.io/astral/uv:0.8.22@sha256:ae1ba4db410fbe1d9eaf16c51add8e7608b271ebbb69bd58a56e80384fd190aa AS uv

FROM docker.io/nvidia/cuda:12.9.0-devel-ubuntu22.04@sha256:0a254a86e28379f7a761c73caf4874247d5e3fbcf57bd99a44856ccf9098e092
USER root
ARG DEBIAN_FRONTEND=noninteractive
# The source's packages but `openssh-client` and `openssh-server`, at the versions of its image,
# from Ubuntu's archive as it was on the day the image was built.
RUN --mount=type=bind,source=fetch/apt_snapshot.sh,target=/context/fetch/apt_snapshot.sh \
    --mount=type=bind,source=fetch/apt_install.sh,target=/context/fetch/apt_install.sh \
    sh /context/fetch/apt_snapshot.sh jammy 20260914T000000Z \
    && sh /context/fetch/apt_install.sh bash build-essential ca-certificates=20260601~22.04.1 \
       git=1:2.34.1-1ubuntu1.17 tar=1.34+dfsg-1ubuntu0.1.22.04.6 libnuma1=2.0.14-3ubuntu2 \
    && useradd --create-home --uid 10001 --user-group --shell /bin/bash frontier \
    && usermod --password x frontier \
    && install -d -o 10001 -g 10001 /workspace \
    && install -d /opt/frontier/bin

COPY --from=uv /uv /opt/frontier/bin/uv
ENV UV_PYTHON_INSTALL_DIR=/opt/python UV_LINK_MODE=copy
RUN /opt/frontier/bin/uv python install 3.12.11 \
    && ln -s "$(/opt/frontier/bin/uv python find --managed-python 3.12.11)" /opt/frontier/runtime-python \
    && /opt/frontier/runtime-python -c 'import sys; assert sys.version_info[:3] == (3, 12, 11)'

# vLLM's source at the commit of its 0.11.0 release, a clone of that commit alone, as the source's
# was. The launch helper refuses any other commit, and clones it for each variant.
RUN git init --quiet /opt/vllm-clean \
    && git -C /opt/vllm-clean remote add origin https://github.com/vllm-project/vllm.git \
    && git -C /opt/vllm-clean fetch --quiet --depth 1 origin b8b302cde434df8c9289a2b465406b47ebab1c2d \
    && git -C /opt/vllm-clean checkout --quiet --detach FETCH_HEAD \
    && test "$(git -C /opt/vllm-clean rev-parse HEAD)" = b8b302cde434df8c9289a2b465406b47ebab1c2d
ADD --checksum=sha256:3861c75ff2b12e24f6d179ff5c084d791b42ded8675d76c8706697c79f68cd62 --chmod=0444 https://files.pythonhosted.org/packages/47/33/d19e0763c34392ec956534536fa837c060495bfff31ed83452135ea7608d/vllm-0.11.0-cp38-abi3-manylinux1_x86_64.whl /opt/frontier/wheels/
COPY vllm-assets.json /opt/frontier/vllm-assets.json
COPY vllm-runtime /opt/frontier/vllm-runtime
COPY task-runtime /opt/frontier/task-runtime

# Every package of the source's image at its version there. `python-freeze.txt` is what
# `uv pip freeze` printed in that image. `requirements.txt` is `uv pip compile --no-deps
# --generate-hashes` of it, less `pip`, which the Python build brings, and `vllm`, the wheel above.
# The set is closed: vLLM's dependencies resolve to it. The build fails unless the freeze here is
# the source's, byte for byte.
RUN --mount=type=bind,source=requirements.txt,target=/context/requirements.txt \
    --mount=type=bind,source=python-freeze.txt,target=/context/python-freeze.txt \
    /opt/frontier/runtime-python /opt/frontier/vllm-runtime verify-assets \
    && /opt/frontier/bin/uv pip install --no-cache --break-system-packages --python /opt/frontier/runtime-python \
       --require-hashes --no-deps --only-binary :all: --requirement /context/requirements.txt \
    && /opt/frontier/bin/uv pip install --no-cache --break-system-packages --python /opt/frontier/runtime-python \
       --no-deps /opt/frontier/wheels/vllm-0.11.0-cp38-abi3-manylinux1_x86_64.whl \
    && /opt/frontier/bin/uv pip freeze --python /opt/frontier/runtime-python > /opt/frontier/python-freeze.txt \
    && cmp /opt/frontier/python-freeze.txt /context/python-freeze.txt \
    && /opt/frontier/runtime-python -c 'import torch, transformers; assert torch.__version__.split("+")[0] == "2.8.0"; assert transformers.__version__ == "4.55.2"'

ENV PATH=/opt/frontier/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 DO_NOT_TRACK=1 VLLM_TARGET_DEVICE=cuda HOME=/workspace \
    TMPDIR=/workspace PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
RUN chmod 0555 /opt/frontier/wheels \
    && ln -s /opt/frontier/runtime-python /opt/frontier/bin/python3 \
    && ln -s /opt/frontier/runtime-python /opt/frontier/bin/python \
    && /opt/frontier/runtime-python /opt/frontier/task-runtime --snapshot \
    && chmod 0555 /opt/frontier/vllm-runtime /opt/frontier/task-runtime \
    && git config --system --add safe.directory /opt/vllm-clean \
    && git config --system --add safe.directory /opt/vllm-clean/.git

LABEL org.opencontainers.image.title="FrontierCS2 vLLM offline H100 runtime" \
      dev.frontier.problem-id="vllm_llm_serving_optimization" \
      dev.frontier.vllm-commit="b8b302cde434df8c9289a2b465406b47ebab1c2d"
ENTRYPOINT []
CMD []
WORKDIR /workspace
