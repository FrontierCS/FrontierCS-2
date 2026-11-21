# The source's Verifier GPU runner image: the fixed offline
# model runtime (llama.cpp built for CUDA, Qwen3-8B Q4_K_M), ReDoSHunter's Temurin 25 JRE, the
# judge's Python closure, the pristine starter and this task's judge. Its base is the source's
# CUDA 12.8.1 development image, by the same digest, whose libraries llama-server loads.
#
# The JRE is that of Docker Hub's `eclipse-temurin` 25 image, by the digest the source's recipe
# pinned before it copied the JRE from its own published image.
FROM eclipse-temurin@sha256:dbb43d81099eec91e0800226ed2296d03146ee7857165b95112372ad0d67e87f AS java25

# llama.cpp b10868, built from its commit as the source's recipe built it before it copied the
# build from its own published image: on the same CUDA 12.8.1 development base, with upstream's
# `.devops/cuda.Dockerfile` CMake configuration (dynamic CPU backend variants, the CUDA backend)
# and the CUDA architecture limited to sm_80, the A100's. The build tools are from the dated
# Ubuntu archive snapshot that the image's other packages are from.
FROM scratch AS llama-source
ADD https://github.com/ggml-org/llama.cpp.git#304665fe7ac957df95e3ff8c8c4ffdf92dd6ffa3 /llama.cpp

FROM nvidia/cuda@sha256:4b9ed5fa8361736996499f64ecebf25d4ec37ff56e4d11323ccde10aa36e0c43 AS llama-build
ARG DEBIAN_FRONTEND=noninteractive
RUN --mount=type=bind,source=fetch/apt_snapshot.sh,target=/context/fetch/apt_snapshot.sh \
    --mount=type=bind,source=fetch/apt_install.sh,target=/context/fetch/apt_install.sh \
    sh /context/fetch/apt_snapshot.sh noble 20261103T000000Z \
    && sh /context/fetch/apt_install.sh build-essential ca-certificates cmake git
COPY --from=llama-source /llama.cpp /src/llama.cpp
RUN cd /src/llama.cpp \
    && cmake -B build -DCMAKE_BUILD_TYPE=Release \
       -DGGML_NATIVE=OFF -DGGML_CUDA=ON -DGGML_BACKEND_DL=ON -DGGML_CPU_ALL_VARIANTS=ON \
       -DCMAKE_CUDA_ARCHITECTURES=80 -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF \
       -DLLAMA_BUILD_UI=OFF -DLLAMA_USE_PREBUILT_UI=OFF -DLLAMA_OPENSSL=OFF \
       -DCMAKE_EXE_LINKER_FLAGS=-Wl,--allow-shlib-undefined \
    && cmake --build build --config Release --target llama-server -j"$(nproc)" \
    && install -d /app \
    && find build -name '*.so*' -exec cp -P {} /app \; \
    && install -m 0555 build/bin/llama-server /app/llama-server

# The fixed model, Qwen/Qwen3-8B-GGUF at `7c41481f57cb95916b40956ab2f0b139b296d974`
# (Apache-2.0), fetched by the URL the previous release's recipe used and by the full SHA-256 of
# its Git LFS object.
FROM scratch AS model
ADD --checksum=sha256:d98cdcbd03e17ce47681435b5150e34c1417f50b5c0019dd560e4882c5745785 --chmod=0444 https://huggingface.co/Qwen/Qwen3-8B-GGUF/resolve/7c41481f57cb95916b40956ab2f0b139b296d974/Qwen3-8B-Q4_K_M.gguf /Qwen3-8B-Q4_K_M.gguf

FROM nvidia/cuda@sha256:4b9ed5fa8361736996499f64ecebf25d4ec37ff56e4d11323ccde10aa36e0c43 AS common

USER root
ARG DEBIAN_FRONTEND=noninteractive
# The source's packages, less the SSH ones and `libnss-wrapper`, which only its GPU
# runner's session uses, from one dated Ubuntu archive snapshot (`fetch/apt_snapshot.sh`) that
# replaces the base's archive and NVIDIA's CUDA repository. Then the judge's Python closure in a
# virtual environment, as the source builds it: the build tools (`pins/build-requirements.txt`) and
# the packages (`pins/requirements.txt`), every file pinned by SHA-256 and installed with
# `--require-hashes --no-deps`, the sdist of `timeout-decorator` built with those tools and not in
# an isolated environment.
COPY pins/build-requirements.txt pins/requirements.txt /tmp/
RUN --mount=type=bind,source=fetch/apt_snapshot.sh,target=/context/fetch/apt_snapshot.sh \
    --mount=type=bind,source=fetch/apt_install.sh,target=/context/fetch/apt_install.sh \
    sh /context/fetch/apt_snapshot.sh noble 20261103T000000Z \
    && sh /context/fetch/apt_install.sh bash ca-certificates git libgomp1 python3 python3-venv tar \
    && /usr/bin/python3 -c 'import sys; assert sys.version_info[:2] == (3, 12), sys.version' \
    && test -x /usr/bin/unshare \
    && /usr/bin/python3 -m venv /opt/frontier/python \
    && /opt/frontier/python/bin/python -m pip --isolated --disable-pip-version-check install \
       --no-cache-dir --no-deps --require-hashes --only-binary=:all: -r /tmp/build-requirements.txt \
    && /opt/frontier/python/bin/python -m pip --isolated --disable-pip-version-check install \
       --no-cache-dir --no-deps --require-hashes --no-build-isolation -r /tmp/requirements.txt \
    && rm /tmp/build-requirements.txt /tmp/requirements.txt \
    && /opt/frontier/python/bin/python -c 'import yaml, editdistance, exrex, regex, timeout_decorator, tqdm, xeger'
# `runtime-python` is a wrapper that puts the JRE on the path, as the source's is.
# The user 65532 with /frontier its home, as the shared verifier base has them; and /workspace,
# the judge's scratch parent, which the source's `frontier` account owns.
RUN printf '%s\n' '#!/bin/sh' 'export PATH="/opt/frontier/python/bin:/opt/java/openjdk/bin:$PATH"' \
      'exec /opt/frontier/python/bin/python3 "$@"' > /opt/frontier/runtime-python \
    && chmod 0555 /opt/frontier/runtime-python \
    && install --directory --owner=65532 --group=65532 /frontier /workspace \
    && useradd --uid 65532 --user-group --no-create-home --home-dir /frontier --shell /bin/sh frontier-verifier

# llama.cpp b10868 (commit 304665fe, built for sm_80) and the Temurin 25.0.4.1+1 JRE, each in a
# layer of its own that depends on its stage's bytes alone.
COPY --link --chmod=a+rX,a-w --from=java25 /opt/java/openjdk /opt/java/openjdk
COPY --link --chmod=a+rX,a-w --from=llama-build /app/ /app/
# The fixed model, from the stage `model` (below), in a layer of its own that depends on the
# file's bytes alone. The copy creates /models with the default mode: an `ADD --chmod` straight
# into /models would make the directory itself unreadable to the judge's user.
COPY --link --from=model /Qwen3-8B-Q4_K_M.gguf /models/Qwen3-8B-Q4_K_M.gguf

ENV JAVA_HOME=/opt/java/openjdk LD_LIBRARY_PATH=/app:/usr/local/cuda/lib64 \
    PATH=/opt/frontier/python/bin:/opt/java/openjdk/bin:/usr/local/nvidia/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
# libcuda.so.1 comes from the GPU host's driver at run time, so the server cannot start during
# the build; check that every other shared library resolves, the JRE, and the model's
# size.
RUN java -version 2>&1 | grep -q '"25' && ln -s /opt/java/openjdk/bin/java /usr/local/bin/java \
    && test -x /app/llama-server \
    && test -f /app/libggml-cuda.so \
    && test "$(stat --format=%s /models/Qwen3-8B-Q4_K_M.gguf)" = 5027783488 \
    && ! for f in /app/llama-server /app/*.so; do ldd "$f"; done | grep 'not found' | grep -v 'libcuda\.so'

LABEL org.opencontainers.image.title="FrontierCS2 verifier lrr" \
      dev.frontier.problem-id="lrr" \
      dev.frontier.generator="frontiersmith2"

# The verifier writes its result and evidence below /frontier/output, which is Pumpkin's
# /pumpkin/out, and its scratch below /frontier, on the container's writable layer.
RUN ln --symbolic /pumpkin/out /frontier/output

# /opt/frontier goes on the interpreter's import path (frontier.pth), so the judge and every
# worker it starts import isolate by name; the launcher is the source's.
COPY --from=task-tools /verifier/*.py /lib/edit_scope.py /lib/isolate.py \
    /lib/install_frontier_pth.py /opt/frontier/
# The starter that the shared verifier overlays the submission on, and the judge applies its
# reference patch to (its shell scripts executable, as the source makes them); the
# generated files; and each mode's selector.
COPY --chmod=a+rX,a-w repo /opt/frontier/task/pristine
COPY --chmod=a+rX,a-w generated/judge/config.yaml /opt/frontier/task/config.yaml
COPY generated/verifier/verifier.json /opt/frontier/verifier.json
COPY --chmod=a+rX,a-w datasets/train.json datasets/eval.json /opt/frontier/datasets/
RUN /opt/frontier/runtime-python /opt/frontier/install_frontier_pth.py /opt/frontier \
    && /opt/frontier/runtime-python -c 'import zstandard, yaml' \
    && printf '#!/bin/sh\nexec /opt/frontier/runtime-python /opt/frontier/verify.py "$@"\n' > /opt/frontier/verifier \
    && chmod 0444 /opt/frontier/verifier.json /opt/frontier/*.py \
    && chmod 0555 /opt/frontier /opt/frontier/verifier \
    && find /opt/frontier/task/pristine -type f -name '*.sh' -exec chmod 0555 {} + \
    && chmod a-w /opt/frontier/task /opt/frontier/task/pristine /opt/frontier/datasets

ENV HOME=/frontier PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /frontier

# The held-out strata of the final suite are for the final evaluation alone. `train/trim.py`
# removes the final suite from a copy of `workloads.json` in a stage of its own, from which the
# train judge's image copies the judge, so that the held-out regexes are in none of its layers.
FROM common AS judge-train
COPY judge /judge
COPY train/trim.py /trim.py
RUN /opt/frontier/runtime-python /trim.py /judge \
    && echo "ec185093e1343b9d9d67f91a1dacf9d6e0143ba84c36b0de3d99bcd32592f8fd  /judge/workloads.json" | sha256sum --check --strict - \
    && ! grep --quiet heldout /judge/workloads.json

# The train judge's image: the judge with the final suite left out.
FROM common AS train
COPY --chmod=a+rX,a-w --from=judge-train /judge /opt/frontier/task
USER 65532:65532

# The final judge's image: the judge as it is.
FROM common AS eval
COPY --chmod=a+rX,a-w judge /opt/frontier/task
USER 65532:65532
