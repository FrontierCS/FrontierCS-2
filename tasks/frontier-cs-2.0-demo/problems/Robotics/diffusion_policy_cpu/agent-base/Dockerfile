# syntax=docker/dockerfile:1.7
# Task CPU base for diffusion_policy_cpu. Composed with the Candidate overlay at Run time,
# so it must not depend on its own ENTRYPOINT, CMD or named user.
#
# Build context is this task's own root directory (the one holding judge/,
# agent-base/, verifier/) — not agent-base/ itself.  Keeping every Agent-visible
# byte below agent-base/frontier/task/ gives native and generated tasks the same
# checked-in image tree:
#   docker build -f agent-base/Dockerfile -t <tag> .
FROM python@sha256:6dafe336b64dbf76f7ec872d07e58b5c7de0e2ea5d4c0853a05c3b6be3278b00

LABEL org.opencontainers.image.title="FrontierCS2 task diffusion_policy_cpu" \
      dev.frontier.problem-id="diffusion_policy_cpu" \
      dev.frontier.generator="frontiersmith2" \
      dev.frontier.generator-run="unknown"

ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates git python3 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 10001 --user-group --shell /bin/sh frontier \
    && install -d -o 10001 -g 10001 /workspace


# The solver-visible tree, and nothing else. The evaluator, the reference, the
# tests and the beat-reference specification are in the Verifier image; a copy of
# this directory that also carried them would hand the answer over.
COPY agent-base/frontier/task /frontier/task

# COPY can preserve restrictive author umasks; all public helpers need traversal/read access.
RUN chmod 0555 /frontier/task/bin/*.sh \
    && ln -s /frontier/task/bin/prepare_workspace.sh /frontier/task/prepare_workspace.sh \
    && ln -s /frontier/task/bin/make_submission.sh /frontier/task/make_submission.sh \
    && find /frontier/task -type d -exec chmod 0555 {} + \
    && find /frontier/task -type f -exec chmod 0444 {} + \
    && find /frontier/task -type f -name '*.sh' -exec chmod 0555 {} + \
    && chmod 0444 /frontier/task/README.md

USER 10001:10001
ENV HOME=/workspace TMPDIR=/workspace PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1

WORKDIR /workspace

# The judge's measurement runtime (judge/measurement-requirements.txt, the same
# pinned artifacts), so the repository runs locally; the Agent sandbox has no
# network access. Five pins are source-only on PyPI and are built here with the
# pinned setuptools/wheel.
USER 0:0
RUN <<'SH'
set -eu
cat > /tmp/measurement-requirements.txt <<'PINS'
GitPython==3.1.62 --hash=sha256:7002251225e10e29d2e1f49e6532613fe5d5d9f0b6f1f02997a52b38fe56899e
ImageIO==2.37.4 --hash=sha256:1ab2e22c8debf700f24c3ac43e8f95f3b3a8110c83b93411e97b4b0b2cd1c7e6
Jinja2==3.1.6 --hash=sha256:85ece4451f492d0c13c5dd7c13a64681a86afae63a5f347908daf103ce6d2f67
MarkupSafe==3.0.3 --hash=sha256:f42d0984e947b8adf7dd6dde396e720934d12c506ce84eea8476409563607591
PyWavelets==1.8.0 --hash=sha256:880a0197e9fa108939af50a95e97c1bf9b7d3e148e0fad92ea60a9ed8c8947c0
PyYAML==6.0.3 --hash=sha256:9c7708761fccb9397fe64bbc0395abcae8c4bf7b0eac081e12b809bf47700d0b
Shapely==1.8.4 --hash=sha256:007f0d51d045307dc3addd1c318d18f450c565c8ea96ea41304e020ca34d85b7
antlr4-python3-runtime==4.9.3 --hash=sha256:f224469b4168294902bb1efa80a8bf7855f24c99aef99cbefc1bcd3cce77881b
asciitree==0.3.3 --hash=sha256:4aa4b9b649f85e3fcb343363d97564aa1fb62e249677f2e18a96765145cc0f6e
av==12.3.0 --hash=sha256:0220fce2a62d71cc5e89617419b6224ddb43f1753b00f68b5c9af8b5f41d38c9
certifi==2026.7.22 --hash=sha256:62f22742b58a1a33014a2b6b706588a8d7e2a88ae7bd1a6ebe8c992928483775
cffi==2.1.1 --hash=sha256:194cffa889098ced9976c3fc6340305e43f6303657d298da55366907c05c22d6
charset-normalizer==3.5.1 --hash=sha256:96eefc178f8636b9c760c5829345307fd81cfae9ab1e80997dbddeb0f54ee9a3
click==8.5.0 --hash=sha256:255bc9599cf7748b4b1a446ccc735421bd08a2ae529a8b88597d3de5664ee360
cloudpickle==3.1.2 --hash=sha256:9acb47f6afd73f60dc1df93bb801b472f05ff42fa6c84167d25cb206be1fbf4a
contourpy==1.3.2 --hash=sha256:ad687a04bc802cbe8b9c399c07162a3c35e227e2daccf1668eb1f278cb698631
cycler==0.12.1 --hash=sha256:85cef7cff222d8644161529808465972e51340599459b8ac3ccbac5a854e0d30
diffusers==0.11.1 --hash=sha256:5b3f0d113d369cb675d292c3dffb35d0bac96af095727b5380ca3d48eb0e3e94
dill==0.3.5.1 --hash=sha256:33501d03270bbe410c72639b350e941882a8b0fd55357580fbc873fba0c59302
docker-pycreds==0.4.0 --hash=sha256:7266112468627868005106ec19cd0d722702d2b7d5912a28e19b826c3d37af49
einops==0.4.1 --hash=sha256:932b12bb3176caef629cc513fc8a442338fdbfe1e560794a6e7306dcee65a8af
entrypoints==0.4 --hash=sha256:f174b5ff827504fd3cd97cc3f8649f3693f51538c7e4bdf3ef002c8429d42f9f
fasteners==0.20 --hash=sha256:9422c40d1e350e4259f509fb2e608d6bc43c0136f79a00db1b49046029d0b3b7
filelock==3.32.3 --hash=sha256:7f0ca4bcc0e181c60dbbd8aa9ab5b120ebb99e4e064e83636340056f833a1f09
fonttools==4.65.0 --hash=sha256:fb53892b570f7f1f0055e75fc4de32673e32f749c4c8a606b63d5c436650e634
fsspec==2026.7.0 --hash=sha256:b57ddbafedfaef7018c1ecab32aa200a9d7ca26b77965f64e48b70061249d279
gitdb==4.0.12 --hash=sha256:67073e15955400952c6565cc3e707c554a4eea2e428946f7a4c162fab9bd9bcf
gym==0.21.0 --hash=sha256:0fd1ce165c754b4017e37a617b097c032b8c3feb8a0394ccc8777c7c50dddff3
gym-notices==0.1.0 --hash=sha256:a943af4446cb619d04fd1e470b9272b4473e08a06d1c7cc9005755a4a0b8c905
huggingface-hub==0.25.2 --hash=sha256:1897caf88ce7f97fe0110603d8f66ac264e3ba6accdf30cd66cc0fed5282ad25
hydra-core==1.3.2 --hash=sha256:fa0238a9e31df3373b35b0bfb672c34cc92718d21f81311d8996a16de1141d8b
idna==3.19 --hash=sha256:815e7be7a7806d54abb586dc943addc79e8b2ee16915059658cbeff4b1b43bf4
importlib_metadata==9.0.1 --hash=sha256:bba5600596a7e21f3eef53281cf28d6a5195634d2f2b78ff9501a3272c6eaab0
kiwisolver==1.5.1 --hash=sha256:e05c2f7925f1d88778e53cb44f14e0223204a3bdd09a41664750363acfb1f2ef
llvmlite==0.40.1 --hash=sha256:bbd5e82cc990e5a3e343a3bf855c26fdfe3bfae55225f00efd01c05bbda79918
matplotlib==3.6.1 --hash=sha256:2cc5d726d4d42865f909c5208a7841109d76584950dd0587b01a77cc279d4ab7
mpmath==1.3.0 --hash=sha256:a0b2b9fe80bbcd81a6647ff13108738cfb482d481d826cc0e02f5b35e5c88d2c
networkx==3.4.2 --hash=sha256:df5d4365b724cf81b8c6a7312509d0c22386097011ad1abe274afd5e9d3bbc5f
numba==0.57.1 --hash=sha256:c0602e4f896e6a6d844517c3ab434bc978e7698a22a733cc8124465898c28fa8
numcodecs==0.10.2 --hash=sha256:cd0850692fddcbd4f4a2b3d690b3fbd5b3adf82dcc5e72c46f89ba77cffde29d
numpy==1.23.5 --hash=sha256:5e05b1c973a9f858c74367553e236f287e749465f773328c8ef31abe18f691e1
omegaconf==2.3.1 --hash=sha256:3d701d14e9a8828f1edd28bb70b725908b34277cdd72cf7d6a83f94dadc6b6a0
opencv-python-headless==4.6.0.66 --hash=sha256:2c032c373e447c3fc2a670bca20e2918a1205a6e72854df60425fd3f82c78c32
packaging==26.3 --hash=sha256:d7193f7c8e4e93f444fde0262bf90af30e16fa0ad0ad44cb553c87339b23cd1c
pathtools==0.1.2 --hash=sha256:7c35c5421a39bb82e58018febd90e3b6e5db34c5443aaaf742b3f33d4655f1c0
pillow==12.3.0 --hash=sha256:f0606c8bf2cdefea14a43530f7657cbbb7ecf1c4222512492ef4a4434a9501ec
pip==23.0.1 --hash=sha256:236bcb61156d76c4b8a05821b988c7b8c35bf0da28a4b614e8d6ab5212c25c6f
promise==2.3 --hash=sha256:dfd18337c523ba4b6a58801c164c1904a9d4d1b1747c7d5dbf45b693a49d93d0
protobuf==3.20.3 --hash=sha256:9aae4406ea63d825636cc11ffb34ad3379335803216ee3a856787bcf5ccc751e
psutil==7.2.2 --hash=sha256:076a2d2f923fd4821644f5ba89f059523da90dc9014e85f8e45a5774ca5bc6f9
pycparser==3.0 --hash=sha256:b727414169a36b7d524c1c3e31839a521725078d7b2ff038656844266160a992
pygame==2.1.3 --hash=sha256:b75481cc17a22679c69014ef2322d55cfa66be0923abbd9206e01ef10bb5dab6
pymunk==6.2.1 --hash=sha256:73e80f0141709434c1adc3fbb8cb7dc9684dc3ad09985a1ae3e81ab5a3706b8f
pyparsing==3.3.2 --hash=sha256:850ba148bd908d7e2411587e247a1e4f0327839c40e2e5e6d05a007ecc69911d
python-dateutil==2.9.0.post0 --hash=sha256:a8b2bc7bffae282281c8140a97d3aa9c14da0b136dfe83f850eea9a5f7470427
regex==2026.9.10 --hash=sha256:c3d95d7d9538b5b726dd6fcd7b6117a71e6565202f6d64f5845fb4d8f203f533
requests==2.34.2 --hash=sha256:2a0d60c172f83ac6ab31e4554906c0f3b3588d37b5cb939b1c061f4907e278e0
scikit-image==0.19.3 --hash=sha256:cfbb073f23deb48e0e60c47f8741d8089121d89cc78629ea8c5b51096efc5be7
scipy==1.9.1 --hash=sha256:8cc81ac25659fec73599ccc52c989670e5ccd8974cf34bacd7b54a8d809aff1a
sentry-sdk==2.69.2 --hash=sha256:25bc0a6c55bba0b83398212a7247f4f9f3190f189a1909b68e4807ef4b785895
setproctitle==1.3.7 --hash=sha256:00afa6fc507967d8c9d592a887cdc6c1f5742ceac6a4354d111ca0214847732c
setuptools==65.5.0 --hash=sha256:f62ea9da9ed6289bfe868cd6845968a2c854d1427f8548d52cae02a42b4f0356
shortuuid==1.0.13 --hash=sha256:a482a497300b49b4953e15108a7913244e1bb0d41f9d332f5e9925dba33a3c5a
six==1.17.0 --hash=sha256:4721f391ed90541fddacab5acf947aa0d3dc7d27b2e1e8eda2be8970586c3274
smmap==5.0.3 --hash=sha256:c106e05d5a61449cf6ba9a1e650227ecfb141590d2a98412103ff35d89fc7b2f
sympy==1.14.0 --hash=sha256:e091cc3e99d2141a0ba2847328f5479b05d94a6635cb96148ccb3f34671bd8f5
tifffile==2025.5.10 --hash=sha256:e37147123c0542d67bc37ba5cdd67e12ea6fbe6e86c52bee037a9eb6a064e5ad
torch==2.2.2+cpu --hash=sha256:02c4fac3c964e73f5f49003e0060c697f73b67c10cc23f51c592facb29e1bd53
tqdm==4.70.1 --hash=sha256:c293e525e6fef9c20e8728fd4612df02a0aa31bb5fe91ecd93e123b1b7bffa73
typing_extensions==4.16.0 --hash=sha256:481caa481374e813c1b176ada14e97f1f67a4539ce9cfeb3f350d78d6370c2e8
urllib3==2.8.0 --hash=sha256:0cf3cae568d36aa9576b28dfb35f11328f1cb974ca7647d9475ebb86c75ac6e3
wandb==0.13.3 --hash=sha256:a246ed531ae405f2c95d299adb7311f2cf333c6c73ce36e3fcaaae60f0817c6c
wheel==0.38.4 --hash=sha256:b60533f3f5d530e971d6737ca6d58681ee434818fab630c83a734bb10c083ce8
zarr==2.12.0 --hash=sha256:3713f28f3fafa838f62bb442504da8e89bb969ecd1da09c9a96927536ec9be34
zipp==4.1.0 --hash=sha256:25ad4e16390cd314347dd8f1de67a2ac538ae658ed4ab9db16029c07c188e97f
PINS
grep -E '^(pip|setuptools|wheel)==' /tmp/measurement-requirements.txt > /tmp/bootstrap-requirements.txt
python3 -m pip install --no-cache-dir --no-deps --require-hashes -r /tmp/bootstrap-requirements.txt
python3 -m pip install --no-cache-dir --no-deps --require-hashes --no-build-isolation \
    --extra-index-url https://download.pytorch.org/whl/cpu -r /tmp/measurement-requirements.txt
python3 -m pip check
rm -f /tmp/measurement-requirements.txt /tmp/bootstrap-requirements.txt
SH
USER 10001:10001
