# syntax=docker/dockerfile:1.7
# The accepted r3 runtime supplies the pinned offline models, toolchain, source
# tree and trusted SSH bootstrap. Its original recipe is base.Dockerfile.
FROM docker.io/joyemang33/fcs2-task-sparse_video_attention_policy-agent-gpu@sha256:d014d7fbf4641b489a5fda4849d8a484d1dfa8df1c3b176386a0f5c97ea075fe

USER root
# HunyuanVideo diffusers subset (Tencent Hunyuan Community License; see
# docs/task-authoring/sparse_video_attention_policy/hunyuan/LICENSE), pinned by
# revision. model-assets.json lists every file with its size and LFS sha256.
# The pinned base predates HunyuanVideo; add the same pinned snapshot here.
RUN chmod 0755 /opt/frontier/models
RUN HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0 python3 -c 'import os; from pathlib import Path; from huggingface_hub import snapshot_download; files=['"'"'LICENSE'"'"', '"'"'Notice'"'"', '"'"'README.md'"'"', '"'"'model_index.json'"'"', '"'"'scheduler/scheduler_config.json'"'"', '"'"'text_encoder/config.json'"'"', '"'"'text_encoder/model-00001-of-00004.safetensors'"'"', '"'"'text_encoder/model-00002-of-00004.safetensors'"'"', '"'"'text_encoder/model-00003-of-00004.safetensors'"'"', '"'"'text_encoder/model-00004-of-00004.safetensors'"'"', '"'"'text_encoder/model.safetensors.index.json'"'"', '"'"'text_encoder_2/config.json'"'"', '"'"'text_encoder_2/model.safetensors'"'"', '"'"'tokenizer/special_tokens_map.json'"'"', '"'"'tokenizer/tokenizer.json'"'"', '"'"'tokenizer/tokenizer_config.json'"'"', '"'"'tokenizer_2/merges.txt'"'"', '"'"'tokenizer_2/special_tokens_map.json'"'"', '"'"'tokenizer_2/tokenizer_config.json'"'"', '"'"'tokenizer_2/vocab.json'"'"', '"'"'vae/config.json'"'"', '"'"'vae/diffusion_pytorch_model.safetensors'"'"']; target='"'"'/opt/frontier/models/HunyuanVideo'"'"'; snapshot_download(repo_id='"'"'tencent/HunyuanVideo'"'"', revision='"'"'2a15b5574ee77888e51ae6f593b2ceed8ce813e5'"'"', local_dir=target, allow_patterns=files); [os.chmod(Path(target) / name, 0o444) for name in files]'
RUN HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0 python3 -c 'import os; from pathlib import Path; from huggingface_hub import snapshot_download; files=['"'"'transformer/config.json'"'"', '"'"'transformer/diffusion_pytorch_model-00001-of-00006.safetensors'"'"', '"'"'transformer/diffusion_pytorch_model-00002-of-00006.safetensors'"'"', '"'"'transformer/diffusion_pytorch_model-00003-of-00006.safetensors'"'"', '"'"'transformer/diffusion_pytorch_model-00004-of-00006.safetensors'"'"', '"'"'transformer/diffusion_pytorch_model-00005-of-00006.safetensors'"'"', '"'"'transformer/diffusion_pytorch_model-00006-of-00006.safetensors'"'"', '"'"'transformer/diffusion_pytorch_model.safetensors.index.json'"'"']; target='"'"'/opt/frontier/models/HunyuanVideo'"'"'; snapshot_download(repo_id='"'"'tencent/HunyuanVideo'"'"', revision='"'"'2a15b5574ee77888e51ae6f593b2ceed8ce813e5'"'"', local_dir=target, allow_patterns=files); [os.chmod(Path(target) / name, 0o444) for name in files]'
RUN find /opt/frontier/models/HunyuanVideo -type d -exec chmod 0555 {} + \
    && chmod 0555 /opt/frontier/models
RUN command -v tar && command -v /opt/frontier/bin/frontier-gpu-sshd-bootstrap
COPY agent-base/frontier/task/repo /frontier/task/repo
COPY agent-gpu/task_runtime.py /opt/frontier/task-runtime
ENV CC=/usr/bin/gcc CXX=/usr/bin/g++
RUN rm -f /frontier/task/repo/make_submission.sh \
    && /opt/frontier/runtime-python /opt/frontier/task-runtime --snapshot \
    && chmod 0555 /opt/frontier/task-runtime \
    && find /frontier/task/repo -type d -exec chmod 0555 {} + \
    && find /frontier/task/repo -type f -exec chmod 0444 {} + \
    && find /frontier/task/repo -type f -name '*.sh' -exec chmod 0555 {} +

USER 10001:10001
WORKDIR /workspace
