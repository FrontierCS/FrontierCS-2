# The source's offline runtime, which the agent's image and the verifier's start from: Debian 12
# with Python 3.10, PyTorch 2.6, Diffusers, PEFT and PaddleOCR, and SD3.5 Medium, the OCR teacher
# and PaddleOCR's models at /opt/frontier/models. The source published this image without its
# recipe; this one follows the steps that the image's history records, with every fetch pinned.

# The models, each file by its SHA-256 (the source's `asset_audit.json` lists the 34):
# - Stable Diffusion 3.5 Medium at revision `b940f670f0eda2d07fbb75229e779da1ad11eb80`, which
#   Hugging Face serves only to a signed-in account, from FrontierCS's unmodified copy of it, by
#   that repository's commit;
# - the OCR teacher, `jieliu/SD3.5M-FlowGRPO-Text` at revision
#   `1a0e6426ac8b8f53347ac66d0d544a1fc2253791`;
# - PaddleOCR's detection, recognition and classification models, the archives that PaddleOCR
#   publishes, from FrontierCS's unmodified copy of them, by that repository's commit.
FROM scratch AS models
ADD --checksum=sha256:9b60b9d0411f54d4087484ad4acf28ca6fd9f6a84c30ddacb04de2c623254749 https://huggingface.co/datasets/FrontierCS2/FrontierCS-2-assets/resolve/69cb388c97f9ffbd015fc932f8283a529581c618/stable-diffusion-3.5-medium/LICENSE.md /models/stable-diffusion-3.5-medium/LICENSE.md
ADD --checksum=sha256:076c5b3ce1cb9bd33751656cac2b094b6ab93212a82624dab5405f4596737f7a https://huggingface.co/datasets/FrontierCS2/FrontierCS-2-assets/resolve/69cb388c97f9ffbd015fc932f8283a529581c618/stable-diffusion-3.5-medium/README.md /models/stable-diffusion-3.5-medium/README.md
ADD --checksum=sha256:b5e97ca0d3c0cd9a1e3ed91a71bbdd489de7941b23fe038b3aaa9acf36fd1543 https://huggingface.co/datasets/FrontierCS2/FrontierCS-2-assets/resolve/69cb388c97f9ffbd015fc932f8283a529581c618/stable-diffusion-3.5-medium/model_index.json /models/stable-diffusion-3.5-medium/model_index.json
ADD --checksum=sha256:677e1ed36ca0de4ac8cc766ca58869e8649673d5f1d47eb8f8aef511a18ae8f7 https://huggingface.co/datasets/FrontierCS2/FrontierCS-2-assets/resolve/69cb388c97f9ffbd015fc932f8283a529581c618/stable-diffusion-3.5-medium/scheduler/scheduler_config.json /models/stable-diffusion-3.5-medium/scheduler/scheduler_config.json
ADD --checksum=sha256:3ae637e6409ed10c125e13b935461060d629ee069eea7cb8c3e9eee26b01ff1a https://huggingface.co/datasets/FrontierCS2/FrontierCS-2-assets/resolve/69cb388c97f9ffbd015fc932f8283a529581c618/stable-diffusion-3.5-medium/text_encoder/config.json /models/stable-diffusion-3.5-medium/text_encoder/config.json
ADD --checksum=sha256:71e183d11db0c6b6282a4d9e0abb74125edc8692393e89ed8ee5571005f35cb1 https://huggingface.co/datasets/FrontierCS2/FrontierCS-2-assets/resolve/69cb388c97f9ffbd015fc932f8283a529581c618/stable-diffusion-3.5-medium/text_encoder/model.safetensors /models/stable-diffusion-3.5-medium/text_encoder/model.safetensors
ADD --checksum=sha256:786ac791f8ae2168eca583978d89e97b704579e4873a2ca36e2d8b5af561f625 https://huggingface.co/datasets/FrontierCS2/FrontierCS-2-assets/resolve/69cb388c97f9ffbd015fc932f8283a529581c618/stable-diffusion-3.5-medium/text_encoder_2/config.json /models/stable-diffusion-3.5-medium/text_encoder_2/config.json
ADD --checksum=sha256:ec310df2af79c318e24d20511b601a591ca8cd4f1fce1d8dff822a356bcdb1f4 https://huggingface.co/datasets/FrontierCS2/FrontierCS-2-assets/resolve/69cb388c97f9ffbd015fc932f8283a529581c618/stable-diffusion-3.5-medium/text_encoder_2/model.safetensors /models/stable-diffusion-3.5-medium/text_encoder_2/model.safetensors
ADD --checksum=sha256:9035948b0904c536918a88b5d984a52cfca26a5c57efa9e90149790cf38c151e https://huggingface.co/datasets/FrontierCS2/FrontierCS-2-assets/resolve/69cb388c97f9ffbd015fc932f8283a529581c618/stable-diffusion-3.5-medium/text_encoder_3/config.json /models/stable-diffusion-3.5-medium/text_encoder_3/config.json
ADD --checksum=sha256:4f2751ceeb2a96edd693e539dc5d6bba0b8d3814f49a9b3798403a0cec4b2e3d https://huggingface.co/datasets/FrontierCS2/FrontierCS-2-assets/resolve/69cb388c97f9ffbd015fc932f8283a529581c618/stable-diffusion-3.5-medium/text_encoder_3/model-00001-of-00002.safetensors /models/stable-diffusion-3.5-medium/text_encoder_3/model-00001-of-00002.safetensors
ADD --checksum=sha256:f63154532130422309532ff56f11945fbea8266c958e3133e8e5aef85c6293c7 https://huggingface.co/datasets/FrontierCS2/FrontierCS-2-assets/resolve/69cb388c97f9ffbd015fc932f8283a529581c618/stable-diffusion-3.5-medium/text_encoder_3/model-00002-of-00002.safetensors /models/stable-diffusion-3.5-medium/text_encoder_3/model-00002-of-00002.safetensors
ADD --checksum=sha256:3bacec0f0cf392399d4a385908f67dd73df99c9e9cfee669f148858ba9fbdb0a https://huggingface.co/datasets/FrontierCS2/FrontierCS-2-assets/resolve/69cb388c97f9ffbd015fc932f8283a529581c618/stable-diffusion-3.5-medium/text_encoder_3/model.safetensors.index.json /models/stable-diffusion-3.5-medium/text_encoder_3/model.safetensors.index.json
ADD --checksum=sha256:9fd691f7c8039210e0fced15865466c65820d09b63988b0174bfe25de299051a https://huggingface.co/datasets/FrontierCS2/FrontierCS-2-assets/resolve/69cb388c97f9ffbd015fc932f8283a529581c618/stable-diffusion-3.5-medium/tokenizer/merges.txt /models/stable-diffusion-3.5-medium/tokenizer/merges.txt
ADD --checksum=sha256:2cdb3b8331a60c92fc1e55a13e9fd61fd2293c5a51275fdcccd62b780052530e https://huggingface.co/datasets/FrontierCS2/FrontierCS-2-assets/resolve/69cb388c97f9ffbd015fc932f8283a529581c618/stable-diffusion-3.5-medium/tokenizer/special_tokens_map.json /models/stable-diffusion-3.5-medium/tokenizer/special_tokens_map.json
ADD --checksum=sha256:6bdcee9ccce2a16ca2b4c0c5ed00b42c50ea225f4472a8c4c1e963a2902c2881 https://huggingface.co/datasets/FrontierCS2/FrontierCS-2-assets/resolve/69cb388c97f9ffbd015fc932f8283a529581c618/stable-diffusion-3.5-medium/tokenizer/tokenizer_config.json /models/stable-diffusion-3.5-medium/tokenizer/tokenizer_config.json
ADD --checksum=sha256:e089ad92ba36837a0d31433e555c8f45fe601ab5c221d4f607ded32d9f7a4349 https://huggingface.co/datasets/FrontierCS2/FrontierCS-2-assets/resolve/69cb388c97f9ffbd015fc932f8283a529581c618/stable-diffusion-3.5-medium/tokenizer/vocab.json /models/stable-diffusion-3.5-medium/tokenizer/vocab.json
ADD --checksum=sha256:9fd691f7c8039210e0fced15865466c65820d09b63988b0174bfe25de299051a https://huggingface.co/datasets/FrontierCS2/FrontierCS-2-assets/resolve/69cb388c97f9ffbd015fc932f8283a529581c618/stable-diffusion-3.5-medium/tokenizer_2/merges.txt /models/stable-diffusion-3.5-medium/tokenizer_2/merges.txt
ADD --checksum=sha256:c4dbb96da703fb38f10ccf0490df2fd476811c5a3e71b7e0189cffeed3224e25 https://huggingface.co/datasets/FrontierCS2/FrontierCS-2-assets/resolve/69cb388c97f9ffbd015fc932f8283a529581c618/stable-diffusion-3.5-medium/tokenizer_2/special_tokens_map.json /models/stable-diffusion-3.5-medium/tokenizer_2/special_tokens_map.json
ADD --checksum=sha256:e70eb3e2b85a508e32bb81c32c13fc3f2b015548ca469b94627c2f32e445cba1 https://huggingface.co/datasets/FrontierCS2/FrontierCS-2-assets/resolve/69cb388c97f9ffbd015fc932f8283a529581c618/stable-diffusion-3.5-medium/tokenizer_2/tokenizer_config.json /models/stable-diffusion-3.5-medium/tokenizer_2/tokenizer_config.json
ADD --checksum=sha256:e089ad92ba36837a0d31433e555c8f45fe601ab5c221d4f607ded32d9f7a4349 https://huggingface.co/datasets/FrontierCS2/FrontierCS-2-assets/resolve/69cb388c97f9ffbd015fc932f8283a529581c618/stable-diffusion-3.5-medium/tokenizer_2/vocab.json /models/stable-diffusion-3.5-medium/tokenizer_2/vocab.json
ADD --checksum=sha256:7a1985a994c41886db38c719d2a3d2f40606663cc19d7c5d6a85d349320e06d2 https://huggingface.co/datasets/FrontierCS2/FrontierCS-2-assets/resolve/69cb388c97f9ffbd015fc932f8283a529581c618/stable-diffusion-3.5-medium/tokenizer_3/special_tokens_map.json /models/stable-diffusion-3.5-medium/tokenizer_3/special_tokens_map.json
ADD --checksum=sha256:d60acb128cf7b7f2536e8f38a5b18a05535c9e14c7a355904270e15b0945ea86 https://huggingface.co/datasets/FrontierCS2/FrontierCS-2-assets/resolve/69cb388c97f9ffbd015fc932f8283a529581c618/stable-diffusion-3.5-medium/tokenizer_3/spiece.model /models/stable-diffusion-3.5-medium/tokenizer_3/spiece.model
ADD --checksum=sha256:652ffdfc379606bad8edfb653f92dcf28e2e5dbf1cdfe50d685d29b2cad12dd6 https://huggingface.co/datasets/FrontierCS2/FrontierCS-2-assets/resolve/69cb388c97f9ffbd015fc932f8283a529581c618/stable-diffusion-3.5-medium/tokenizer_3/tokenizer.json /models/stable-diffusion-3.5-medium/tokenizer_3/tokenizer.json
ADD --checksum=sha256:83b3e7737eef754efca607d9f580798cc35457daa96052cc60179f095a375c14 https://huggingface.co/datasets/FrontierCS2/FrontierCS-2-assets/resolve/69cb388c97f9ffbd015fc932f8283a529581c618/stable-diffusion-3.5-medium/tokenizer_3/tokenizer_config.json /models/stable-diffusion-3.5-medium/tokenizer_3/tokenizer_config.json
ADD --checksum=sha256:b6ea8128277df286573ce1937c36ce42a5a84e976331091ffccb1d05893dc299 https://huggingface.co/datasets/FrontierCS2/FrontierCS-2-assets/resolve/69cb388c97f9ffbd015fc932f8283a529581c618/stable-diffusion-3.5-medium/transformer/config.json /models/stable-diffusion-3.5-medium/transformer/config.json
ADD --checksum=sha256:f81eb93a348be218af757dd095cbda45090380d9bbda91049442e0e40bb4cc95 https://huggingface.co/datasets/FrontierCS2/FrontierCS-2-assets/resolve/69cb388c97f9ffbd015fc932f8283a529581c618/stable-diffusion-3.5-medium/transformer/diffusion_pytorch_model.safetensors /models/stable-diffusion-3.5-medium/transformer/diffusion_pytorch_model.safetensors
ADD --checksum=sha256:58557f2439dfa867450caef425b5d11160be8aa9c34d60dbf23a94a6a94cb060 https://huggingface.co/datasets/FrontierCS2/FrontierCS-2-assets/resolve/69cb388c97f9ffbd015fc932f8283a529581c618/stable-diffusion-3.5-medium/vae/config.json /models/stable-diffusion-3.5-medium/vae/config.json
ADD --checksum=sha256:8f53304a79335b55e13ec50f63e5157fee4deb2f30d5fae0654e2b2653c109dc https://huggingface.co/datasets/FrontierCS2/FrontierCS-2-assets/resolve/69cb388c97f9ffbd015fc932f8283a529581c618/stable-diffusion-3.5-medium/vae/diffusion_pytorch_model.safetensors /models/stable-diffusion-3.5-medium/vae/diffusion_pytorch_model.safetensors
ADD --checksum=sha256:e57cc6e82babc575a586e5cb5a2c777f7199d61b908d3bb2425269f3ed303cf1 https://huggingface.co/jieliu/SD3.5M-FlowGRPO-Text/resolve/1a0e6426ac8b8f53347ac66d0d544a1fc2253791/README.md /models/SD3.5M-FlowGRPO-Text/README.md
ADD --checksum=sha256:3498723f85aa83d755415ff899840ed12708835222e5bea3bc3f320f35ced3b3 https://huggingface.co/jieliu/SD3.5M-FlowGRPO-Text/resolve/1a0e6426ac8b8f53347ac66d0d544a1fc2253791/adapter_config.json /models/SD3.5M-FlowGRPO-Text/adapter_config.json
ADD --checksum=sha256:a82776def305a94d7caa3f7164852bab70c749a5a3ec6de3ab5ee7b5e2d9427e https://huggingface.co/jieliu/SD3.5M-FlowGRPO-Text/resolve/1a0e6426ac8b8f53347ac66d0d544a1fc2253791/adapter_model.safetensors /models/SD3.5M-FlowGRPO-Text/adapter_model.safetensors
ADD --checksum=sha256:e740eb522521827cc83da3ba33e1c70ccfd9b3cf7d9a24a420ce4ca5ebb8794f https://huggingface.co/datasets/FrontierCS2/FrontierCS-2-assets/resolve/5eb62383ed966908fe33b646d1319f6d07492257/paddleocr/en_PP-OCRv3_det_infer.tar /models/ocr/en_PP-OCRv3_det_infer.tar
ADD --checksum=sha256:d5ae153a6fbdacd5776543d3194c4258154ed6f83467d55de1cc345f7dd20322 https://huggingface.co/datasets/FrontierCS2/FrontierCS-2-assets/resolve/5eb62383ed966908fe33b646d1319f6d07492257/paddleocr/en_PP-OCRv4_rec_infer.tar /models/ocr/en_PP-OCRv4_rec_infer.tar
ADD --checksum=sha256:507352585040d035da3b1e6374694ad679a850acb0a36a8d0d47984176357717 https://huggingface.co/datasets/FrontierCS2/FrontierCS-2-assets/resolve/5eb62383ed966908fe33b646d1319f6d07492257/paddleocr/ch_ppocr_mobile_v2.0_cls_infer.tar /models/ocr/ch_ppocr_mobile_v2.0_cls_infer.tar

FROM python:3.10.21-slim-bookworm@sha256:2559be987fd64d61badbdafd303ea58a9ccab36d6c3c08bce219e762177d2eca

ARG DEBIAN_FRONTEND=noninteractive
# The source's packages, from one dated snapshot of Debian's archive, which replaces the base's
# archive: the source installed them from the archive as it was that day.
RUN --mount=type=bind,source=fetch/apt_install.sh,target=/context/fetch/apt_install.sh \
    rm --force /etc/apt/sources.list /etc/apt/sources.list.d/* \
    && printf 'Types: deb\nURIs: https://snapshot.debian.org/archive/debian/%s\nSuites: bookworm bookworm-updates\nComponents: main\nSigned-By: /usr/share/keyrings/debian-archive-keyring.gpg\n\nTypes: deb\nURIs: https://snapshot.debian.org/archive/debian-security/%s\nSuites: bookworm-security\nComponents: main\nSigned-By: /usr/share/keyrings/debian-archive-keyring.gpg\n' \
       20260922T180000Z 20260922T180000Z > /etc/apt/sources.list.d/debian.sources \
    && printf 'Acquire::Check-Valid-Until "false";\nAcquire::Retries "10";\nAcquire::Retries::Delay::Maximum "60";\n' > /etc/apt/apt.conf.d/80-frontier-snapshot \
    && sh /context/fetch/apt_install.sh bash git tar ca-certificates libgl1 libglib2.0-0 libgomp1 libsm6 libxext6 libxrender1

# The source's Python packages. `requirements.txt` is the source's list, which names 20 packages
# and left the rest to pip; `pins/requirements.in` is every package the published image holds, at
# its version there, and `pins/requirements.txt` is `uv pip compile --no-deps --generate-hashes` of
# it, which `pip` installs under `--require-hashes`.
COPY runtime/requirements.txt /opt/flow-opd/requirements.txt
RUN --mount=type=bind,source=pins/requirements.txt,target=/context/pins/requirements.txt \
    python -m pip install --no-cache-dir --require-hashes --no-deps --requirement /context/pins/requirements.txt

# The Stability AI Community License and the attribution that it requires of whoever distributes
# the model, and the models, read-only.
ADD --checksum=sha256:9b60b9d0411f54d4087484ad4acf28ca6fd9f6a84c30ddacb04de2c623254749 https://huggingface.co/datasets/FrontierCS2/FrontierCS-2-assets/resolve/69cb388c97f9ffbd015fc932f8283a529581c618/stable-diffusion-3.5-medium/LICENSE.md /opt/flow-opd/STABILITY-AI-LICENSE.md
COPY runtime/NOTICE /opt/flow-opd/NOTICE
COPY --from=models /models /opt/frontier/models
COPY runtime/integrity-manifest.json runtime/integrity-progress.json /opt/frontier/models/
# The source's check of every model file against the manifest, then PaddleOCR's three models
# unpacked beside their archives, where the OCR scorer reads them.
RUN --mount=type=bind,source=runtime/check_models.py,target=/context/check_models.py \
    python /context/check_models.py /opt/frontier/models
RUN find /opt/frontier/models -type d -exec chmod 0555 {} + \
    && find /opt/frontier/models -type f -exec chmod 0444 {} + \
    && chmod 0555 /opt/frontier \
    && chmod --recursive a+rX /opt/flow-opd
ENV FRONTIER_MODEL_ROOT=/opt/frontier/models HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=disabled PYTHONDONTWRITEBYTECODE=1
RUN python -c "import torch, torchvision, diffusers, transformers, accelerate, peft, paddleocr, paddle; print(torch.__version__)"
# The packages of the source's GPU runner's session, which the published image holds too.
RUN --mount=type=bind,source=fetch/apt_install.sh,target=/context/fetch/apt_install.sh \
    sh /context/fetch/apt_install.sh openssh-server libnss-wrapper procps
