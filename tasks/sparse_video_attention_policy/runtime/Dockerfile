# What FrontierCS's Agent GPU runner and Verifier GPU runner of this task share: the PyTorch 2.6.0
# CUDA 12.4 devel base by the source's digest, its apt packages, the two video models and the VGG16
# weights that LPIPS reads. It is built once for the agent's image and the verifier's, which add
# each its own pip packages (the source holds the two to different versions) and build on it, so
# that both share its layers, the 122 GB of weights among them.

# The models (`model-assets.json`, Wan2.1-T2V-14B-Diffusers and the HunyuanVideo subset, every file
# the source's `snapshot_download` calls fetch), each file fetched by its revision and SHA-256: the
# source's LFS SHA-256 for the 43 large files, and for the 28 small ones the SHA-256 of the bytes
# whose git blob ID is the one `model-assets.json` gives. Each stage holds at most
# 30 GB, a layer of its own when copied below with `COPY --link`.
FROM scratch AS wan-transformer-a
# Wan-AI/Wan2.1-T2V-14B-Diffusers @ 38ec498cb3208fb688890f8cc7e94ede2cbd7f68: 29.5 GB
ADD --checksum=sha256:63eb07f03808d8ec9b6bfc1b4f4968d7d1253f2569fec18fae114b4530554181 --chmod=0444 \
    https://huggingface.co/Wan-AI/Wan2.1-T2V-14B-Diffusers/resolve/38ec498cb3208fb688890f8cc7e94ede2cbd7f68/transformer/diffusion_pytorch_model-00001-of-00012.safetensors /Wan2.1-T2V-14B-Diffusers/transformer/diffusion_pytorch_model-00001-of-00012.safetensors
ADD --checksum=sha256:8d24b40a5ea6bb21cdc0115128241ffeb44184b2419f4b1585af1573ed5c7839 --chmod=0444 \
    https://huggingface.co/Wan-AI/Wan2.1-T2V-14B-Diffusers/resolve/38ec498cb3208fb688890f8cc7e94ede2cbd7f68/transformer/diffusion_pytorch_model-00002-of-00012.safetensors /Wan2.1-T2V-14B-Diffusers/transformer/diffusion_pytorch_model-00002-of-00012.safetensors
ADD --checksum=sha256:7f550f93b3c36c13015dd551109ff63528251864c9513acc2e0a502807b8a91c --chmod=0444 \
    https://huggingface.co/Wan-AI/Wan2.1-T2V-14B-Diffusers/resolve/38ec498cb3208fb688890f8cc7e94ede2cbd7f68/transformer/diffusion_pytorch_model-00003-of-00012.safetensors /Wan2.1-T2V-14B-Diffusers/transformer/diffusion_pytorch_model-00003-of-00012.safetensors
ADD --checksum=sha256:6a701d601e2cbe3da3515b24d59f56deb5ca2c973c778ff7911f579ec5b8ec11 --chmod=0444 \
    https://huggingface.co/Wan-AI/Wan2.1-T2V-14B-Diffusers/resolve/38ec498cb3208fb688890f8cc7e94ede2cbd7f68/transformer/diffusion_pytorch_model-00004-of-00012.safetensors /Wan2.1-T2V-14B-Diffusers/transformer/diffusion_pytorch_model-00004-of-00012.safetensors
ADD --checksum=sha256:7eb65433e529b975f84897f6572af1ae148d82e7fd0ee7c5f4191158736b104f --chmod=0444 \
    https://huggingface.co/Wan-AI/Wan2.1-T2V-14B-Diffusers/resolve/38ec498cb3208fb688890f8cc7e94ede2cbd7f68/transformer/diffusion_pytorch_model-00005-of-00012.safetensors /Wan2.1-T2V-14B-Diffusers/transformer/diffusion_pytorch_model-00005-of-00012.safetensors
ADD --checksum=sha256:aae199dea0aeb96347e9af9e3cb90679c1c8c2f42c76ee3cead2ea240c909d07 --chmod=0444 \
    https://huggingface.co/Wan-AI/Wan2.1-T2V-14B-Diffusers/resolve/38ec498cb3208fb688890f8cc7e94ede2cbd7f68/transformer/diffusion_pytorch_model-00006-of-00012.safetensors /Wan2.1-T2V-14B-Diffusers/transformer/diffusion_pytorch_model-00006-of-00012.safetensors

FROM scratch AS wan-transformer-b
# Wan-AI/Wan2.1-T2V-14B-Diffusers @ 38ec498cb3208fb688890f8cc7e94ede2cbd7f68: 27.7 GB
ADD --checksum=sha256:53c24e663dece6ff0737b03e5cc670118998231877ba7bad29824ff0500589d4 --chmod=0444 \
    https://huggingface.co/Wan-AI/Wan2.1-T2V-14B-Diffusers/resolve/38ec498cb3208fb688890f8cc7e94ede2cbd7f68/transformer/diffusion_pytorch_model-00007-of-00012.safetensors /Wan2.1-T2V-14B-Diffusers/transformer/diffusion_pytorch_model-00007-of-00012.safetensors
ADD --checksum=sha256:333eaca9dce56bf3083bbab4630aa081b7c403a27ba94894a7756d631529fd9a --chmod=0444 \
    https://huggingface.co/Wan-AI/Wan2.1-T2V-14B-Diffusers/resolve/38ec498cb3208fb688890f8cc7e94ede2cbd7f68/transformer/diffusion_pytorch_model-00008-of-00012.safetensors /Wan2.1-T2V-14B-Diffusers/transformer/diffusion_pytorch_model-00008-of-00012.safetensors
ADD --checksum=sha256:8af811e7ed2ab64a3f6d07827d1b975a9804b78d63c2b3989024bc1b67ac6e5d --chmod=0444 \
    https://huggingface.co/Wan-AI/Wan2.1-T2V-14B-Diffusers/resolve/38ec498cb3208fb688890f8cc7e94ede2cbd7f68/transformer/diffusion_pytorch_model-00009-of-00012.safetensors /Wan2.1-T2V-14B-Diffusers/transformer/diffusion_pytorch_model-00009-of-00012.safetensors
ADD --checksum=sha256:e8b631e38f0ad21a63c631069658d4be4b55e0cb15e8f5848567482b5862ba63 --chmod=0444 \
    https://huggingface.co/Wan-AI/Wan2.1-T2V-14B-Diffusers/resolve/38ec498cb3208fb688890f8cc7e94ede2cbd7f68/transformer/diffusion_pytorch_model-00010-of-00012.safetensors /Wan2.1-T2V-14B-Diffusers/transformer/diffusion_pytorch_model-00010-of-00012.safetensors
ADD --checksum=sha256:21032fe150df6f3ac26780fb8ca38d05c4a9585a1b774ceadb5d42bc96ff1b27 --chmod=0444 \
    https://huggingface.co/Wan-AI/Wan2.1-T2V-14B-Diffusers/resolve/38ec498cb3208fb688890f8cc7e94ede2cbd7f68/transformer/diffusion_pytorch_model-00011-of-00012.safetensors /Wan2.1-T2V-14B-Diffusers/transformer/diffusion_pytorch_model-00011-of-00012.safetensors
ADD --checksum=sha256:7b36878bf76c01410c3be92e5239a10106dfdc932567553228879979e0ac903d --chmod=0444 \
    https://huggingface.co/Wan-AI/Wan2.1-T2V-14B-Diffusers/resolve/38ec498cb3208fb688890f8cc7e94ede2cbd7f68/transformer/diffusion_pytorch_model-00012-of-00012.safetensors /Wan2.1-T2V-14B-Diffusers/transformer/diffusion_pytorch_model-00012-of-00012.safetensors

FROM scratch AS wan-text-encoder
# Wan-AI/Wan2.1-T2V-14B-Diffusers @ 38ec498cb3208fb688890f8cc7e94ede2cbd7f68: 22.7 GB
ADD --checksum=sha256:4087b6192155a4643f6d29fd326c4610103130674011ef4d0a53f8bce5de967d --chmod=0444 \
    https://huggingface.co/Wan-AI/Wan2.1-T2V-14B-Diffusers/resolve/38ec498cb3208fb688890f8cc7e94ede2cbd7f68/text_encoder/config.json /Wan2.1-T2V-14B-Diffusers/text_encoder/config.json
ADD --checksum=sha256:c0ef3a140898e228a3520c9adec60743d2e8e5b3d229651bb37f1a3921919f99 --chmod=0444 \
    https://huggingface.co/Wan-AI/Wan2.1-T2V-14B-Diffusers/resolve/38ec498cb3208fb688890f8cc7e94ede2cbd7f68/text_encoder/model-00001-of-00005.safetensors /Wan2.1-T2V-14B-Diffusers/text_encoder/model-00001-of-00005.safetensors
ADD --checksum=sha256:481c7b2b39771c44df6dd8d13ee12ed072d731b4a650bd092885d4d52db229ad --chmod=0444 \
    https://huggingface.co/Wan-AI/Wan2.1-T2V-14B-Diffusers/resolve/38ec498cb3208fb688890f8cc7e94ede2cbd7f68/text_encoder/model-00002-of-00005.safetensors /Wan2.1-T2V-14B-Diffusers/text_encoder/model-00002-of-00005.safetensors
ADD --checksum=sha256:f93148bcc04052a169e1e49bfcf6125df6cf9bf243cb9c627da75266cf8e35c3 --chmod=0444 \
    https://huggingface.co/Wan-AI/Wan2.1-T2V-14B-Diffusers/resolve/38ec498cb3208fb688890f8cc7e94ede2cbd7f68/text_encoder/model-00003-of-00005.safetensors /Wan2.1-T2V-14B-Diffusers/text_encoder/model-00003-of-00005.safetensors
ADD --checksum=sha256:a451792c739c05bca4606190cc2dd16731411bac03b4cf6aacc5767321f857c9 --chmod=0444 \
    https://huggingface.co/Wan-AI/Wan2.1-T2V-14B-Diffusers/resolve/38ec498cb3208fb688890f8cc7e94ede2cbd7f68/text_encoder/model-00004-of-00005.safetensors /Wan2.1-T2V-14B-Diffusers/text_encoder/model-00004-of-00005.safetensors
ADD --checksum=sha256:7e76e18d224531b8197a46231cb53daf7f2f6ca707130252becf933026ac4eea --chmod=0444 \
    https://huggingface.co/Wan-AI/Wan2.1-T2V-14B-Diffusers/resolve/38ec498cb3208fb688890f8cc7e94ede2cbd7f68/text_encoder/model-00005-of-00005.safetensors /Wan2.1-T2V-14B-Diffusers/text_encoder/model-00005-of-00005.safetensors
ADD --checksum=sha256:8af791f24a6447aa30c95786f40adf23d3c9df97470d2f36159c8f1113f120b4 --chmod=0444 \
    https://huggingface.co/Wan-AI/Wan2.1-T2V-14B-Diffusers/resolve/38ec498cb3208fb688890f8cc7e94ede2cbd7f68/text_encoder/model.safetensors.index.json /Wan2.1-T2V-14B-Diffusers/text_encoder/model.safetensors.index.json

FROM scratch AS wan-rest
# Wan-AI/Wan2.1-T2V-14B-Diffusers @ 38ec498cb3208fb688890f8cc7e94ede2cbd7f68: 0.5 GB
ADD --checksum=sha256:691a307df82d34ed8794db7542199ccfbb0465ed21950c02ab8ff14891d47ea5 --chmod=0444 \
    https://huggingface.co/Wan-AI/Wan2.1-T2V-14B-Diffusers/resolve/38ec498cb3208fb688890f8cc7e94ede2cbd7f68/.gitattributes /Wan2.1-T2V-14B-Diffusers/.gitattributes
ADD --checksum=sha256:a8371c257da24b53abca003c12ef4421fbd81eb9dfe7d3e49960016080beb51b --chmod=0444 \
    https://huggingface.co/Wan-AI/Wan2.1-T2V-14B-Diffusers/resolve/38ec498cb3208fb688890f8cc7e94ede2cbd7f68/README.md /Wan2.1-T2V-14B-Diffusers/README.md
ADD --checksum=sha256:b0e225caffb4b31295ad150f95ee852e4c3dde4a00ac8f79a2ff500f2ce26b8d --chmod=0444 \
    https://huggingface.co/Wan-AI/Wan2.1-T2V-14B-Diffusers/resolve/38ec498cb3208fb688890f8cc7e94ede2cbd7f68/assets/comp_effic.png /Wan2.1-T2V-14B-Diffusers/assets/comp_effic.png
ADD --checksum=sha256:59aec08409f2d46b0e640e4e120dc7cca52c08c3de56d026602dbcff1ebf241a --chmod=0444 \
    https://huggingface.co/Wan-AI/Wan2.1-T2V-14B-Diffusers/resolve/38ec498cb3208fb688890f8cc7e94ede2cbd7f68/assets/data_for_diff_stage.jpg /Wan2.1-T2V-14B-Diffusers/assets/data_for_diff_stage.jpg
ADD --checksum=sha256:6823b3206d8d0cb18d3b5b949dec1217f1178109ba11f14e977b67e1f7b8a248 --chmod=0444 \
    https://huggingface.co/Wan-AI/Wan2.1-T2V-14B-Diffusers/resolve/38ec498cb3208fb688890f8cc7e94ede2cbd7f68/assets/i2v_res.png /Wan2.1-T2V-14B-Diffusers/assets/i2v_res.png
ADD --checksum=sha256:96cddc0f667293436d0b9f92a299b6346b65b231d38ee49719a33d46c91fe1e3 --chmod=0444 \
    https://huggingface.co/Wan-AI/Wan2.1-T2V-14B-Diffusers/resolve/38ec498cb3208fb688890f8cc7e94ede2cbd7f68/assets/logo.png /Wan2.1-T2V-14B-Diffusers/assets/logo.png
ADD --checksum=sha256:91db579092446be2a834bc67721a8e4346936f38c4edb912f459ca3e10f8f439 --chmod=0444 \
    https://huggingface.co/Wan-AI/Wan2.1-T2V-14B-Diffusers/resolve/38ec498cb3208fb688890f8cc7e94ede2cbd7f68/assets/t2v_res.jpg /Wan2.1-T2V-14B-Diffusers/assets/t2v_res.jpg
ADD --checksum=sha256:b7705db79f2e1428ec7a1e6fff8c4fbde062fb95bb233516ddbd04b20007c845 --chmod=0444 \
    https://huggingface.co/Wan-AI/Wan2.1-T2V-14B-Diffusers/resolve/38ec498cb3208fb688890f8cc7e94ede2cbd7f68/assets/vben_1.3b_vs_sota.png /Wan2.1-T2V-14B-Diffusers/assets/vben_1.3b_vs_sota.png
ADD --checksum=sha256:9a0e86ca85046d2675f97984b88b6e74df07bba8a62a31ab8a1aef50d4eda44e --chmod=0444 \
    https://huggingface.co/Wan-AI/Wan2.1-T2V-14B-Diffusers/resolve/38ec498cb3208fb688890f8cc7e94ede2cbd7f68/assets/vben_vs_sota.png /Wan2.1-T2V-14B-Diffusers/assets/vben_vs_sota.png
ADD --checksum=sha256:195dceec6570289d8b01cc51d2e28a7786216f19de55b23978a52610d1646a66 --chmod=0444 \
    https://huggingface.co/Wan-AI/Wan2.1-T2V-14B-Diffusers/resolve/38ec498cb3208fb688890f8cc7e94ede2cbd7f68/assets/video_dit_arch.jpg /Wan2.1-T2V-14B-Diffusers/assets/video_dit_arch.jpg
ADD --checksum=sha256:d8f9e7f7353848056a615c8ef35ab86ec22976bb46cb27405008b4089701945c --chmod=0444 \
    https://huggingface.co/Wan-AI/Wan2.1-T2V-14B-Diffusers/resolve/38ec498cb3208fb688890f8cc7e94ede2cbd7f68/assets/video_vae_res.jpg /Wan2.1-T2V-14B-Diffusers/assets/video_vae_res.jpg
ADD --checksum=sha256:077e3d965090c9028c69c00931675f42e1acc815c6eb450ab291b3b72d211a8e --chmod=0444 \
    https://huggingface.co/Wan-AI/Wan2.1-T2V-14B-Diffusers/resolve/38ec498cb3208fb688890f8cc7e94ede2cbd7f68/examples/i2v_input.JPG /Wan2.1-T2V-14B-Diffusers/examples/i2v_input.JPG
ADD --checksum=sha256:b8b28e022329acf975e027579a1d2dc3de44dba28f4c909b98a880fbf5b0de7d --chmod=0444 \
    https://huggingface.co/Wan-AI/Wan2.1-T2V-14B-Diffusers/resolve/38ec498cb3208fb688890f8cc7e94ede2cbd7f68/model_index.json /Wan2.1-T2V-14B-Diffusers/model_index.json
ADD --checksum=sha256:3fed2abbd9bbc301a74db01947198057ec5049808910dccab320925bf27bea6e --chmod=0444 \
    https://huggingface.co/Wan-AI/Wan2.1-T2V-14B-Diffusers/resolve/38ec498cb3208fb688890f8cc7e94ede2cbd7f68/scheduler/scheduler_config.json /Wan2.1-T2V-14B-Diffusers/scheduler/scheduler_config.json
ADD --checksum=sha256:456b58fd240a06c743a7c2cf8008bec501240d68ebd1fc4018ea569505fea270 --chmod=0444 \
    https://huggingface.co/Wan-AI/Wan2.1-T2V-14B-Diffusers/resolve/38ec498cb3208fb688890f8cc7e94ede2cbd7f68/tokenizer/special_tokens_map.json /Wan2.1-T2V-14B-Diffusers/tokenizer/special_tokens_map.json
ADD --checksum=sha256:e3909a67b780650b35cf529ac782ad2b6b26e6d1f849d3fbb6a872905f452458 --chmod=0444 \
    https://huggingface.co/Wan-AI/Wan2.1-T2V-14B-Diffusers/resolve/38ec498cb3208fb688890f8cc7e94ede2cbd7f68/tokenizer/spiece.model /Wan2.1-T2V-14B-Diffusers/tokenizer/spiece.model
ADD --checksum=sha256:20a46ac256746594ed7e1e3ef733b83fbc5a6f0922aa7480eda961743de080ef --chmod=0444 \
    https://huggingface.co/Wan-AI/Wan2.1-T2V-14B-Diffusers/resolve/38ec498cb3208fb688890f8cc7e94ede2cbd7f68/tokenizer/tokenizer.json /Wan2.1-T2V-14B-Diffusers/tokenizer/tokenizer.json
ADD --checksum=sha256:1d8d2a216bf8e70ac15b7ddcea566c4dd0433c024b39a58ca5e4c66bd78defbd --chmod=0444 \
    https://huggingface.co/Wan-AI/Wan2.1-T2V-14B-Diffusers/resolve/38ec498cb3208fb688890f8cc7e94ede2cbd7f68/tokenizer/tokenizer_config.json /Wan2.1-T2V-14B-Diffusers/tokenizer/tokenizer_config.json
ADD --checksum=sha256:2992fe554957b20824172ebcb91070286fa4a35d94bd9a1c806cee136539719c --chmod=0444 \
    https://huggingface.co/Wan-AI/Wan2.1-T2V-14B-Diffusers/resolve/38ec498cb3208fb688890f8cc7e94ede2cbd7f68/transformer/config.json /Wan2.1-T2V-14B-Diffusers/transformer/config.json
ADD --checksum=sha256:2cf9df7a3544a8b1f78cf1b7a32d5644ce33c215b4a91af3cf5da0ac17e9b2f4 --chmod=0444 \
    https://huggingface.co/Wan-AI/Wan2.1-T2V-14B-Diffusers/resolve/38ec498cb3208fb688890f8cc7e94ede2cbd7f68/transformer/diffusion_pytorch_model.safetensors.index.json /Wan2.1-T2V-14B-Diffusers/transformer/diffusion_pytorch_model.safetensors.index.json
ADD --checksum=sha256:f0c1cc1d7decb5badc384f54691746a27a9aeff49f7ebca974e583389342d527 --chmod=0444 \
    https://huggingface.co/Wan-AI/Wan2.1-T2V-14B-Diffusers/resolve/38ec498cb3208fb688890f8cc7e94ede2cbd7f68/vae/config.json /Wan2.1-T2V-14B-Diffusers/vae/config.json
ADD --checksum=sha256:d6e524b3fffede1787a74e81b30976dce5400c4439ba64222168e607ed19e793 --chmod=0444 \
    https://huggingface.co/Wan-AI/Wan2.1-T2V-14B-Diffusers/resolve/38ec498cb3208fb688890f8cc7e94ede2cbd7f68/vae/diffusion_pytorch_model.safetensors /Wan2.1-T2V-14B-Diffusers/vae/diffusion_pytorch_model.safetensors

FROM scratch AS hunyuan-transformer-a
# tencent/HunyuanVideo @ 2a15b5574ee77888e51ae6f593b2ceed8ce813e5: 14.8 GB
ADD --checksum=sha256:bdb957b35585ea74ae42ca92865a68fa1bf1ebc6c5b7e686a889e5c977dc24c7 --chmod=0444 \
    https://huggingface.co/tencent/HunyuanVideo/resolve/2a15b5574ee77888e51ae6f593b2ceed8ce813e5/transformer/diffusion_pytorch_model-00001-of-00006.safetensors /HunyuanVideo/transformer/diffusion_pytorch_model-00001-of-00006.safetensors
ADD --checksum=sha256:f16c2fabff7b4d290613975ed5fd0853b72b09463078b0d496d29c79a884ca4d --chmod=0444 \
    https://huggingface.co/tencent/HunyuanVideo/resolve/2a15b5574ee77888e51ae6f593b2ceed8ce813e5/transformer/diffusion_pytorch_model-00002-of-00006.safetensors /HunyuanVideo/transformer/diffusion_pytorch_model-00002-of-00006.safetensors
ADD --checksum=sha256:65761c1b920d043b94cf1de1790cbd4871f3ea41182aa3b0ad6825249945bd6e --chmod=0444 \
    https://huggingface.co/tencent/HunyuanVideo/resolve/2a15b5574ee77888e51ae6f593b2ceed8ce813e5/transformer/diffusion_pytorch_model-00003-of-00006.safetensors /HunyuanVideo/transformer/diffusion_pytorch_model-00003-of-00006.safetensors

FROM scratch AS hunyuan-transformer-b
# tencent/HunyuanVideo @ 2a15b5574ee77888e51ae6f593b2ceed8ce813e5: 10.8 GB
ADD --checksum=sha256:7f1c385ac8a924f42f1e26238ed96fcabcf05434d51a25af54252c09e64efaa0 --chmod=0444 \
    https://huggingface.co/tencent/HunyuanVideo/resolve/2a15b5574ee77888e51ae6f593b2ceed8ce813e5/transformer/diffusion_pytorch_model-00004-of-00006.safetensors /HunyuanVideo/transformer/diffusion_pytorch_model-00004-of-00006.safetensors
ADD --checksum=sha256:d81f37cbee8d8fee160535d3f1f55fc9dc09741b38a3f8da3c6ae3487a11d350 --chmod=0444 \
    https://huggingface.co/tencent/HunyuanVideo/resolve/2a15b5574ee77888e51ae6f593b2ceed8ce813e5/transformer/diffusion_pytorch_model-00005-of-00006.safetensors /HunyuanVideo/transformer/diffusion_pytorch_model-00005-of-00006.safetensors
ADD --checksum=sha256:0b027bd330d575c3c6e20e8fc9c585f9075742f467de687772c80898436a88fe --chmod=0444 \
    https://huggingface.co/tencent/HunyuanVideo/resolve/2a15b5574ee77888e51ae6f593b2ceed8ce813e5/transformer/diffusion_pytorch_model-00006-of-00006.safetensors /HunyuanVideo/transformer/diffusion_pytorch_model-00006-of-00006.safetensors

FROM scratch AS hunyuan-rest
# tencent/HunyuanVideo @ 2a15b5574ee77888e51ae6f593b2ceed8ce813e5: 16.3 GB
ADD --checksum=sha256:6ba8ea0dc1c464c9ab549b281be5d3afcab93c3308a6967d22240150d844c54d --chmod=0444 \
    https://huggingface.co/tencent/HunyuanVideo/resolve/2a15b5574ee77888e51ae6f593b2ceed8ce813e5/LICENSE /HunyuanVideo/LICENSE
ADD --checksum=sha256:6f7d8a47d6a713f0a1f88841520a9f55b431c422a52d83962695883ae2fe8008 --chmod=0444 \
    https://huggingface.co/tencent/HunyuanVideo/resolve/2a15b5574ee77888e51ae6f593b2ceed8ce813e5/Notice /HunyuanVideo/Notice
ADD --checksum=sha256:da812a4662500e50055a9ce8636cf13f1f16e21b2640424451a56ac0e32498ff --chmod=0444 \
    https://huggingface.co/tencent/HunyuanVideo/resolve/2a15b5574ee77888e51ae6f593b2ceed8ce813e5/README.md /HunyuanVideo/README.md
ADD --checksum=sha256:34b794bc21967bdc0bdc56be8158f883272659c07d8cf89669b2d5b15adbf52c --chmod=0444 \
    https://huggingface.co/tencent/HunyuanVideo/resolve/2a15b5574ee77888e51ae6f593b2ceed8ce813e5/model_index.json /HunyuanVideo/model_index.json
ADD --checksum=sha256:f1d67f3728a1a54de830ccb3b5856308a3f2844bad5950c60523fe87d6bb2bbe --chmod=0444 \
    https://huggingface.co/tencent/HunyuanVideo/resolve/2a15b5574ee77888e51ae6f593b2ceed8ce813e5/scheduler/scheduler_config.json /HunyuanVideo/scheduler/scheduler_config.json
ADD --checksum=sha256:085cebf4169836dc4a521e5aa88fb2bfe742ddf7a5d56f93bab4e79e540da567 --chmod=0444 \
    https://huggingface.co/tencent/HunyuanVideo/resolve/2a15b5574ee77888e51ae6f593b2ceed8ce813e5/text_encoder/config.json /HunyuanVideo/text_encoder/config.json
ADD --checksum=sha256:f5ad57d3eda300a3195bc9c0bb36ab76ebe88831f128e9851e63440aff4a6741 --chmod=0444 \
    https://huggingface.co/tencent/HunyuanVideo/resolve/2a15b5574ee77888e51ae6f593b2ceed8ce813e5/text_encoder/model-00001-of-00004.safetensors /HunyuanVideo/text_encoder/model-00001-of-00004.safetensors
ADD --checksum=sha256:54accb98811931fca7598da4f7239b03b912eaa2bd5fe639f2da00923374f4a0 --chmod=0444 \
    https://huggingface.co/tencent/HunyuanVideo/resolve/2a15b5574ee77888e51ae6f593b2ceed8ce813e5/text_encoder/model-00002-of-00004.safetensors /HunyuanVideo/text_encoder/model-00002-of-00004.safetensors
ADD --checksum=sha256:d67c77f57cab4c9bf7f4420c256aed684c8ac7b49c6ab72cff9924e9513db9f1 --chmod=0444 \
    https://huggingface.co/tencent/HunyuanVideo/resolve/2a15b5574ee77888e51ae6f593b2ceed8ce813e5/text_encoder/model-00003-of-00004.safetensors /HunyuanVideo/text_encoder/model-00003-of-00004.safetensors
ADD --checksum=sha256:c93f133c65ab1aeaa9ed1e998901a306636375b2f57fa53cd279241147a9a0e9 --chmod=0444 \
    https://huggingface.co/tencent/HunyuanVideo/resolve/2a15b5574ee77888e51ae6f593b2ceed8ce813e5/text_encoder/model-00004-of-00004.safetensors /HunyuanVideo/text_encoder/model-00004-of-00004.safetensors
ADD --checksum=sha256:ab9a659309b52018d958a2de7b2301fa31f60faa702838a0263e385f29b9b3a8 --chmod=0444 \
    https://huggingface.co/tencent/HunyuanVideo/resolve/2a15b5574ee77888e51ae6f593b2ceed8ce813e5/text_encoder/model.safetensors.index.json /HunyuanVideo/text_encoder/model.safetensors.index.json
ADD --checksum=sha256:2872d09ef8a7453db129a67a3e2d90e4dfa4de1febd0fcfa820ed006dceaff9a --chmod=0444 \
    https://huggingface.co/tencent/HunyuanVideo/resolve/2a15b5574ee77888e51ae6f593b2ceed8ce813e5/text_encoder_2/config.json /HunyuanVideo/text_encoder_2/config.json
ADD --checksum=sha256:660c6f5b1abae9dc498ac2d21e1347d2abdb0cf6c0c0c8576cd796491d9a6cdd --chmod=0444 \
    https://huggingface.co/tencent/HunyuanVideo/resolve/2a15b5574ee77888e51ae6f593b2ceed8ce813e5/text_encoder_2/model.safetensors /HunyuanVideo/text_encoder_2/model.safetensors
ADD --checksum=sha256:e87284810dc7a4d9dbab5ea6b713b3e72a6a909006483760773fb129534b9fe0 --chmod=0444 \
    https://huggingface.co/tencent/HunyuanVideo/resolve/2a15b5574ee77888e51ae6f593b2ceed8ce813e5/tokenizer/special_tokens_map.json /HunyuanVideo/tokenizer/special_tokens_map.json
ADD --checksum=sha256:d2c593db4aa75b17a42c1f74d7cc38e257eaeed222e6a52674c65544165dcbaa --chmod=0444 \
    https://huggingface.co/tencent/HunyuanVideo/resolve/2a15b5574ee77888e51ae6f593b2ceed8ce813e5/tokenizer/tokenizer.json /HunyuanVideo/tokenizer/tokenizer.json
ADD --checksum=sha256:88d8723716a3368430f3a02fd4614c8368b9f87b28c0053fcbdd80b88888b0f7 --chmod=0444 \
    https://huggingface.co/tencent/HunyuanVideo/resolve/2a15b5574ee77888e51ae6f593b2ceed8ce813e5/tokenizer/tokenizer_config.json /HunyuanVideo/tokenizer/tokenizer_config.json
ADD --checksum=sha256:9fd691f7c8039210e0fced15865466c65820d09b63988b0174bfe25de299051a --chmod=0444 \
    https://huggingface.co/tencent/HunyuanVideo/resolve/2a15b5574ee77888e51ae6f593b2ceed8ce813e5/tokenizer_2/merges.txt /HunyuanVideo/tokenizer_2/merges.txt
ADD --checksum=sha256:2cdb3b8331a60c92fc1e55a13e9fd61fd2293c5a51275fdcccd62b780052530e --chmod=0444 \
    https://huggingface.co/tencent/HunyuanVideo/resolve/2a15b5574ee77888e51ae6f593b2ceed8ce813e5/tokenizer_2/special_tokens_map.json /HunyuanVideo/tokenizer_2/special_tokens_map.json
ADD --checksum=sha256:39bae5313c2da6467866dee758e754d69f0e302a875027b142531cb9c51c7c28 --chmod=0444 \
    https://huggingface.co/tencent/HunyuanVideo/resolve/2a15b5574ee77888e51ae6f593b2ceed8ce813e5/tokenizer_2/tokenizer_config.json /HunyuanVideo/tokenizer_2/tokenizer_config.json
ADD --checksum=sha256:e089ad92ba36837a0d31433e555c8f45fe601ab5c221d4f607ded32d9f7a4349 --chmod=0444 \
    https://huggingface.co/tencent/HunyuanVideo/resolve/2a15b5574ee77888e51ae6f593b2ceed8ce813e5/tokenizer_2/vocab.json /HunyuanVideo/tokenizer_2/vocab.json
ADD --checksum=sha256:87e056fd4f2d59a5999fb172819ea322944b2c42b95623d875e6c8da8fe790fa --chmod=0444 \
    https://huggingface.co/tencent/HunyuanVideo/resolve/2a15b5574ee77888e51ae6f593b2ceed8ce813e5/transformer/config.json /HunyuanVideo/transformer/config.json
ADD --checksum=sha256:99f9c26610498461f2e4f6fcb1f1432a6618771d8f405b66104aa0893fefaef1 --chmod=0444 \
    https://huggingface.co/tencent/HunyuanVideo/resolve/2a15b5574ee77888e51ae6f593b2ceed8ce813e5/transformer/diffusion_pytorch_model.safetensors.index.json /HunyuanVideo/transformer/diffusion_pytorch_model.safetensors.index.json
ADD --checksum=sha256:8dd1ef5ce49d2a56cdb5ea4c6553244320cd7185918463578c4c08a31f55e50a --chmod=0444 \
    https://huggingface.co/tencent/HunyuanVideo/resolve/2a15b5574ee77888e51ae6f593b2ceed8ce813e5/vae/config.json /HunyuanVideo/vae/config.json
ADD --checksum=sha256:7c68a6295f9034a88225fbafb1f3258291a08d57a1fdb938233fa57b1b8f4883 --chmod=0444 \
    https://huggingface.co/tencent/HunyuanVideo/resolve/2a15b5574ee77888e51ae6f593b2ceed8ce813e5/vae/diffusion_pytorch_model.safetensors /HunyuanVideo/vae/diffusion_pytorch_model.safetensors

# The VGG16 weights that LPIPS loads, by the SHA-256 whose prefix their name gives (the source's
# `torch.hub` check).
FROM scratch AS vgg16
ADD --checksum=sha256:397923af8e79cdbb6a7127f12361acd7a2f83e06b05044ddf496e83de57a5bf0 --chmod=0444 \
    https://download.pytorch.org/models/vgg16-397923af.pth /vgg16-397923af.pth

FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel@sha256:0cf3402e946b7c384ba943ee05c90b4c5a4a05227923921f2b0918c011cfaf56

USER root
ARG DEBIAN_FRONTEND=noninteractive
# The source's packages less the SSH ones and
# `libnss-wrapper`, which serve only its SSH runner, from one dated Ubuntu archive snapshot
# (`fetch/apt_snapshot.sh`) that replaces the base's archive and NVIDIA's CUDA repository.
RUN --mount=type=bind,source=fetch/apt_snapshot.sh,target=/context/fetch/apt_snapshot.sh \
    --mount=type=bind,source=fetch/apt_install.sh,target=/context/fetch/apt_install.sh \
    sh /context/fetch/apt_snapshot.sh jammy 20261103T000000Z \
    && sh /context/fetch/apt_install.sh bash ca-certificates cmake ffmpeg g++-11 gcc-11 git make python3 python3-pip tar

# The models and the VGG16 weights, read-only and offline from here on.
COPY --link --from=wan-transformer-a / /opt/frontier/models/
COPY --link --from=wan-transformer-b / /opt/frontier/models/
COPY --link --from=wan-text-encoder / /opt/frontier/models/
COPY --link --from=wan-rest / /opt/frontier/models/
COPY --link --from=hunyuan-transformer-a / /opt/frontier/models/
COPY --link --from=hunyuan-transformer-b / /opt/frontier/models/
COPY --link --from=hunyuan-rest / /opt/frontier/models/
COPY --link --from=vgg16 /vgg16-397923af.pth /opt/frontier/models/torch/hub/checkpoints/vgg16-397923af.pth
RUN find /opt/frontier/models -type d -exec chmod 0555 {} + \
    && install --directory /opt/frontier/bin
ENV FRONTIER_MODEL_ROOT=/opt/frontier/models HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    TORCH_HOME=/opt/frontier/models/torch
