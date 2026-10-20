FROM python:3.12-slim-bookworm@sha256:72d3d75f2639ab82b34b29390ad3d6e0827c775befee94edda8e9976818f488d

COPY --chmod=0555 controller.py /opt/frontier/controller.py
COPY --chmod=0555 modal_credentials.py /opt/frontier/modal_credentials.py

USER 65532:65532
ENTRYPOINT ["/usr/local/bin/python3", "/opt/frontier/controller.py"]
