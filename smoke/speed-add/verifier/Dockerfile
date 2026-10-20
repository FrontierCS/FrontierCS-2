FROM rust:1.93.1-bookworm@sha256:7c4ae649a84014c467d79319bbf17ce2632ae8b8be123ac2fb2ea5be46823f31 AS build

WORKDIR /src
COPY Cargo.toml Cargo.lock ./
COPY src ./src
RUN cargo build --locked --release

FROM debian:bookworm-slim@sha256:abd67ffcfa541b485a3dff59865ab629aa048a6c613e639d36e7456b0b229241

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates python3 \
    && rm -rf /var/lib/apt/lists/* \
    && mkdir -p /opt/frontier /frontier/output \
    && chown -R 65532:65532 /frontier

COPY --chmod=0755 --from=build \
    /src/target/release/frontier-speed-add-verifier \
    /opt/frontier/speed-add-verifier

USER 65532:65532
ENTRYPOINT ["/opt/frontier/speed-add-verifier"]
