# Build context must be the repo root (build.rs needs sibling ../proto):
# docker build -f gateway-rs/Dockerfile .

FROM rust:1-slim-bookworm AS builder
# build.rs (tonic-prost-build -> prost-build) needs a real `protoc` on PATH —
# same fix as the gateway-rs CI job (see .github/workflows/ci.yml).
RUN apt-get update && apt-get install -y --no-install-recommends protobuf-compiler \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /build
COPY proto ./proto
COPY gateway-rs ./gateway-rs
RUN cargo build --release --manifest-path gateway-rs/Cargo.toml

FROM debian:bookworm-slim
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates \
    && rm -rf /var/lib/apt/lists/*
COPY --from=builder /build/gateway-rs/target/release/mathforge-gateway /usr/local/bin/mathforge-gateway

ENV MATHFORGE_GATEWAY_HOST=0.0.0.0
EXPOSE 50052
CMD ["mathforge-gateway"]
