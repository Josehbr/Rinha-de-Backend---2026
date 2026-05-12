# ── Stage 1: builder PGO-instrumentado ───────────────────────────────────────
# Compila fraud-api com profile-generate. Esse binário não é usado em produção;
# apenas roda contra uma workload representativa para coletar dados PGO.
FROM rust:1.95-slim AS pgo-instrumented

WORKDIR /app

RUN apt-get update && apt-get install -y musl-tools && rm -rf /var/lib/apt/lists/*
RUN rustup target add x86_64-unknown-linux-musl
RUN rustup component add llvm-tools-preview

# Cache deps
COPY Cargo.toml Cargo.lock* ./
COPY .cargo .cargo
RUN mkdir -p src/bin && \
    echo 'fn main() {}' > src/main.rs && \
    echo 'fn main() {}' > src/bin/build_index.rs && \
    cargo build --release --target x86_64-unknown-linux-musl 2>/dev/null || true

COPY src ./src
RUN touch src/main.rs src/bin/build_index.rs

# Build com profile-generate. PGO_DIR fica fora do mount cargo pra persistir.
ENV PGO_DIR=/tmp/pgo-data
RUN mkdir -p $PGO_DIR && \
    RUSTFLAGS="-C target-cpu=x86-64-v3 -C link-arg=-static -C profile-generate=$PGO_DIR" \
    cargo build --release --target x86_64-unknown-linux-musl --bin fraud-api && \
    cp target/x86_64-unknown-linux-musl/release/fraud-api /usr/local/bin/fraud-api-pgo-gen

# Compila build-index com flags normais (não precisa PGO)
RUN cargo build --release --target x86_64-unknown-linux-musl --bin build-index && \
    cp target/x86_64-unknown-linux-musl/release/build-index /usr/local/bin/build-index

# ── Stage 2: build do índice ──────────────────────────────────────────────────
FROM debian:bookworm-slim AS index-builder

WORKDIR /app
COPY --from=pgo-instrumented /usr/local/bin/build-index ./
COPY resources/ ./resources/
RUN ./build-index

# ── Stage 3: profile-run — coleta dados PGO ───────────────────────────────────
# Roda o fraud-api PGO-instrumentado, faz curl contra ele com payloads de exemplo,
# captura o profile data. O resultado merged fica em /pgo/merged.profdata.
FROM rust:1.95-slim AS pgo-runner

WORKDIR /app
RUN apt-get update && apt-get install -y python3 curl && rm -rf /var/lib/apt/lists/*
RUN rustup component add llvm-tools-preview

COPY --from=pgo-instrumented /usr/local/bin/fraud-api-pgo-gen /usr/local/bin/fraud-api
COPY --from=index-builder /app/index.bin ./index.bin
COPY resources/ ./resources/
COPY test/test-data.json /app/test/
COPY scripts/pgo-workload.sh /usr/local/bin/pgo-workload.sh
RUN chmod +x /usr/local/bin/pgo-workload.sh

ENV PGO_DIR=/tmp/pgo-data
ENV INDEX_PATH=/app/index.bin
ENV MCC_RISK_PATH=/app/resources/mcc_risk.json
# monoio binds to UDS; no TCP port needed for profiling.
ENV BIND_UDS=/tmp/pgo-api.sock
ENV UDS=/tmp/pgo-api.sock
# LLVM_PROFILE_FILE força flush do profile no exit (suporta padrões %m, %p)
ENV LLVM_PROFILE_FILE=/tmp/pgo-data/profraw-%m_%p.profraw

# Sobe servidor, espera UDS aparecer, roda workload, derruba, merge profile
RUN mkdir -p $PGO_DIR /pgo && \
    fraud-api & \
    SERVER_PID=$! && \
    until [ -S /tmp/pgo-api.sock ]; do sleep 0.1; done && \
    curl -sf --unix-socket /tmp/pgo-api.sock http://localhost/ready && \
    /usr/local/bin/pgo-workload.sh && \
    kill -TERM $SERVER_PID && \
    wait $SERVER_PID 2>/dev/null || true && \
    sleep 2 && \
    echo "[pgo] profraw files:" && \
    ls -la $PGO_DIR/ && \
    LLVM_PROFDATA=$(find /usr/local/rustup -name llvm-profdata | head -1) && \
    echo "Using profdata: $LLVM_PROFDATA" && \
    $LLVM_PROFDATA merge -o /pgo/merged.profdata $PGO_DIR/*.profraw && \
    ls -la /pgo/

# ── Stage 4: builder final com profile-use ────────────────────────────────────
FROM rust:1.95-slim AS pgo-optimized

WORKDIR /app

RUN apt-get update && apt-get install -y musl-tools && rm -rf /var/lib/apt/lists/*
RUN rustup target add x86_64-unknown-linux-musl
RUN rustup component add llvm-tools-preview

COPY Cargo.toml Cargo.lock* ./
COPY .cargo .cargo
RUN mkdir -p src/bin && \
    echo 'fn main() {}' > src/main.rs && \
    echo 'fn main() {}' > src/bin/build_index.rs && \
    cargo build --release --target x86_64-unknown-linux-musl 2>/dev/null || true

COPY src ./src
RUN touch src/main.rs src/bin/build_index.rs

COPY --from=pgo-runner /pgo/merged.profdata /pgo/merged.profdata

# Build final com profile-use. PGO + LTO + codegen-units=1 produz o binário hot
RUN RUSTFLAGS="-C target-cpu=x86-64-v3 -C link-arg=-static -C profile-use=/pgo/merged.profdata -C llvm-args=-pgo-warn-missing-function" \
    cargo build --release --target x86_64-unknown-linux-musl --bin fraud-api

# ── Stage 5: utilitário healthcheck ──────────────────────────────────────────
FROM busybox:1.36-musl AS healthcheck-tools

# ── Stage 6: runtime scratch ──────────────────────────────────────────────────
FROM scratch

WORKDIR /app

COPY --from=pgo-optimized /app/target/x86_64-unknown-linux-musl/release/fraud-api ./
COPY --from=index-builder /app/index.bin ./
COPY resources/mcc_risk.json ./resources/
COPY --from=healthcheck-tools /bin/busybox /bin/busybox

EXPOSE 9998

CMD ["./fraud-api"]
