FROM rustlang/rust:nightly-bookworm

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates curl gnupg lsb-release wget build-essential cmake python3 pkg-config libffi-dev \
    && wget -qO- https://apt.llvm.org/llvm-snapshot.gpg.key | gpg --dearmor -o /usr/share/keyrings/llvm.gpg \
    && echo "deb [signed-by=/usr/share/keyrings/llvm.gpg] http://apt.llvm.org/bookworm/ llvm-toolchain-bookworm-20 main" \
        > /etc/apt/sources.list.d/llvm20.list \
    && apt-get update && apt-get install -y --no-install-recommends \
        llvm-20 llvm-20-dev clang-20 libclang-20-dev \
    && rm -rf /var/lib/apt/lists/*

ENV LLVM_SYS_201_PREFIX=/usr/lib/llvm-20
ENV PATH="/usr/lib/llvm-20/bin:${PATH}"
ENV CC=clang-20
ENV CXX=clang++-20

WORKDIR /src
COPY . .

RUN rustup toolchain install nightly --profile minimal && rustup default nightly
RUN cargo build --release

CMD ["cargo", "run", "--release", "--bin", "tridentix", "--", "run", "examples/hello.trix"]
