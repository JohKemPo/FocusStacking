# ==========================================
# Estágio 1: Build (Compilação)
# ==========================================
FROM ubuntu:24.04 AS builder
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    libopencv-dev \
    libspdlog-dev \
    libyaml-cpp-dev \
    libcurl4-openssl-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY CMakeLists.txt .
COPY include/ ./include/
COPY src/ ./src/
COPY config/ ./config/

RUN mkdir build && cd build && \
    cmake .. -DCMAKE_BUILD_TYPE=Release && \
    make -j$(nproc)

# ==========================================
# Estágio 2: Runtime (Execução)
# ==========================================
FROM ubuntu:24.04
ENV DEBIAN_FRONTEND=noninteractive

# Variável crucial para o Docker no Windows achar os drivers injetados pela máquina host (WSL2)
ENV LD_LIBRARY_PATH=/usr/lib/wsl/lib:${LD_LIBRARY_PATH}

RUN apt-get update && apt-get install -y \
    libopencv-core4* \
    libopencv-imgproc4* \
    libopencv-imgcodecs4* \
    libopencv-video4* \
    libopencv-calib3d4* \
    libopencv-features2d4* \
    libspdlog1* \
    libyaml-cpp0* \
    libcurl4 \
    ocl-icd-libopencl1 \
    clinfo \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY --from=builder /app/build/FocusStackingCPP /usr/local/bin/FocusStackingCPP
COPY --from=builder /app/config ./config
COPY entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

# O ICD da NVIDIA so existe em runtime (injetado pelo --gpus all), por isso e resolvido
# dinamicamente pelo entrypoint em vez de hardcoded no build.
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]