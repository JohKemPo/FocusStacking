#!/bin/bash
set -e

# Resolve dinamicamente a lib OpenCL da NVIDIA injetada pelo Docker Desktop/WSL2 em runtime
# (nao existe no build time, por isso nao pode ser hardcoded no Dockerfile)
mkdir -p /etc/OpenCL/vendors

NVIDIA_OCL=$(find /usr/lib/wsl/lib /usr/lib/x86_64-linux-gnu -iname 'libnvidia-opencl.so*' 2>/dev/null | sort -V | tail -n1)

if [ -n "$NVIDIA_OCL" ]; then
    echo "$NVIDIA_OCL" > /etc/OpenCL/vendors/nvidia.icd
    echo "[entrypoint] OpenCL ICD da NVIDIA registrado: $NVIDIA_OCL" >&2
else
    echo "[entrypoint] AVISO: nenhuma lib libnvidia-opencl.so encontrada. GPU via OpenCL indisponivel; fallback para CPU." >&2
fi

exec /usr/local/bin/FocusStackingCPP "$@"