# Build whisper.cpp with the CUDA backend on stock Arch Linux.
FROM archlinux:latest AS build
ENV TERM=xterm
ARG WHISPER_CPP_REF=v1.9.5

RUN pacman -Suy --noprogressbar --noconfirm && \
    pacman -S --noprogressbar --noconfirm --needed git cmake make gcc cuda curl

# SM86 = GeForce RTX 3090; there is no GPU inside the build container, so the
# architecture must be given explicitly. Arch's cuda installs to /opt/cuda,
# which is not on PATH — point CMake at it directly.
RUN git clone --depth 1 --branch ${WHISPER_CPP_REF} https://github.com/ggml-org/whisper.cpp /src && \
    cmake -S /src -B /src/build \
        -DGGML_CUDA=ON \
        -DBUILD_SHARED_LIBS=OFF \
        -DCUDAToolkit_ROOT=/opt/cuda \
        -DCMAKE_CUDA_COMPILER=/opt/cuda/bin/nvcc \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_CUDA_ARCHITECTURES=86 \
        -DWHISPER_BUILD_TESTS=OFF && \
    cmake --build /src/build --target whisper-server -j"$(nproc)" && \
    mkdir -p /opt/whisper/bin && \
    cp /src/build/bin/whisper-server /opt/whisper/bin/whisper-server

FROM archlinux:latest
ENV TERM=xterm

# The whisper-server binary needs only the CUDA runtime libraries; the driver
# itself (libcuda.so.1) is injected by the NVIDIA container runtime. Same
# thin-runtime pattern as a CUDA base image, on stock Arch. The driver must
# be >= 580 (CUDA 13).
RUN pacman -Suy --noprogressbar --noconfirm && \
    pacman -S --noprogressbar --noconfirm --needed python python-pip curl && \
    useradd -m -s /bin/bash -d /app app

COPY --from=build /opt/cuda/lib64/libcudart.so.13 /opt/cuda/lib64/
COPY --from=build /opt/cuda/lib64/libcublas.so.13 /opt/cuda/lib64/
COPY --from=build /opt/cuda/lib64/libcublasLt.so.13 /opt/cuda/lib64/

USER app
WORKDIR /app
ENV PIP_BREAK_SYSTEM_PACKAGES=1
# Non-root pip installs console scripts to ~/.local/bin.
ENV PATH="/app/.local/bin:$PATH"
RUN pip install --no-cache-dir "poetry==2.5.1"

COPY --chown=app pyproject.toml poetry.lock ./
RUN poetry config virtualenvs.in-project true --local && \
    poetry install --only main --no-interaction --no-root

COPY --chown=app stt_gateway/ /app/stt_gateway/
COPY --chown=app benchmarks/ /app/benchmarks/
COPY --chown=app scripts/ /app/scripts/
COPY --chmod=0755 --chown=app entrypoint.sh /app/entrypoint.sh
COPY --from=build --chown=app /opt/whisper/bin/whisper-server /opt/whisper/bin/whisper-server

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONPATH=/app \
    LD_LIBRARY_PATH=/opt/cuda/lib64 \
    STT_ENGINE_BIN=/opt/whisper/bin/whisper-server \
    STT_MODELS_DIR=/models

EXPOSE 8030
ENTRYPOINT ["/app/entrypoint.sh"]
