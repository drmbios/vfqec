FROM python:3.11.13-slim AS builder
WORKDIR /build
RUN apt-get update && apt-get install -y --no-install-recommends build-essential cmake ninja-build ca-certificates \
 && rm -rf /var/lib/apt/lists/*
# Pinned upstream archives avoid full Git-history fetches in ARM PyMatching builds.
ADD --checksum=sha256:81964fe578e9bd7c94dfdb09c8e4d6e6759e19967e397dbea48d1c10e45d0df2 https://codeload.github.com/google/googletest/tar.gz/refs/tags/release-1.12.1 /tmp/gtest.tar.gz
ADD --checksum=sha256:5c1cb841d465e3a60787ff91ea20e984276ed39ac9cdfebf64086a37311aebc0 https://codeload.github.com/quantumlib/Stim/tar.gz/1320ad7eac7de34d2e9c70daa44fbc6d84174450 /tmp/stim.tar.gz
RUN mkdir -p /opt/googletest /opt/stim \
 && tar xzf /tmp/gtest.tar.gz --strip-components=1 -C /opt/googletest \
 && tar xzf /tmp/stim.tar.gz --strip-components=1 -C /opt/stim
ENV CMAKE_ARGS="-DFETCHCONTENT_SOURCE_DIR_GOOGLETEST=/opt/googletest -DFETCHCONTENT_SOURCE_DIR_STIM=/opt/stim" \
    CMAKE_BUILD_PARALLEL_LEVEL=2
COPY pyproject.toml requirements.lock ./
RUN --mount=type=cache,target=/root/.cache/pip pip install pip==25.2 setuptools==80.9.0 wheel==0.45.1 \
 && pip wheel --no-build-isolation --wheel-dir=/wheels -r requirements.lock
COPY src ./src
RUN pip wheel --no-deps --no-build-isolation --wheel-dir=/wheels .

FROM python:3.11.13-slim AS runtime
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR=/tmp/matplotlib \
    HOME=/home/vfqec OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
RUN groupadd --gid 10001 vfqec && useradd --uid 10001 --gid vfqec --create-home vfqec
COPY --from=builder /wheels /wheels
RUN pip install --no-cache-dir --no-index --find-links=/wheels vfqec && rm -rf /wheels
WORKDIR /app
COPY examples ./examples
RUN mkdir /app/results && chown -R vfqec:vfqec /app
USER vfqec
ARG GIT_COMMIT=unversioned
ENV VFQEC_GIT_COMMIT=$GIT_COMMIT
EXPOSE 8000 8501
HEALTHCHECK --interval=30s --timeout=5s CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"
CMD ["uvicorn", "vfqec.api.app:app", "--host", "0.0.0.0", "--port", "8000"]

# Optional Linux x86_64 NVIDIA worker. CPU services always use the runtime target.
FROM runtime AS gpu
USER root
RUN pip uninstall -y qiskit-aer && pip install --no-cache-dir qiskit-aer-gpu==0.17.2
ENV AER_DEVICE=GPU
USER vfqec
