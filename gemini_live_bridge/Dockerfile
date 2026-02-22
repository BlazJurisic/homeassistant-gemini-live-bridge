ARG BUILD_FROM=ghcr.io/home-assistant/aarch64-base:latest
FROM $BUILD_FROM

# Install Python and runtime dependencies
RUN apk add --no-cache \
    python3 \
    py3-pip \
    py3-numpy \
    speexdsp-dev

# Copy requirements and install with build tools (swig needed for speexdsp)
COPY requirements.txt /tmp/
RUN apk add --no-cache --virtual .build-deps \
        gcc g++ musl-dev python3-dev swig && \
    pip3 install --break-system-packages --no-cache-dir -r /tmp/requirements.txt && \
    apk del .build-deps

# Copy add-on files
COPY run.py /
RUN chmod a+x /run.py

CMD [ "python3", "/run.py" ]
