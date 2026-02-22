ARG BUILD_FROM=ghcr.io/home-assistant/aarch64-base:latest
FROM $BUILD_FROM

# Install Python and dependencies (speexdsp-dev provides libspeexdsp.so for ctypes)
RUN apk add --no-cache \
    python3 \
    py3-pip \
    py3-numpy \
    speexdsp-dev

# Copy requirements
COPY requirements.txt /tmp/
RUN pip3 install --break-system-packages --no-cache-dir -r /tmp/requirements.txt

# Copy add-on files
COPY run.py /
RUN chmod a+x /run.py

CMD [ "python3", "/run.py" ]
