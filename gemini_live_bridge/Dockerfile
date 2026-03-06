ARG BUILD_FROM=ghcr.io/home-assistant/aarch64-base:latest
FROM $BUILD_FROM

# Install Python and dependencies
RUN apk add --no-cache \
    python3 \
    py3-pip \
    py3-numpy

# Copy requirements
COPY requirements.txt /tmp/
RUN pip3 install --break-system-packages --no-cache-dir -r /tmp/requirements.txt

# Copy add-on files
COPY run.py audio.py ha_client.py device_connection.py /
COPY providers/ /providers/
COPY tools/ /tools/
RUN chmod a+x /run.py

CMD [ "python3", "/run.py" ]
