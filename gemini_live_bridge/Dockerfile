ARG BUILD_FROM=ghcr.io/home-assistant/aarch64-base:latest

# Stage 1: Build React webapp
FROM node:20-alpine AS webapp
WORKDIR /app
COPY webapp/package*.json ./
RUN npm ci
COPY webapp/ ./
RUN npm run build

# Stage 2: Python runtime
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
COPY run.py audio.py ha_client.py device_connection.py web_server.py /
COPY providers/ /providers/
COPY tools/ /tools/
COPY --from=webapp /app/dist /webapp/
RUN chmod a+x /run.py

CMD [ "python3", "/run.py" ]
