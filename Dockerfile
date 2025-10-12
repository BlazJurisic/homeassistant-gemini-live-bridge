ARG BUILD_FROM
FROM $BUILD_FROM

# Install Python and dependencies
RUN apk add --no-cache \
    python3 \
    py3-pip \
    py3-aiohttp

# Copy requirements
COPY requirements.txt /tmp/
RUN pip3 install --no-cache-dir -r /tmp/requirements.txt

# Copy add-on files
COPY run.py /
RUN chmod a+x /run.py

CMD [ "python3", "/run.py" ]
