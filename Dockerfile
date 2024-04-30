ARG PY_VERSION=3.10

# stage 1: build
FROM python:${PY_VERSION} AS builder
ARG PY_VERSION

WORKDIR /app

COPY ./requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt && \
  find /usr/local/lib/python${PY_VERSION}/site-packages \( -type d -a -name test -o -name tests \) -o \( -type f -a -name '*.pyc' -o -name '*.pyo' \) -exec rm -rf '{}' \+

# stage 2: final
FROM python:${PY_VERSION}-slim
ARG PY_VERSION

WORKDIR /app

COPY --from=builder /usr/local/lib/python${PY_VERSION}/site-packages /usr/local/lib/python${PY_VERSION}/site-packages/

COPY . .

ENV DB_PATH=/db

VOLUME /db

EXPOSE 5110/tcp
CMD [ "python", "./run_localGPT_API.py" ]
