FROM python:3.12-slim

RUN apt-get update \
 && apt-get install -y --no-install-recommends ca-certificates \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . .
RUN pip install --no-cache-dir "mistralai-agents>=1.1.2,<1.2" pandas \
 && useradd --create-home --shell /usr/sbin/nologin agent \
 && chown -R agent:agent /app

USER agent

CMD ["agents", "run-worker"]
