# Runtime image for the Index Brain console and engine.
#
# Two stages: a builder that installs into a virtualenv, and a slim runtime
# that copies it. The build tools do not ship to production, which keeps the
# attack surface of a box that holds broker credentials smaller.
FROM python:3.12-slim AS builder

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /build
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Dependencies before source, so a code change does not reinstall numpy.
COPY pyproject.toml README.md ./
COPY index_option_brain/__init__.py index_option_brain/__init__.py
RUN pip install --no-cache-dir .

COPY index_option_brain/ index_option_brain/
RUN pip install --no-cache-dir --no-deps .


FROM python:3.12-slim AS runtime

ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# curl is here for the container healthcheck below and nothing else.
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --shell /usr/sbin/nologin brain

COPY --from=builder /opt/venv /opt/venv
WORKDIR /app
COPY --chown=brain:brain index_option_brain/ index_option_brain/
COPY --chown=brain:brain docs/ docs/
COPY --chown=brain:brain scripts/ scripts/

# Never root. This process holds broker credentials once one is connected.
USER brain
EXPOSE 8000

# Watches /ready, not /health: a process that is alive but blind looks
# identical to a healthy one from /health, and that is exactly the failure
# an unattended box needs to surface.
#
# start-period is generous because the first poll has to warm NSE's cookies
# before it can read anything.
HEALTHCHECK --interval=60s --timeout=10s --start-period=90s --retries=3 \
    CMD curl -fsS http://127.0.0.1:8000/ready > /dev/null || exit 1

# One worker, deliberately. The bar aggregator holds observed history in
# memory, so a second worker would build a second, different history and the
# console would show whichever one answered.
CMD ["uvicorn", "index_option_brain.app.main:app", \
     "--host", "0.0.0.0", "--port", "8000", "--workers", "1", \
     "--proxy-headers", "--forwarded-allow-ips", "*"]
