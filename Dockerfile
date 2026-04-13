# =============================================================================
# Application Container (CPU inference + development)
# =============================================================================
# Runs the Phoenix web app with CPU-only EXLA backend.
# Source code is bind-mounted from the host at /app.
# Deps and _build live on named volumes so compilation persists across restarts.
#
# Build:
#   docker build -t chatbot:latest .
#
# Usage with docker-compose:
#   docker compose up
# =============================================================================

FROM hexpm/elixir:1.18.3-erlang-27.2.4-ubuntu-noble-20250127

RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl ca-certificates build-essential inotify-tools \
    && rm -rf /var/lib/apt/lists/*

ENV LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    XLA_TARGET=cpu \
    XLA_BUILD=false

WORKDIR /app

RUN mix local.hex --force && mix local.rebar --force

COPY scripts/entrypoint.sh /usr/local/bin/entrypoint.sh
COPY scripts/exla_preflight.sh /app/scripts/exla_preflight.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

EXPOSE 4000

ENTRYPOINT ["entrypoint.sh"]
CMD ["mix", "phx.server"]
