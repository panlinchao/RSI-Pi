# Evolver runtime for the RSI loop.
#
# The evolver is pi itself: it inspects feedback and generates one candidate
# modification. It gets read-only development evidence and a separate writable
# candidate workspace.
# It uses its own bash/read/edit/write tools to inspect feedback and edit code.
#
# The driver mounts evidence, retained code, and the fixed budget guard read-only.
# Only the candidate workspace is writable.
#
# Build once:
#   docker build -t rsi-evolver:0.86.1 open_research_RSI/bench/evolver

FROM node:22.23.3-bookworm-slim

# Pinned to the same version the task containers install, so the agent that
# evolves the module is the agent that runs it.
ARG PI_VERSION=0.86.1

# git is not needed by pi itself, but it makes the workspace diffable when
# debugging a proposal by hand.
RUN apt-get update \
    && apt-get install -y --no-install-recommends git ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN npm install --global --ignore-scripts --no-audit --no-fund \
    @earendil-works/pi-coding-agent@${PI_VERSION} \
    && pi --version

WORKDIR /workspace
