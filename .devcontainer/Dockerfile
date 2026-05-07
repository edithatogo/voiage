# Dockerfile for voiage devcontainer
FROM mcr.microsoft.com/devcontainers/python:3.11

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Set up working directory
WORKDIR /workspace

# Copy project files
COPY pyproject.toml uv.lock* ./

# Install dependencies
RUN uv sync --extra dev || true

# Install pre-commit
RUN uv run pre-commit install

# Set up scalene for profiling
RUN uv add scalene

# Set up mutmut for mutation testing
RUN uv add mutmut
