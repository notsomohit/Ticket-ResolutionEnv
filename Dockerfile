FROM python:3.10-slim

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Set up user
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Copy dependency files
COPY --chown=user pyproject.toml requirements.txt uv.lock ./

# Install dependencies using uv
# We sync to ensure uv.lock is respected. --no-dev to avoid dev dependencies.
RUN uv sync --no-dev

# Copy the rest of the application
COPY --chown=user . .

# Ensure we use the virtualenv created by uv
ENV PATH="/app/.venv/bin:$PATH"

# Run the app on port 7860 for HF Spaces
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
