FROM python:3.10-slim

# Set up user
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Upgrade pip and install uv
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir uv

# Copy dependency files
COPY --chown=user pyproject.toml requirements.txt uv.lock ./

# Install dependencies using uv sync or fallback to pip
# Ensure we install into the user's home or a predictable location
RUN uv sync --no-dev || pip install --no-cache-dir --user -r requirements.txt

# Copy the rest of the application
COPY --chown=user . .

# Ensure PATH includes both the .venv/bin (from uv) AND the user's local bin
ENV PATH="/app/.venv/bin:/home/user/.local/bin:$PATH"

# Expose the correct port for HF Spaces
EXPOSE 7860

# Run the app
CMD ["python", "app.py"]
