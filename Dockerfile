FROM python:3.10-slim

# Set up user
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install uv safely
RUN pip install --no-cache-dir uv

# Copy dependency files
COPY --chown=user pyproject.toml requirements.txt uv.lock ./

# Install dependencies using uv sync or fallback to pip
RUN uv sync --no-dev || pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY --chown=user . .

# Ensure we use the virtualenv created by uv if it exists
ENV PATH="/app/.venv/bin:$PATH"

# Expose the correct port for HF Spaces
EXPOSE 7860

# Run the app
CMD ["python", "app.py"]
