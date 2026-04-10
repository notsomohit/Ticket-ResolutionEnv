FROM python:3.10-slim

# Set up user
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Upgrade pip and install dependencies
# We use pip directly to ensure they are in the default path for the user
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir --user -r requirements.txt

# Copy the rest of the application
COPY --chown=user . .

# Expose the correct port for HF Spaces
EXPOSE 7860

# Run the app using uvicorn directly to ensure it's found
# This avoids any "python app.py" environment issues if the script-based run fails
CMD ["python", "app.py"]
