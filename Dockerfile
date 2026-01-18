# Use the NVIDIA PyTorch base image recommended by Microsoft VibeVoice
FROM nvcr.io/nvidia/pytorch:24.07-py3

# 1. Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

# 2. Install uv
# We download the standalone uv binary for speed and reliability
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# 3. Set the working directory
WORKDIR /app

# 4. Copy dependency files first (for better layer caching)
COPY pyproject.toml uv.lock ./

# 5. Install dependencies 
# --frozen ensures we use the exact versions in uv.lock
# --no-install-project skips installing the current app as a package for now
RUN uv sync --frozen --no-install-project

# 6. Copy the rest of the application code
COPY . .

# 7. Final sync to include the local project code
RUN uv sync --frozen

# 8. Set the entrypoint to run your main script
# Using 'uv run' ensures the virtual environment is used automatically
ENTRYPOINT ["uv", "run", "python", "main.py"]