RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    curl \
    nodejs \
    npm \
    && rm -rf /var/lib/apt/lists/*