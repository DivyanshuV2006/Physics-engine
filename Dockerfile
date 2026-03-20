# Use the official Kubric-ready image so Blender/Kubric stack is preinstalled.
FROM kubricdockerhub/kubruntu

# Keep all project code and outputs under one mounted workspace path.
WORKDIR /workspace

# Install Python dependencies early to leverage Docker layer caching.
COPY requirements.txt /workspace/requirements.txt
RUN pip install --no-cache-dir -r /workspace/requirements.txt

# Copy project sources; compose volume mount will override this at runtime.
COPY . /workspace

# Default behavior: run the synthetic dataset generator.
CMD ["python", "dataset-generator.py"]
