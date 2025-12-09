# Use an official Python base image (you can change version if needed)
FROM python:3.10-slim

# Set a working directory inside the container
WORKDIR /app

# Install system dependencies needed for ProDy compilation
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    build-essential \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy your project files into the container
COPY . /app

# Install ProDy and any other dependencies you need
RUN pip install --no-cache-dir prody numpy biopython matplotlib scipy contact_map

# Set the default command to run your Python script
# Replace with the name of your main script or use bash for interactive use
CMD ["bash"]
