#FROM tensorflow/tensorflow:latest
# Set working directory
#WORKDIR /app

# Install minimal additional dependencies without progress bars
#RUN pip install --no-cache-dir --progress-bar off --ignore-installed flask numpy

# Copy your application
#COPY . /app/

# Expose port
#EXPOSE 5000

# Run the application
#CMD ["python", "app.py"]

# Stage 1: Build dependencies
# Stage 1: Build dependencies
FROM python:3.12 AS builder

WORKDIR /app

# Avoid threading issues in low-resource environments
ENV OPENBLAS_NUM_THREADS=1
ENV TF_NUM_INTRAOP_THREADS=1
ENV TF_NUM_INTEROP_THREADS=1

# Set OpenBLAS to use a single thread to avoid pthread_create issues
ENV OPENBLAS_NUM_THREADS=1

# Install pip and poetry (no progress bar to avoid threading/thread limit issues)
RUN pip install --no-cache-dir --progress-bar off pip==24.2 && \
    pip install --no-cache-dir --progress-bar off poetry==1.8.3

# Copy poetry configuration
COPY pyproject.toml poetry.lock* /app/

# Avoid creating a virtual environment
RUN poetry config virtualenvs.create false

# Export requirements to pip format
RUN poetry export -f requirements.txt --without-hashes > requirements.txt

# Stage 2: Final image
FROM python:3.12

WORKDIR /app

# Set OpenBLAS to single-threaded again in final image
ENV OPENBLAS_NUM_THREADS=1

# Copy and install requirements
COPY --from=builder /app/requirements.txt .
RUN pip install --no-cache-dir --progress-bar off -r requirements.txt

# Copy application code
COPY . /app/

# Expose app port
EXPOSE 5000

# Run app
CMD ["python", "app.py"]

