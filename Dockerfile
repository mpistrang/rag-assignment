FROM python:3.12-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY hybrid-search/ ./hybrid-search/

# product-documentation is mounted as a volume at runtime
# Create the directory so imports don't fail
RUN mkdir -p product-documentation

# Default command (can be overridden)
CMD ["python", "--version"]
