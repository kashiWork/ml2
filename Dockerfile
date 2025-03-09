FROM python:3.9-slim

# Set the working directory
WORKDIR /app

# Copy all files to the container
COPY . /app

# Upgrade pip and install dependencies
RUN pip install --upgrade pip && pip install -r requirements.txt

# Expose the port (Render will map this externally)
EXPOSE 7860

# Run the FastAPI app using Uvicorn; the module is "summarizer" (from summarizer.py) and the app instance is "app"
CMD ["uvicorn", "summarizer:app", "--host", "0.0.0.0", "--port", "7860"]
