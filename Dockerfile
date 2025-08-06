FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app code
COPY . .

# Streamlit must bind to 0.0.0.0 and correct port
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0
ENV STREAMLIT_SERVER_PORT=${PORT}

# Expose port for Fly.io
EXPOSE 8080

# Start the app
CMD ["streamlit", "run", "data2boxplot.py"]
