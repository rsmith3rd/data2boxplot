FROM python:3.11

# Set working directory
WORKDIR /app

# Copy everything in
COPY . .

# Install dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Run Streamlit on the correct port
CMD streamlit run data2boxplot.py --server.port=$PORT --server.enableCORS false --server.enableXsrfProtection false
