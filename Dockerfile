FROM python:3.10
WORKDIR /app
COPY . /app
RUN pip install -r requirements.txt
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.enableXsrfProtection", "false", "--server.port", "8501", "--server.address", "0.0.0.0"]
