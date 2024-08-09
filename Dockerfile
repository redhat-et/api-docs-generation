FROM python:3.11

WORKDIR /app

COPY ./app/requirements.txt .

RUN pip install -r requirements.txt

COPY ./app  .

COPY ./data ./data
COPY ./data ./data

CMD ["streamlit", "run",  "app.py", "--browser.gatherUsageStats", "false", "--server.address", "0.0.0.0"]
