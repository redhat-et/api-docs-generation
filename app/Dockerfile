FROM python:3.11 as builder


COPY ./data/raw/markdown_data.json ./markdown_data.json
COPY ./data/raw/nested_data.json ./nested_data.json

FROM python:3.11

WORKDIR /app

COPY ./app/requirements.txt .

RUN pip install -r requirements.txt

ADD ./app  .

CMD ["streamlit", "run",  "app.py", "--browser.gatherUsageStats", "false", "--server.address", "0.0.0.0"]
