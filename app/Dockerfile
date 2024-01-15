FROM python:3.11 as builder


COPY ./data/raw/markdown_data.json ./markdown_data.json
COPY ./data/raw/nested_data.json ./nested_data.json

# COPY ./scripts/prepare-data.py prepare-data.py
# RUN python3 prepare-data.py "./backports.json" "./filtered-backports.json"

FROM python:3.11

WORKDIR /app

COPY ./app/requirements.txt .

RUN pip install -r requirements.txt

# try to circumvent permissions error when running in OCP
# RUN mkdir -p ./git_apply_test_repo
# RUN chmod 666 ./git_apply_test_repo

COPY ./app  .

# COPY --from=builder ./filtered-backports.json ./data/backports.json

CMD ["streamlit", "run",  "app.py", "--browser.gatherUsageStats", "false", "--server.address", "0.0.0.0"]