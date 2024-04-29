FROM python:3

COPY ./requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY ingest.py constants.py ./
COPY . .

EXPOSE 5110/tcp
CMD [ "python", "./run_localGPT_API.py" ]
