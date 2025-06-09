FROM python:3.11

COPY requirements.txt .

RUN pip install --no-deps -r requirements.txt

COPY settings/ ./settings/
COPY img/ ./img/
COPY genai_playground.py /
COPY genai_utils.py /

CMD ["python", "./genai_playground.py"]