FROM python:3.6-slim

WORKDIR /code
COPY . /code

RUN pip install -r requirements.txt

CMD [ "python", "train.py" ]
