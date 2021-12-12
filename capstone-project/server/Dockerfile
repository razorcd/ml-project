# docker build -t project1_v0.2 .
# docker run -ti --rm -p 9000:9696 project1_v0.2

FROM python:3.8.12-slim

RUN pip install pipenv

WORKDIR /app

COPY ["Pipfile", "Pipfile.lock", "./"]
RUN pipenv install --system --deploy

COPY ["serve.py", "model_xg_0.4_4.bin", "./"]

EXPOSE 9696

ENTRYPOINT ["gunicorn", "--bind=0.0.0.0:9696", "serve:app"]