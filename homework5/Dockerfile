# docker build -t homework5 .
# docker run -ti --rm -p 9000:9696 homework5

FROM agrigorev/zoomcamp-model:3.8.12-slim

RUN pip install pipenv

WORKDIR /app

COPY ["Pipfile", "Pipfile.lock", "./"]
RUN pipenv install --system --deploy

COPY ["serve.py", "model2.bin", "dv.bin", "./"]

EXPOSE 9697

ENTRYPOINT ["gunicorn", "--bind=0.0.0.0:9697", "serve:app"]