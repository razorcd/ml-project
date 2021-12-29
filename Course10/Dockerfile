# docker build -t zoomcamp-10:xception-1 -f Dockerfile .
# docker run -ti --rm -p 8500:8500 zoomcamp-10:xception-1

FROM tensorflow/serving:2.7.0

COPY clothing-model-saved /models/clothing-model/1
ENV MODEL_NAME="clothing-model"