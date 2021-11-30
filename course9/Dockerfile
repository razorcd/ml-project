# aws ecr-public get-login-password --region us-east-1 --profile razorcd | docker login --username AWS --password-stdin public.ecr.aws/g5w4y6q4
# docker build -t public.ecr.aws/g5w4y6q4/razorcd .
# docker push public.ecr.aws/g5w4y6q4/razorcd
# Event: {"url": "http://bit.ly/mlbookcamp-pants"}

FROM public.ecr.aws/lambda/python:3.8

RUN pip install keras-image-helper
# RUN pip install --extra-index-url https://google-coral.github.io/py-repo/ tflite_runtime
RUN pip install https://github.com/alexeygrigorev/tflite-aws-lambda/raw/main/tflite/tflite_runtime-2.7.0-cp38-cp38-linux_x86_64.whl

COPY clothing-model.tflite .
COPY course9.py .

CMD ["course9.lambda_handler"]