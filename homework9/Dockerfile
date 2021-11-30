# aws ecr-public get-login-password --region us-east-1 --profile razorcd | docker login --username AWS --password-stdin public.ecr.aws/g5w4y6q4
# docker build -t public.ecr.aws/g5w4y6q4/razorcd .
# docker push public.ecr.aws/g5w4y6q4/razorcd
# curl -v -X POST -d "{\"url\": \"https://upload.wikimedia.org/wikipedia/commons/1/18/Vombatus_ursinus_-Maria_Island_National_Park.jpg\"}"   http://localhost:8080/2015-03-31/functions/function/invocations
# Event: {"url": "http://bit.ly/mlbookcamp-pants"}

FROM agrigorev/zoomcamp-cats-dogs-lambda:v2

RUN pip install keras-image-helper
RUN pip install --extra-index-url https://google-coral.github.io/py-repo/ tflite_runtime
# RUN pip install https://github.com/alexeygrigorev/tflite-aws-lambda/raw/main/tflite/tflite_runtime-2.7.0-cp38-cp38-linux_x86_64.whl
RUN pip install pillow
RUN pip install tensorflow
RUN pip install scipy

COPY homework9.py .

CMD ["homework9.lambda_handler"]