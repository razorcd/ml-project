#!/usr/bin/env python
# coding: utf-8

# pipenv install grpcio==1.42.0 flask gunicorn keras-image-helper

# USE:
# (base) ➜  ~ curl -X POST -d "{\"url\":\"http://bit.ly/mlbookcamp-pants\"}" -H 'Content-Type: application/json' localhost:9696/predict
# {
#   "dress": -1.8682903051376343, 
#   "hat": -4.761245250701904, 
#   "longsleeve": -2.316983461380005, 
#   "outwear": -1.0625708103179932, 
#   "pants": 9.887161254882812, 
#   "shirt": -2.8124334812164307, 
#   "shoes": -3.6662826538085938, 
#   "shorts": 3.200361728668213, 
#   "skirt": -2.6023378372192383, 
#   "t-shirt": -4.835046291351318
# }


# Call server:
# curl -X POST -d "{\"url\":\"http://bit.ly/mlbookcamp-pants\"}" -H 'Content-Type: application/json' localhost:9696/predict

#create grpc client, load predict image and return prediction

import grpc
import os

from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc
from keras_image_helper import create_preprocessor
from proto import np_to_protobuf

model_classes = ['dress', 'hat', 'longsleeve', 'outwear', 'pants', 'shirt', 'shoes', 'shorts', 'skirt', 't-shirt']

tf_host = os.getenv("TF_SERVING_HOST", "localhost:8500")
print("TF host on " + str(tf_host))


channel = grpc.insecure_channel(tf_host)
stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)

preprocessor = create_preprocessor('xception', target_size=(299,299))



def prepare_request(inputX):
    pb_request = predict_pb2.PredictRequest()
    pb_request.model_spec.name="clothing-model"
    pb_request.model_spec.signature_name = "serving_default"

    pb_request.inputs['input_8'].CopyFrom(np_to_protobuf(inputX))
    return pb_request

def prepare_response(pb_response):
    preds = pb_response.outputs['dense_7'].float_val    
    return dict(zip(model_classes, preds))

def predict(url):
    # url = 'http://bit.ly/mlbookcamp-pants'
    X = preprocessor.from_url(url)

    request = prepare_request(X)
    pb_response = stub.Predict(request, timeout=20.0)
    return prepare_response(pb_response)





from flask import Flask
from flask import request
from flask import jsonify

app = Flask('script')

@app.route('/predict', methods=['POST'])
def predict_endpoint():
    print("Request: "+str(request))
    data = request.get_json()
    print("Request json: "+str(data))
    url = data['url']
    result = predict(url)
    json_result = jsonify(result)
    print("Response data: "+str(result))
    print("Response: "+str(json_result))
    return json_result
    

if __name__=='__main__':
    # result = predict('http://bit.ly/mlbookcamp-pants')
    # print(result)
    app.run(debug=True, host='0.0.0.0', port=9696)