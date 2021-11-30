# ipython
# import course9
# event = {'url': 'https://upload.wikimedia.org/wikipedia/commons/9/9a/Pug_600.jpg'}
# homework9.lambda_handler(event, None)


from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
from tensorflow.keras.preprocessing import image

import tensorflow.lite as tflite
# import tflite_runtime.interpreter as tflite



# interpreter = tflite.Interpreter(model_path='dogs_cats_10_0.687.tflite')
interpreter = tflite.Interpreter(model_path='cats-dogs-v2.tflite')
interpreter.allocate_tensors()

print("Input details:", interpreter.get_input_details())
input_index = interpreter.get_input_details()[0]['index']

print("Output details:", interpreter.get_output_details())
output_index = interpreter.get_output_details()[0]['index']


from io import BytesIO
from urllib import request

from PIL import Image

def download_image(url):
    with request.urlopen(url) as resp:
        buffer = resp.read()
    stream = BytesIO(buffer)
    img = Image.open(stream)
    return img


def prepare_image(img, target_size):
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img = img.resize(target_size, Image.NEAREST)
    return img


def predict(url):
    img1 = download_image(url)
    resized_img = prepare_image(img1, (150,150))

    x = image.img_to_array(resized_img)
    x = x.reshape((1,) + x.shape)

    datagen = ImageDataGenerator(rescale=1./255)

    X = datagen.flow(x).next()

    interpreter.set_tensor(input_index, X)
    interpreter.invoke()

    preds = interpreter.get_tensor(output_index)

    return {"result": str(preds[0][0])}


    # X = preprocessor.from_url(url)

    # interpreter.set_tensor(input_index, X)
    # interpreter.invoke()

    # preds = interpreter.get_tensor(output_index)

    # classes = ["dress", "hat", "longsleeve", "outwear", "pants", "shirt", "shoes", "shorts", "skirt", "t-shirt"]
    # float_predictions = preds[0].tolist()
    # return dict(zip(classes,  float_predictions))


def lambda_handler(event, context):
    print('Request event:', event)
    # img_url = 'http://bit.ly/mlbookcamp-pants'
    img_url = event['url']
    result = predict(img_url)
    return result




