# ipython
# import course9
# event = {'url': 'http://bit.ly/mlbookcamp-pants'}
# course9.lambda_handler(event, None)


# Load TensorFlow Lite model
# !pip install --extra-index-url https://google-coral.github.io/py-repo/ tflite_runtime

# import tensorflow.lite as tflite
import tflite_runtime.interpreter as tflite
# from tflite_runtime.interpreter import Interpreter
# from tensorflow.keras.applications.xception import preprocess_input

interpreter = tflite.Interpreter(model_path='clothing-model.tflite')
interpreter.allocate_tensors()

print("Input details:", interpreter.get_input_details())
input_index = interpreter.get_input_details()[0]['index']

print("Output details:", interpreter.get_output_details())
output_index = interpreter.get_output_details()[0]['index']

# LOAD IMG:
# from tensorflow.keras.preprocessing.image import load_img
# import numpy as np
# img = load_img('pants.jpg', target_size=(150,150))
# x = np.array(img)
# X = np.array([x])
# X = preprocess_input(X)


# LOAD IMG:
# # # !pip install keras_image_helper
from keras_image_helper import create_preprocessor
preprocessor = create_preprocessor('xception', target_size=(150,150))
# X = preprocessor.from_path('pants.jpg')


def predict(url):
    X = preprocessor.from_url(url)

    interpreter.set_tensor(input_index, X)
    interpreter.invoke()

    preds = interpreter.get_tensor(output_index)

    classes = ["dress", "hat", "longsleeve", "outwear", "pants", "shirt", "shoes", "shorts", "skirt", "t-shirt"]
    float_predictions = preds[0].tolist()
    return dict(zip(classes,  float_predictions))


def lambda_handler(event, context):
    print('Request event:', event)
    # img_url = 'http://bit.ly/mlbookcamp-pants'
    img_url = event['url']
    result = predict(img_url)
    return result




