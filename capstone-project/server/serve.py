from flask import Flask
from flask import request
from flask import jsonify
import pickle
import xgboost as xgb

app = Flask("Berlin rent")

eta = 0.6
max_depth = 20
input_file = f'model_xg_{eta}_{max_depth}.bin'

neighbourhoods = ['spandau', 'weißensee', 'mitte', 'kreuzberg', 'tiergarten', 'köpenick', 'marzahn', 'hohenschönhausen', 'hellersdorf', 'prenzlauer_berg', 'pankow', 'charlottenburg', 'tempelhof', 'neukölln', 'wilmersdorf', 'wedding', 'friedrichshain', 'reinickendorf', 'treptow', 'schöneberg', 'lichtenberg', 'steglitz', 'zehlendorf']
heatings = ['low', 'normal', 'high']


with open(input_file, "rb") as f_in:
    dv, model = pickle.load(f_in)


def predict(data_dict, dv, model):
    X = dv.transform(data_dict)
    features = dv.get_feature_names()
    dval = xgb.DMatrix(X, feature_names=features)
    y_pred = model.predict(dval)
    return y_pred, X 

def build_error_message(message, status_code):
    return jsonify({"error": str(message), "statis": str(status_code)})

@app.route("/predict", methods=["POST"])
def predict_endpoint():
    app.logger.info("Received Request: %s", request.data)
    try:
        data = request.get_json()["data"]
    except Exception as e:
        return build_error_message("Parsing request data error: "+str(e)+". Request payload: "+str(request.data), "400 Bad Rquest"), 400


    for d in data:
        d["neighbourhood"] = d["neighbourhood"].lower()
        d["heating"] = d["heating"].lower()
        if d["neighbourhood"] not in neighbourhoods: return build_error_message("Unknown neighbourhood: "+str(d["neighbourhood"]), "422 Unprocessable Entity"), 422
        if d["heating"] not in heatings: return build_error_message("Unknown heating: "+str(d["heating"]), "422 Unprocessable Entity"), 422

    y_predictions, X_predictions = predict(data, dv, model)

    results = {"prediction": []}
    for y in y_predictions:
        results["prediction"].append({
            "baseRent": int(y)
        })
    
    app.logger.info("Sending Response: %s", str(results))
    print(results)
    json_response = ""
    try:
         json_response = jsonify(results)
    except Exception as e:
        return build_error_message("Parsing prediction result error: "+str(e), "422 Unprocessable Entity"), 422
    
    return json_response, 200


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=9696)    


#input columns:
"""
categorical_columns = [
    'neighbourhoods', # ['spandau', 'weißensee', 'mitte', 'kreuzberg', 'tiergarten', 'köpenick', 'marzahn', 'hohenschönhausen', 'hellersdorf', 'prenzlauer_berg', 'pankow', 'charlottenburg', 'tempelhof', 'neukölln', 'wilmersdorf', 'wedding', 'friedrichshain', 'reinickendorf', 'treptow', 'schöneberg', 'lichtenberg', 'steglitz', 'zehlendorf']
    'heating', # low, normal, high
]
numerical_columns = [
    'newlyConst', 
    'balcony', 
    'hasKitchen', 
    'cellar', 
    'baseRent',
    'livingSpace', 
    'lift', 
    'noRooms', 
    'garden', 
]
"""

# Request Example:
"""
curl -X POST \
  http://localhost:9696/predict \
  -H 'Content-Type: application/json' \
  -d '{
	"data":[
		{
			"neighbourhood": "friedrichshain",
			"heating": "normal",
			"newlyConst": false, 
		    "balcony": true, 
		    "hasKitchen": true, 
		    "cellar": true, 
		    "livingSpace": 62, 
		    "lift": false, 
		    "noRooms": 2, 
		    "garden": false
		},
		{
			"neighbourhood": "Steglitz",
			"heating": "normal",
			"newlyConst": false, 
		    "balcony": false, 
		    "hasKitchen": true, 
		    "cellar": true, 
		    "livingSpace":75, 
		    "lift": false, 
		    "noRooms": 3, 
		    "garden": true
		}
	]
}'
"""