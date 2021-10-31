from flask import Flask
from flask import request
from flask import jsonify
import pickle
import xgboost as xgb

app = Flask("low_income")

eta = 0.4
max_depth = 4
input_file = f'model_xg_{eta}_{max_depth}.bin'

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
def predict_low_income():
    app.logger.info("Received Request: %s", request.data)
    try:
        data = request.get_json()["data"]
    except Exception as e:
        return build_error_message("Parsing request data error: "+str(e)+". Request payload: "+str(request.data), "400 Bad Rquest"), 400
    
    y_predictions, X_predictions = predict(data, dv, model)

    results = {"predictions": []}
    for y in y_predictions:
        low_income = y > 0.5
        results["predictions"].append({
            "low_income_probability": str(round(y,2)),
            "low_income": bool(low_income)
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
categorical_columns:
    'workclass', # Private, Self-emp-not-inc, Self-emp-inc, Federal-gov, Local-gov, State-gov, Without-pay, Never-worked.
    'education', # Bachelors, Some-college, 11th, HS-grad, Prof-school, Assoc-acdm, Assoc-voc, 9th, 7th-8th, 12th, Masters, 1st-4th, 10th, Doctorate, 5th-6th, Preschool.
    'marital-status', # Married-civ-spouse, Divorced, Never-married, Separated, Widowed, Married-spouse-absent, Married-AF-spouse.
    'occupation', # Tech-support, Craft-repair, Other-service, Sales, Exec-managerial, Prof-specialty, Handlers-cleaners, Machine-op-inspct, Adm-clerical, Farming-fishing, Transport-moving, Priv-house-serv, Protective-serv, Armed-Forces.
    'relationship', # Wife, Own-child, Husband, Not-in-family, Other-relative, Unmarried.
    'race', # White, Asian-Pac-Islander, Amer-Indian-Eskimo, Other, Black.
    'sex', # Female, Male.

numerical_columns:
    'age', # int
    'education-num', # int
    'hours-per-week', # int
"""

# Request Example:
"""
curl -X POST \
  http://localhost:9696/predict \
  -H 'Content-Type: application/json' \
  -d '{
	"data":[
		{
			"workclass": "Private",
			 "education": "HS-grad",
			 "marital-status": "Divorced",
			 "occupation": "Machine-op-inspct",
			 "relationship": "Not-in-family",
			 "race": "White",
			 "sex": "Female",
			 "age": 59,
			 "education-num": 9,
			 "hours-per-week": 40
		},
		{
			"workclass": "Private",
			 "education": "Bachelors",
			 "marital-status": "Never-married",
			 "occupation": "Exec-managerial",
			 "relationship": "Not-in-family",
			 "race": "White",
			 "sex": "Male",
			 "age": 48,
			 "education-num": 20,
			 "hours-per-week": 45
		}
	]
}'
"""