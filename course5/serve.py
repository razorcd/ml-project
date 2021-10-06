from flask import Flask
from flask import request
from flask import jsonify
import pickle

app = Flask("churn")

C = 1.0
input_file = f"model_C={C}.bin"

with open(input_file, "rb") as f_in:
    dv, model = pickle.load(f_in)


def predict(customers, dv, model):
    X = dv.transform(customers)
    y_pred = model.predict_proba(X)[:, 1]
    return y_pred


@app.route("/predict", methods=["POST"])
def predict_route():
    app.logger.info("Request for %s", request.data)
    customer = request.get_json()["customers"]
    
    y_predictions = predict(customer, dv, model)

    results = {"predictions": []}
    for y in y_predictions:
        churn = y > 0.5
        results["predictions"].append({
            "churn_probability": round(y,2),
            "churn": bool(churn)
        })
    
    app.logger.info("Response for %s", str(results))
    return jsonify(results)


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=9696)    



# "{\"gender\": \"female\", \"seniorcitizen\": 0, \"partner\": \"yes\", \"dependents\": \"no\", \"phoneservice\": \"no\", \"multiplelines\": \"no_phone_service\", \"internetservice\": \"dsl\", \"onlinesecurity\": \"no\", \"onlinebackup\": \"yes\", \"deviceprotection\": \"no\", \"techsupport\": \"no\", \"streamingtv\": \"no\", \"streamingmovies\": \"no\", \"contract\": \"month-to-month\", \"paperlessbilling\": \"yes\", \"paymentmethod\": \"electronic_check\", \"tenure\": 1, \"monthlycharges\": 29.85, \"totalcharges\": 29.85}"

# curl -X POST \
#   http://localhost:9696/predict \
#   -H 'Content-Type: application/json' \
#   -d '{
# 	"customers":[
# 		{
# 		  "gender": "female",
# 		  "seniorcitizen": 0,
# 		  "partner": "yes",
# 		  "dependents": "no",
# 		  "phoneservice": "no",
# 		  "multiplelines": "no_phone_service",
# 		  "internetservice": "dsl",
# 		  "onlinesecurity": "no",
# 		  "onlinebackup": "yes",
# 		  "deviceprotection": "no",
# 		  "techsupport": "no",
# 		  "streamingtv": "no",
# 		  "streamingmovies": "no",
# 		  "contract": "month-to-month",
# 		  "paperlessbilling": "yes",
# 		  "paymentmethod": "electronic_check",
# 		  "tenure": 1,
# 		  "monthlycharges": 29.85,
# 		  "totalcharges": 29.85
# 		}
# 	]
# }'