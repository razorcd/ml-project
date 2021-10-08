from flask import Flask
from flask import request
from flask import jsonify
import pickle

app = Flask("churn")

C = 2
input_file = f"model{C}.bin"
dv_file = f"dv.bin"

with open(input_file, "rb") as f_in:
    model = pickle.load(f_in)
with open(dv_file, "rb") as f_in:
    dv = pickle.load(f_in)


def predict(data, dv, model):
    X = dv.transform(data)
    y_pred = model.predict_proba(X)[:, 1]
    return y_pred


@app.route("/predict", methods=["POST"])
def predict_churn():
    app.logger.info("Request for %s", request.data)
    customer = request.get_json()["customers"]
    
    y_predictions = predict(customer, dv, model)

    results = {"predictions": []}
    for y in y_predictions:
        churn = y > 0.5
        results["predictions"].append({
            "churn_probability": y,
            "churn": bool(churn)
        })
    
    app.logger.info("Response for %s", str(results))
    return jsonify(results)


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=9697)    


# curl -X POST -d "{\"customers\":[{\"contract\": \"two_year\", \"tenure\": 12, \"monthlycharges\": 10}]}" -H 'Content-Type: application/json' localhost:9697/predict
