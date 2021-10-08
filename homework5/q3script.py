import pickle

C = 1
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


data = {
    "contract": "two_year", 
    "tenure": 12, 
    "monthlycharges": 19.7
}

result = predict(data, dv, model)
print("probability that this customer is churning:",result)
