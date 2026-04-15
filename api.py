from flask import Flask, request, jsonify
import pandas as pd
import joblib

app = Flask(__name__)

model = joblib.load("model.pkl")

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    
    df = pd.DataFrame([data])
    df = pd.get_dummies(df)

    df = df.reindex(columns=model.feature_names_in_, fill_value=0)

    prob = model.predict_proba(df)[0][1]

    return jsonify({
        "risk_probability": float(prob)
    })

if __name__ == "__main__":
    app.run(debug=True)