import os
import joblib
import pandas as pd
from flask import Flask, render_template, request

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "house_price_model.pkl")

FEATURES = [
    "OverallQual",
    "GrLivArea",
    "TotalBsmtSF",
    "GarageCars",
    "YearBuilt",
    "Neighborhood"
]

# Load model once at startup
model = joblib.load(MODEL_PATH)


@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    error = None

    if request.method == "POST":
        try:
            data = {
                "OverallQual": float(request.form["OverallQual"]),
                "GrLivArea": float(request.form["GrLivArea"]),
                "TotalBsmtSF": float(request.form["TotalBsmtSF"]),
                "GarageCars": float(request.form["GarageCars"]),
                "YearBuilt": float(request.form["YearBuilt"]),
                "Neighborhood": request.form["Neighborhood"].strip()
            }

            X = pd.DataFrame([data], columns=FEATURES)
            pred = model.predict(X)[0]
            prediction = f"${pred:,.2f}"

        except Exception as e:
            error = f"Error: {str(e)}"

    return render_template("index.html", prediction=prediction, error=error)


if __name__ == "__main__":
    # Render uses PORT env variable
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)