from flask import Flask, render_template, request
from model_loader import predict
import os

app = Flask(__name__)

# Limit uploads to 5 MB for safety
app.config["MAX_CONTENT_LENGTH"] = 5 * 1024 * 1024

@app.route("/", methods=["GET"])
def home():
    # Show the upload page with no result yet
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict_route():
    if "image" not in request.files:
        return render_template("index.html", error="No file uploaded.")

    file = request.files["image"]
    if file.filename == "":
        return render_template("index.html", error="No file selected.")

    try:
        result = predict(file)
        return render_template(
            "index.html",
            prediction=result["predicted_class"],
            confidence=result["confidence"],
            is_confident=result["is_confident"],
            all_probs=result["all_probs"]
        )
    except Exception as e:
        return render_template("index.html", error=f"Something went wrong: {str(e)}")

if __name__ == "__main__":
    # For local testing only; Render uses Gunicorn instead
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)