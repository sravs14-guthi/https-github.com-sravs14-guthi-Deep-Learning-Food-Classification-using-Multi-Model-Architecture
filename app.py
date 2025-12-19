from flask import Flask, render_template, request, jsonify, url_for
from werkzeug.utils import secure_filename
import os, json, random, datetime

# --- CONFIG ---
app = Flask(__name__, static_folder="static", template_folder="templates")
UPLOAD_FOLDER = os.path.join(app.static_folder, "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
ALLOWED_EXT = {"png", "jpg", "jpeg", "gif"}

# load nutrition data
CLASS_JSON = "class.json"
with open(CLASS_JSON, "r", encoding="utf-8") as f:
    nutrition_data = json.load(f)

# attempt to locate model files (but don't load heavy h5 files)
def find_models(folder):
    if not os.path.isdir(folder):
        return []
    models = []
    for fn in sorted(os.listdir(folder)):
        if fn.lower().endswith((".h5", ".keras", ".hdf5")):
            models.append(os.path.join(folder, fn))
    return models

model_index = {
    "custom_model": find_models("custom_models"),
    "resnet_model": find_models("resnet_models"),
    "vgg_model": find_models("vgg_models"),
}

# helper: allowed file
def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXT

# Mock evaluation stats (you can replace with real per-model stats)
default_metrics = {
    "accuracy": 0.9,
    "precision": 0.75,
    "recall": 0.8,
    "f1_score": 0.77,
    "confusion_matrix": [[160, 8, 2], [13, 131, 26], [7, 5, 49]]
}

# ROUTES
@app.route("/")
def index():
    # classes list sorted
    classes = sorted(list(nutrition_data.keys()))
    # pass sample thumbnail for initial display
    sample_image = url_for("static", filename="images/sample_food.jpg") if os.path.exists(os.path.join(app.static_folder, "images", "sample_food.jpg")) else None
    return render_template("index.html",
                           classes=classes,
                           nutrition=nutrition_data,
                           sample_image=sample_image,
                           model_index=model_index)

@app.route("/predict", methods=["POST"])
def predict():
    # check file
    file = request.files.get("file")
    selected_class = request.form.get("selected_class", "")
    model_type = request.form.get("model_type", "custom_model")

    if not file or file.filename == "":
        return jsonify(success=False, error="No file uploaded.")

    if not allowed_file(file.filename):
        return jsonify(success=False, error="File type not allowed.")

    # save file
    filename = datetime.datetime.now().strftime("%Y%m%d%H%M%S_") + secure_filename(file.filename)
    saved_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(saved_path)
    image_url = url_for("static", filename=f"uploads/{filename}")

    # Choose model used (if any exist)
    available = model_index.get(model_type, [])
    if available:
        # pick first model file name for display
        model_used = os.path.basename(available[0])
        # NOTE: For reliability in this template we won't attempt heavy keras.load_model here.
        # If you want to run real inference, replace the block below with proper model loading and preprocessing.
        # For now we'll mock predictions but keep the hook.
        predicted_class = random.choice([selected_class] + list(nutrition_data.keys()))
        confidence = round(random.uniform(50.0, 99.9), 2)
        metrics = default_metrics
    else:
        # no model file found: return safe mock (so UI always shows something)
        model_used = "(no model file)"
        predicted_class = selected_class or random.choice(list(nutrition_data.keys()))
        confidence = round(random.uniform(55.0, 98.0), 2)
        metrics = default_metrics

    # build response
    resp = {
        "success": True,
        "image_url": image_url,
        "predicted_class": predicted_class,
        "selected_class": selected_class,
        "model_used": model_used,
        "confidence": f"{confidence}",
        "accuracy": metrics.get("accuracy"),
        "precision": metrics.get("precision"),
        "recall": metrics.get("recall"),
        "f1_score": metrics.get("f1_score"),
        "confusion_matrix": metrics.get("confusion_matrix"),
    }
    return jsonify(resp)

if __name__ == "__main__":
    app.run(debug=True)