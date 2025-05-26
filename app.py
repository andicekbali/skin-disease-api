from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
from PIL import Image
import os
from werkzeug.utils import secure_filename
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

app = Flask(__name__)
CORS(app)  # Mengizinkan akses dari Flutter

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# 1. Load dataset dan buat model KNN
df = pd.read_csv("skin_dataset.csv")

# Pastikan dataset dalam rentang 0-1 (hanya normalisasi jika perlu)
if df[["R", "G", "B"]].max().max() > 1:
    print("⚠️ Dataset terdeteksi dalam rentang 0-255, melakukan normalisasi ke 0-1...")
    df[["R", "G", "B"]] = df[["R", "G", "B"]] / 255.0

X = df[["R", "G", "B"]].values
y = df["Label"].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

@app.route("/")
def home():
    return "API KNN untuk Klasifikasi Kulit"

# 2. Endpoint untuk klasifikasi berdasarkan RGB
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json
        if not all(k in data for k in ["R", "G", "B"]):
            return jsonify({"error": "Data RGB tidak lengkap"}), 400

        # Normalisasi input RGB (0-255) ke skala 0-1
        r, g, b = data["R"] / 255.0, data["G"] / 255.0, data["B"] / 255.0
        prediction = knn.predict(np.array([r, g, b]).reshape(1, -1))

        return jsonify({"prediction": prediction[0]})

    except OSError:
        print("⚠️ Client menutup koneksi sebelum menerima respons")
        return "", 204  # Response kosong untuk menghindari error
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# 3. Endpoint untuk klasifikasi dari gambar
@app.route("/predict_image", methods=["POST"])
def predict_image():
    try:
        if "file" not in request.files:
            return jsonify({"error": "Tidak ada file yang diunggah"}), 400

        file = request.files["file"]
        filename = secure_filename(file.filename)
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)

        # Membuka gambar dan menghitung rata-rata nilai RGB
        image = Image.open(filepath).convert("RGB")
        np_image = np.array(image)
        r_avg = np.mean(np_image[:, :, 0])
        g_avg = np.mean(np_image[:, :, 1])
        b_avg = np.mean(np_image[:, :, 2])

        # Normalisasi ke 0-1 sebelum diproses model
        r_avg_norm = r_avg / 255.0
        g_avg_norm = g_avg / 255.0
        b_avg_norm = b_avg / 255.0

        # Prediksi dengan KNN dan dapatkan probabilitas
        prediction = knn.predict(np.array([r_avg_norm, g_avg_norm, b_avg_norm]).reshape(1, -1))
        probabilities = knn.predict_proba(np.array([r_avg_norm, g_avg_norm, b_avg_norm]).reshape(1, -1))
        
        # Ambil confidence score (probabilitas tertinggi)
        confidence = np.max(probabilities) * 100  # Konversi ke persentase

        return jsonify({
            "prediction": prediction[0],
            "confidence": round(confidence, 2),  # Nilai confidence dengan 2 desimal
            "average_rgb": {"R": round(r_avg), "G": round(g_avg), "B": round(b_avg)}
        })

    except OSError:
        print("⚠️ Client menutup koneksi sebelum menerima respons")
        return "", 204  # Response kosong untuk menghindari error
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
