import tensorflow as tf  # Tambahkan baris ini untuk mengimpor tensorflow
import numpy as np
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt

# Fungsi untuk memuat dan memproses gambar
def load_and_process_image(img_path, target_size=(48, 48)):
    # Memuat gambar
    img = image.load_img(img_path, target_size=target_size)
    # Mengubah gambar menjadi array dan melakukan normalisasi
    img_array = image.img_to_array(img) / 255.0  # Normalisasi
    # Menambahkan dimensi batch (1, 48, 48, 3)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Ganti dengan path gambar yang ingin diuji
img_path = "ffhq_997.png"

# Memproses gambar
img_array = load_and_process_image(img_path)

# Memuat model yang telah dilatih
model = tf.keras.models.load_model("emotion_detection_resnet50.h5")

# Melakukan prediksi
predictions = model.predict(img_array)

# Menampilkan hasil prediksi
class_labels = ['angry', 'fear', 'happy', 'surprise', 'disgust', 'neutral', 'sad']  # Sesuaikan dengan label kelas Anda
predicted_class = class_labels[np.argmax(predictions)]

# Menampilkan gambar dan prediksi
img = image.load_img(img_path, target_size=(48, 48))
plt.imshow(img)
plt.title(f"Predicted: {predicted_class}")
plt.show()

# Menampilkan nilai probabilitas untuk setiap kelas
print("Predictions (probabilities for each class):")
for i, label in enumerate(class_labels):
    print(f"{label}: {predictions[0][i]:.4f}")
