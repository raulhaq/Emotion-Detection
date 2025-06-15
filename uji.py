import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt

# Muat model yang telah disimpan
model = tf.keras.models.load_model('emotion_detection_vgg.h5')

# Label kelas untuk emosi (pastikan urutannya sesuai dengan saat model dilatih)
emotion_labels = ['angry', 'fear', 'happy', 'surprise', 'disgust', 'neutral', 'sad']

# Fungsi untuk mempersiapkan gambar
def prepare_image(img_path, target_size=(48, 48)):
    img = image.load_img(img_path, target_size=target_size)  # Load gambar dan resize
    img_array = image.img_to_array(img)  # Convert gambar menjadi array
    img_array = np.expand_dims(img_array, axis=0)  # Tambahkan dimensi batch (1,)
    img_array = img_array / 255.0  # Normalisasi
    return img_array

# Pilih gambar yang ingin diuji
img_path = 'image0040686.jpg'  # Ganti dengan path gambar yang ingin diuji

# Persiapkan gambar
img = prepare_image(img_path)

# Prediksi dengan model
predictions = model.predict(img)
predicted_class = np.argmax(predictions, axis=1)  # Ambil kelas dengan probabilitas tertinggi

# Tampilkan hasil
predicted_emotion = emotion_labels[predicted_class[0]]

# Visualisasikan gambar dan prediksi
img_to_show = image.load_img(img_path, target_size=(48, 48))
plt.imshow(img_to_show)
plt.title(f'Predicted Emotion: {predicted_emotion}')
plt.show()

print(f"Predicted Emotion: {predicted_emotion}")
