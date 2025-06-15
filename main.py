import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

# Hyperparameters
IMG_HEIGHT, IMG_WIDTH = 48, 48  # Ukuran gambar input yang diinginkan
BATCH_SIZE = 32
EPOCHS = 100  # Jumlah epoch yang lebih banyak untuk transfer learning
LEARNING_RATE = 0.01

# Direktori untuk dataset pelatihan dan pengujian
train_dir = "train"  # Ganti dengan direktori dataset pelatihan Anda
test_dir = "test"    # Ganti dengan direktori dataset pengujian Anda

# Data Augmentation dan Preprocessing untuk pelatihan
train_datagen = ImageDataGenerator(
    rescale=1./255,  # Normalisasi gambar
    rotation_range=20,  # Putar gambar dalam rentang 20 derajat
    width_shift_range=0.2,  # Geser gambar secara horizontal
    height_shift_range=0.2,  # Geser gambar secara vertikal
    shear_range=0.2,  # Distorsi gambar secara acak
    zoom_range=0.2,  # Zoom gambar secara acak
    horizontal_flip=True,  # Putar gambar secara horizontal
    fill_mode='nearest'  # Isi piksel yang hilang akibat transformasi
)

# Preprocessing untuk data uji
test_datagen = ImageDataGenerator(rescale=1./255)  # Hanya normalisasi tanpa augmentasi

# Data Generators untuk pelatihan dan pengujian
train_generator = train_datagen.flow_from_directory(
    train_dir,  # Direktori dataset pelatihan
    target_size=(IMG_HEIGHT, IMG_WIDTH),  # Ubah ukuran gambar menjadi 48x48
    batch_size=BATCH_SIZE,  # Ukuran batch
    class_mode='categorical'  # Mode klasifikasi multi-kelas
)

test_generator = test_datagen.flow_from_directory(
    test_dir,  # Direktori dataset pengujian
    target_size=(IMG_HEIGHT, IMG_WIDTH),  # Ubah ukuran gambar menjadi 48x48
    batch_size=BATCH_SIZE,  # Ukuran batch
    class_mode='categorical'  # Mode klasifikasi multi-kelas
)

# Memuat model ResNet50 tanpa layer klasifikasi (top layer)
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(IMG_HEIGHT, IMG_WIDTH, 3))

# Membekukan layer-layer awal pada ResNet50 untuk mempertahankan fitur yang sudah dipelajari
for layer in base_model.layers:
    layer.trainable = False

# Menambahkan layer custom di atas model ResNet50
x = base_model.output
x = GlobalAveragePooling2D()(x)  # Menggunakan GlobalAveragePooling untuk meratakan output
x = Dense(256, activation='relu')(x)  # Menambahkan fully connected layer
x = Dropout(0.5)(x)  # Regularisasi untuk menghindari overfitting
predictions = Dense(train_generator.num_classes, activation='softmax')(x)  # Output layer sesuai jumlah kelas

# Membangun model lengkap
model = Model(inputs=base_model.input, outputs=predictions)

# Kompilasi Model
model.compile(
    optimizer=Adam(learning_rate=LEARNING_RATE),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Menggunakan EarlyStopping untuk menghindari overfitting
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Melatih Model
history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=test_generator,
    callbacks=[early_stopping]  # Menggunakan early stopping
)

# Menyimpan Model setelah pelatihan
model.save("emotion_detection_resnet50.h5")  # Simpan model ke file .h5

# Evaluasi Model pada Data Pengujian
loss, accuracy = model.evaluate(test_generator)
print(f"Test Loss: {loss:.4f}")
print(f"Test Accuracy: {accuracy:.4f}")
