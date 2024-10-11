# Proyek Akhir : Klasifikasi Gambar by Dicoding Indonesia
Nama : Angger Binuko Paksi
Email : angger.binuko@gmail.com

## Kriteria submission yang harus dipenuhi:
1. Dataset yang dipakai haruslah dataset berikut : rockpaperscissors, atau gunakan link ini pada wget command: https://github.com/dicodingacademy/assets/releases/download/release/rockpaperscissors.zip.
2. Dataset harus dibagi menjadi train set dan validation set.
3. Ukuran validation set harus 40% dari total dataset (data training memiliki 1314 sampel, dan data validasi sebanyak 874 sampel).
4. Harus mengimplementasikan augmentasi gambar.
5. Menggunakan image data generator.
6. Model harus menggunakan model sequential.
7. Pelatihan model tidak melebihi waktu 30 menit.
8. Program dikerjakan pada Google Colaboratory.
9. Akurasi dari model minimal 85%.
10. Dapat memprediksi gambar yang diunggah ke Colab.

## 1. Install wget
Pada langkah ini, jika `wget` belum terinstal, perintah `!pip install wget` digunakan untuk menginstalnya. `wget` diperlukan untuk mengunduh dataset dari URL yang diberikan.

```python
!pip install wget
```

## 2. Import Libraries
Library yang dibutuhkan diimpor, termasuk TensorFlow untuk membuat model, `ImageDataGenerator` untuk preprocessing data, serta `matplotlib` untuk menampilkan gambar yang diunggah.

```python
import os
import wget
import zipfile
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
import numpy as np
from google.colab import files
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
import time
```

## 3. Download Dataset
Dataset diunduh menggunakan `wget` dari URL yang diberikan dan disimpan sebagai file `rockpaperscissors.zip`.

```python
url = 'https://github.com/dicodingacademy/assets/releases/download/release/rockpaperscissors.zip'
dataset_path = 'rockpaperscissors.zip'
wget.download(url, dataset_path)
```

## 4. Extract Dataset
File zip diekstrak ke folder `dataset` menggunakan modul `zipfile`.

```python
with zipfile.ZipFile(dataset_path, 'r') as zip_ref:
    zip_ref.extractall('dataset')
```

## 5. Define Directories
Path ke direktori yang berisi gambar didefinisikan. Semua gambar berada di direktori `rps-cv-images`.

```python
base_dir = 'dataset/rockpaperscissors/rps-cv-images'
train_dir = os.path.join(base_dir)
```

## 6. Preprocess Data dengan ImageDataGenerator dan Augmentasi
`ImageDataGenerator` digunakan untuk mengatur augmentasi gambar dan membagi dataset menjadi 60% training dan 40% validation. Augmentasi meliputi rotasi, pergeseran, zoom, dan pembalikan horizontal.

```python
data_gen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.4,  # 40% untuk validasi
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)
```

## 7. Load MobileNetV2 Model
Model `MobileNetV2` digunakan dengan transfer learning. Layer yang sudah dilatih dibekukan (tidak dilatih ulang) dengan `trainable=False`.

```python
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(150, 150, 3))
base_model.trainable = False
```

## 8. Build Model
Layer tambahan ditambahkan di atas `MobileNetV2` untuk klasifikasi gambar menjadi tiga kelas: Rock, Paper, Scissors.

```python
x = base_model.output
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dense(128, activation='relu')(x)
predictions = tf.keras.layers.Dense(3, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)
```

## 9. Compile Model
Model dikompilasi dengan optimizer `Adam`, loss function `categorical_crossentropy`, dan metrik `accuracy`.

```python
model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)
```

## 10. Define Callbacks
Callback seperti `EarlyStopping`, `ModelCheckpoint`, dan `ReduceLROnPlateau` digunakan untuk menghentikan pelatihan lebih awal, menyimpan model terbaik, dan mengurangi laju pembelajaran jika diperlukan.

```python
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

model_checkpoint = ModelCheckpoint(
    'best_model.keras',
    monitor='val_accuracy',
    save_best_only=True,
    mode='max'
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.2,
    patience=3,
    min_lr=1e-6
)
```

## 11. Train the Model dan Hitung Waktu Pelatihan
Model dilatih dengan dataset training dan validasi selama 20 epoch, dan waktu pelatihan dicatat menggunakan fungsi `time`. 

```python
start_time = time.time()  # Catat waktu mulai

history = model.fit(
    train_data,
    steps_per_epoch=len(train_data),
    validation_data=val_data,
    epochs=20,
    verbose=1,
    callbacks=[early_stopping, model_checkpoint, reduce_lr]
)

end_time = time.time()  # Catat waktu selesai
training_time = end_time - start_time  # Hitung waktu pelatihan

# Tampilkan waktu pelatihan
print(f'Time taken to train the model: {training_time:.2f} seconds')
```

## 12. Evaluate Model
Setelah pelatihan, model dievaluasi menggunakan dataset validasi untuk mengetahui akurasi model.

```python
val_loss, val_accuracy = model.evaluate(val_data)
print(f'Validation accuracy: {val_accuracy * 100:.2f}%')
```

## 13. Prediksi Gambar Baru
Mengunggah gambar, melakukan preprocessing pada gambar, kemudian memprediksi kelas gambar tersebut (Rock, Paper, atau Scissors) menggunakan model yang sudah dilatih. Hasil prediksi dan confidence ditampilkan.

```python
uploaded = files.upload()

for file_name in uploaded.keys():
    img_path = file_name
    img = image.load_img(img_path, target_size=(150, 150))
    
    plt.imshow(img)
    plt.axis('off')
    plt.show()

    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0

    prediction = model.predict(img_array)
    class_indices = ['Rock', 'Paper', 'Scissors']
    predicted_class = class_indices[np.argmax(prediction)]
    confidence = np.max(prediction) * 100

    print(f'The uploaded image is predicted as: {predicted_class}')
    print(f'Prediction confidence: {confidence:.2f}%')
```

