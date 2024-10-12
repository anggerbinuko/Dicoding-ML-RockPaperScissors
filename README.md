# Proyek Akhir: Klasifikasi Gambar (Rock, Paper, Scissors) by Dicoding Indonesia
* Nama : Angger Binuko Paksi
* Email : angger.binuko@gmail.com

## Deskripsi Proyek
Proyek ini adalah implementasi jaringan saraf tiruan (Neural Network) menggunakan TensorFlow untuk mengenali bentuk tangan yang membentuk gunting, batu, atau kertas. Model dilatih menggunakan dataset `rockpaperscissors` dan mampu memprediksi gambar yang diunggah ke Google Colaboratory.

## Kebutuhan Proyek
- Google Colaboratory untuk menjalankan kode.
- Dataset: [rockpaperscissors](https://github.com/dicodingacademy/assets/releases/download/release/rockpaperscissors.zip).
- TensorFlow dan Keras untuk membangun model neural network.
- Matplotlib untuk visualisasi gambar.

## Tahapan Proyek

### 1. Persiapan Dataset
Dataset digunakan untuk melatih model, terdiri dari gambar batu, kertas, dan gunting. File dataset diunduh dari URL dan diekstraksi.

```bash
!wget https://github.com/dicodingacademy/assets/releases/download/release/rockpaperscissors.zip
!unzip -q rockpaperscissors.zip
```

- Dataset dibagi menjadi dua bagian: 60% untuk **training set** dan 40% untuk **validation set**.
- Augmentasi gambar diterapkan untuk memperbanyak variasi data pelatihan.

```python
train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.4,  # 40% untuk validasi
    rotation_range=20,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)
```

### 2. Membangun Model Neural Network
Model dibangun menggunakan arsitektur **Sequential** dengan beberapa lapisan konvolusi dan pooling.

```python
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    layers.MaxPooling2D(2, 2),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),
    layers.Flatten(),
    layers.Dense(512, activation='relu'),
    layers.Dense(3, activation='softmax')  # 3 kelas: rock, paper, scissors
])
```

- Optimizer yang digunakan adalah **Adam** dengan learning rate 0.001.
- Loss function yang dipakai adalah **categorical_crossentropy** untuk klasifikasi multi kelas.

### 3. Pelatihan Model
Model dilatih dengan training set dan dievaluasi menggunakan validation set. Callback diterapkan untuk early stopping dan pengurangan learning rate.

```python
callbacks = [
    EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=0.0001)
]

history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // 32,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // 32,
    epochs=30,
    callbacks=callbacks
)
```

### 4. Memprediksi Gambar yang Diunggah
Setelah model dilatih, pengguna dapat mengunggah gambar dan melakukan prediksi. Gambar yang diunggah akan ditampilkan, diikuti oleh hasil prediksi beserta akurasinya.

```python
from google.colab import files
import matplotlib.pyplot as plt

uploaded = files.upload()

for img_name in uploaded.keys():
    # Prediksi gambar
    predicted_class, predicted_prob, img = predict_image(img_name)
    
    # Menampilkan gambar
    plt.imshow(img)
    plt.axis('off')
    plt.show()
    
    # Menampilkan hasil prediksi
    print(f"Prediction: {predicted_class} ({predicted_prob:.2f}%)")
```

## Struktur Direktori
```
root/
│
├── rockpaperscissors/                              # Dataset yang digunakan
│   └── rps-cv-images/                              #  ambar batu, kertas, dan gunting
├── README.md                                       # File ini
└── Proyek_Akhir_Klasifikasi_Gambar.ipynb           # Notebook dengan kode utama
```

## Cara Menjalankan Proyek
1. Jalankan kode di Google Colab.
2. Eksekusi sel untuk mendownload dan menyiapkan dataset.
3. Latih model dengan menjalankan sel pelatihan.
4. Unggah gambar untuk melakukan prediksi dan lihat hasilnya.

## Spesifikasi Model
- **Input Image Size:** 150x150 pixel.
- **Jumlah Kelas:** 3 (rock, paper, scissors).
- **Optimizer:** Adam.
- **Loss Function:** Categorical Crossentropy.
- **Akurasi Minimal:** 85%.

## Lisensi
Proyek ini dibuat sebagai bagian dari pembelajaran TensorFlow dan Keras pada platform Dicoding. 


