# Machine Learning / AI Project
<p align="center">
  <img src="https://camo.githubusercontent.com/0562f16a4ae7e35dae6087bf8b7805fb7e664a9e7e20ae6d163d94e56b94f32d/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f707974686f6e2d3336373041303f7374796c653d666f722d7468652d6261646765266c6f676f3d707974686f6e266c6f676f436f6c6f723d666664643534">
  <img src="https://camo.githubusercontent.com/05cab52d05663cecbe47a23ca71075ba81b9080dd50561d0f76eb46e902cfef8/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f70616e6461732d2532333135303435382e7376673f7374796c653d666f722d7468652d6261646765266c6f676f3d70616e646173266c6f676f436f6c6f723d7768697465">
  <img src="https://camo.githubusercontent.com/ac5fa240dbb610e4e9aa6d501afef4a8e8c72a3ce067010d83a832b04dc81177/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f54656e736f72466c6f772d2532334646364630302e7376673f7374796c653d666f722d7468652d6261646765266c6f676f3d54656e736f72466c6f77266c6f676f436f6c6f723d7768697465">
  <img src="https://camo.githubusercontent.com/6c1504bc94a0bd93c60f42b1f59baa44de2d68ecffdabd61fe8d2dbe12cd3374/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f4b657261732d2532334430303030302e7376673f7374796c653d666f722d7468652d6261646765266c6f676f3d4b65726173266c6f676f436f6c6f723d7768697465">
</p>

Ini adalah repository untuk contoh struktural yang bisa dipakai untuk melakukan dokumentasi Project Massive anda.

## Teams
- **Dewi Sinta Febriani** (Data Engineer)
- **Andreas M Simanullang** (Machine Learning Engineer)
- **Fajar Ramadhan (M. Arifian)** (Machine Learning Ops)
- **Wahyu Cahyono P** (Design Researcher)
- **Mega Lela Haloho** (Data Engineer)

## Idea Background

### 1. Theme
**Tema:** Pariwisata

### 2. Problem
**Masalah:** Sulitnya mendapat rekomendasi wisata yang cocok untuk mengefisienkan waktu dan uang

### 3. Solution
**Solusi:** Membuat model Machine Learning untuk memberi rekomendasi yang sesuai. Model ini dilatih menggunakan berbagai algoritma dan dataset untuk meningkatkan tingkat kepuasan prediksinya.Data yang dikumpulkan akan diolah dan dibersihkan untuk memastikan kualitas input ke dalam model. Proses evaluasi akan melibatkan berbagai metrik untuk memastikan model memenuhi standar yang diharapkan.
  
## Dataset and Algorithm

### 1. Dataset

#### Data Collection
Kami menemukan data kami di [Kaggle](https://www.kaggle.com/).

#### Data Cleaning
Kami menggunakan pandas untuk membersihkan data. Berikut adalah contoh data sebelum dan sesudah dibersihkan:

**Contoh Data Belum Dibersihkan:**
```
| Place_Name            | Category           | City        | Price | Rating | Time_Minutes | Coordinate                                 | Lat        | Long         |
|-----------------------|--------------------|-------------|-------|--------|--------------|-------------------------------------------|------------|--------------|
| Kawasan Kuliner BSM   | Pusat Perbelanjaan | Jakarta     | 0     | 4.6    |              | {'lat': -6.184258799999999, 'lng': 106.824033} | -6.1842588 | 106.824033   |
| Taman Pintar Yogyakarta | Taman Hiburan    | Yogyakarta  | 6000  | 4.5    | 120          | {'lat': -7.800671500000001, 'lng': 110.3676551} | -7.8006715 | 110.3676551  |
| Pantai Congot         | Bahari             | Yogyakarta  | 3000  | 4.3    |              | {'lat': -7.907542500000001, 'lng': 110.0535658} | -7.9075425 | 110.0535658  |
| GunungTangkuban perahu | Cagar Alam       | Bandung     | 30000 | 4.5    |              | {'lat': -6.759637700000001, 'lng': 107.6097807} | -6.7596377 | 107.6097807  |
| Jalan Braga           | Budaya             | Bandung     | 0     | 4.7    |              | {'lat': -6.9150534, 'lng': 107.6089842}         | -6.9150534 | 107.6089842  |
```

**Contoh Data Setelah Dibersihkan:**
```
| Place_Name            | Category           | City        | Price | Rating | Time_Minutes | Coordinate                                 | Lat        | Long         |
|-----------------------|--------------------|-------------|-------|--------|--------------|-------------------------------------------|------------|--------------|
| GunungTangkuban perahu | Cagar Alam       | Bandung     | 30000 | 4.5    |              | {'lat': -6.759637700000001, 'lng': 107.6097807} | -6.7596377 | 107.6097807  |
| Jalan Braga           | Budaya             | Bandung     | 0     | 4.7    |              | {'lat': -6.9150534, 'lng': 107.6089842}         | -6.9150534 | 107.6089842  |
```

### 2. Algorithm

#### Framework
Kami menggunakan TensorFlow dan Keras untuk membangun dan melatih model.

#### Pembangunan Model
Berikut adalah spesifikasi model yang kami gunakan:
- **Epoch:** 100
- **Learning Rate:** 0.001
- **Batch Size:** 32

```python
import tensorflow as tf
from tensorflow import keras

model = keras.Sequential([
    keras.layers.Dense(64, activation='relu', input_shape=(input_shape,)),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(1)
])

model.compile(optimizer=tf.optimizers.Adam(learning_rate=0.001),
              loss='mean_squared_error',
              metrics=['root_mean_squared_error'])

history = model.fit(
    x=x_train,
    y=y_train,
    epochs=100,
    validation_data=(x_val, y_val),
    callbacks=[myCallback()]
)
```

**Output Training:**
```
Epoch 100/100
72/72 [==============================] - 0s 3ms/step - loss: 0.6521 - root_mean_squared_error: 0.3104 - val_loss: 0.7224 - val_root_mean_squared_error: 0.3550
```

#### Model Evaluation
Kami menggunakan metrik evaluasi berikut untuk menilai performa model:
- **Root Mean Squared Error (RMSE)**

## Prototype
![Screenshot 2024-05-22 091127](https://github.com/Andreas-23/Touristm_chatbot/assets/91534680/a0b0c90f-7016-4da1-b160-9805a9db362e)


## Integration
Chatbot dari Watson Assisstant diintegrasikan ke **Website**

## Deployment
Proses deploy dilakukan di **Watson Studio**

## Result
![Screenshot (781)](https://github.com/Andreas-23/Touristm_chatbot/assets/91534680/a7d559de-8a01-465a-a9be-0eeb4d34b9f0)

## Conclusion
Disesuaikan dengan kebutuhan atau bisa ditiru dari laporan dokumentasi massive.

---
