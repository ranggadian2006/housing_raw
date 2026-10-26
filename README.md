# Housing Raw: Prediksi Harga Rumah

Tugas **Deployment dan Monitoring** untuk model machine learning. Proyek ini membangun model regresi yang memprediksi harga median rumah (`median_house_value`) dari dataset California Housing, lalu men-deploy-nya sebagai API dan memantau kesehatan serta performanya setelah berjalan.

## Dataset

File `housing_raw.csv` berisi 20.656 baris dan 10 kolom (20.642 baris setelah 14 data duplikat dihapus).

| Kolom | Keterangan |
|---|---|
| `longitude`, `latitude` | Koordinat lokasi |
| `housing_median_age` | Umur median rumah |
| `total_rooms`, `total_bedrooms` | Jumlah kamar dan kamar tidur |
| `population`, `households` | Jumlah penduduk dan rumah tangga |
| `median_income` | Pendapatan median |
| `ocean_proximity` | Kedekatan dengan laut (kategori) |
| `median_house_value` | **Target** yang diprediksi |

Kolom `total_bedrooms` memiliki 207 nilai kosong, yang diisi dengan median data latih.

## Alur Pekerjaan

1. **Pra-pemrosesan data**: penanganan nilai kosong, outlier (metode IQR), standardisasi, penghapusan duplikat, dan encoding fitur kategori.
2. **Pembagian data**: 60% data latih (12.385 baris) dan 40% data uji (8.257 baris), `random_state=1`.
3. **Pelatihan tiga model**: Least Angle Regression (Lars), Linear Regression, dan Gradient Boosting Regressor.
4. **Evaluasi** dengan MAE, MSE, dan R².
5. **Deployment**: model dibungkus dalam satu *bundle* (model, imputer, encoder, urutan fitur) dan dilayani lewat API Flask, serta dapat dijalankan dengan Docker.
6. **Monitoring**: pemantauan latensi, data drift, dan performa model terhadap baseline.

## Hasil Evaluasi

Dievaluasi pada 8.257 data uji.

| Model | MAE | MSE | R² |
|---|---|---|---|
| Lars | 89.963 | 1,30 × 10¹⁰ | 0,015 |
| Linear Regression | 67.958 | 8,09 × 10⁹ | 0,386 |
| **Gradient Boosting Regressor** | **39.299** | **3,15 × 10⁹** | **0,761** |

- **MAE**: rata-rata selisih mutlak prediksi dan nilai asli (satuan sama dengan harga rumah).
- **MSE**: rata-rata kuadrat selisih, lebih peka terhadap kesalahan besar.
- **R²**: proporsi variasi harga yang dapat dijelaskan model (makin mendekati 1 makin baik).

Gradient Boosting Regressor menjadi model terbaik pada ketiga metrik, sehingga dipilih untuk deployment.

## Struktur Berkas

| Berkas | Fungsi |
|---|---|
| `housing_raw.csv` | Dataset mentah |
| `Latihan_Mandiri2.ipynb` | Notebook: pra-pemrosesan, pelatihan, evaluasi, dan pembuatan berkas deployment |
| `ml_utils.py` | Fungsi bersama: memuat bundle, preprocessing, prediksi, dan pencatatan log |
| `app.py` | API prediksi (Flask) |
| `monitoring.py` | Skrip monitoring: latensi, data drift, dan performa model |
| `model_bundle.joblib` | Bundle model hasil pelatihan |
| `reference_data.csv` | Sampel data latih sebagai pembanding untuk deteksi drift |
| `requirements.txt` | Daftar dependensi |
| `Dockerfile` | Konfigurasi container |

Berkas selain dataset dan notebook dihasilkan dengan menjalankan notebook dari atas ke bawah.

## Cara Menjalankan

Dikembangkan dengan Python 3.14.

```bash
# 1. Pasang dependensi
pip install -r requirements.txt

# 2. Jalankan notebook Latihan_Mandiri2.ipynb dari atas ke bawah
#    untuk menghasilkan model_bundle.joblib, reference_data.csv, ml_utils.py, app.py, dan monitoring.py

# 3. Jalankan API (mode pengembangan)
python app.py

# atau mode produksi
gunicorn -w 1 --threads 4 -b 0.0.0.0:8000 app:app
```

### Dengan Docker

```bash
docker build -t housing-api .
docker run -p 8000:8000 housing-api
```

## Endpoint API

| Metode | Endpoint | Fungsi |
|---|---|---|
| GET | `/health` | Cek status layanan dan versi model |
| POST | `/predict` | Prediksi harga rumah (satu atau beberapa baris, maksimal 1.000 per permintaan) |
| POST | `/feedback` | Mengirim nilai aktual agar performa model dapat dipantau |
| GET | `/metrics` | Statistik layanan: jumlah permintaan, error rate, dan latensi |

Input yang tidak valid (nilai di luar rentang atau kategori tidak dikenal) ditolak dengan HTTP 422 beserta pesan yang jelas.

### Contoh Penggunaan

```python
import requests

BASE_URL = "http://localhost:8000"

contoh = {
    "longitude": -122.23, "latitude": 37.88, "housing_median_age": 41,
    "total_rooms": 880, "total_bedrooms": 129, "population": 322,
    "households": 126, "median_income": 8.3252, "ocean_proximity": "NEAR BAY",
}

r = requests.post(f"{BASE_URL}/predict", json={"data": [contoh]})
print(r.json())
# {'model_version': '20261017-0820', 'prediction': [415235.59], 'request_ids': ['...']}

# Kirim nilai aktual untuk pemantauan performa
rid = r.json()["request_ids"][0]
requests.post(f"{BASE_URL}/feedback", json={"request_id": rid, "actual_value": 452600})
```

Nilai `ocean_proximity` yang valid: `<1H OCEAN`, `INLAND`, `ISLAND`, `NEAR BAY`, `NEAR OCEAN`.

## Monitoring

Jalankan `python monitoring.py` (dapat dijadwalkan) untuk memeriksa:

- **Layanan**: latensi p95 (batas 500 ms).
- **Data drift**: PSI dan uji Kolmogorov-Smirnov per fitur. PSI di bawah 0,10 berarti stabil, 0,10 sampai 0,25 waspada, dan di atas 0,25 drift.
- **Performa model**: MAE dan R² dari data aktual dibandingkan baseline saat pelatihan. Alert muncul jika MAE naik lebih dari 20%.

Hasil simulasi pada notebook:

| Skenario | MAE | R² | Hasil |
|---|---|---|---|
| Normal (300 data) | 40.350 | 0,736 | Tidak ada alert |
| Pergeseran pasar (pendapatan dan jumlah kamar naik, harga aktual naik 25%) | 66.187 | 0,633 | Alert: drift pada `median_income` (PSI 0,765) |

Baseline saat pelatihan: MAE 39.299 dan R² 0,761.

## Catatan Keamanan

File `joblib` dan `pickle` dapat menjalankan kode saat dimuat. Hanya muat file model yang dibuat sendiri atau berasal dari sumber tepercaya.

## Penulis

Rangga Dian Anggraita, Teknik Informatika, Universitas Pamulang.
