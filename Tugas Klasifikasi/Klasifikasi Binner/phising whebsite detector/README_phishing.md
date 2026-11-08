# README: Dataset Phishing Websites

## Ringkasan dataset

| Item | Keterangan |
|---|---|
| File | `phishing.csv` (dengan header), `phishing.txt` (data yang sama tanpa header) |
| Jumlah data | 11.054 baris pada CSV (TXT berisi 11.055 baris dengan urutan berbeda) |
| Kolom | 32 kolom: `Index` (nomor urut), 30 fitur, 1 target (`class`) |
| Tipe fitur | Semua bernilai `-1`, `0`, atau `1` (hasil pengkodean aturan) |
| Target | `class` |
| Missing value | Tidak ada |
| Data test terpisah | Tidak ada, sehingga data dibagi sendiri menjadi train dan test |

Setiap baris mewakili satu website. Fiturnya adalah ciri-ciri yang diekstrak dari URL dan halaman, misalnya `UsingIP`, `HTTPS`, `AnchorURL`, `SubDomains`, `PrefixSuffix-`, `WebsiteTraffic`, dan `AgeofDomain`.

## Mengapa ini termasuk klasifikasi biner

1. **Pembelajaran terawasi (supervised).** Setiap website sudah diberi label pada kolom `class`.
2. **Targetnya kategorikal, bukan angka kontinu.** `class` menyatakan jenis website (sah atau phishing), bukan besaran yang diukur. Karena itu masalahnya bukan regresi.
3. **Targetnya hanya punya tepat dua nilai.**

   | Nilai `class` | Arti | Jumlah | Persentase |
   |---|---|---|---|
   | 1 | Website sah (legitimate) | 6.157 | 55,7% |
   | -1 | Website phishing | 4.897 | 44,3% |

   Pada notebook, nilai `-1` dikodekan ulang menjadi `0` supaya mengikuti konvensi biner 0/1. Pengkodean ulang ini tidak mengubah sifat masalahnya.
4. **Kedua kelas saling lepas.** Satu website hanya masuk ke salah satu kelas, tanpa kelas ketiga.
5. **Keluaran model berupa keputusan dua pilihan:** website ini sah atau phishing.

## Karakteristik penting

- **Kelas relatif seimbang** (sekitar 56:44), sehingga akurasi cukup layak dipakai bersama precision, recall, dan F1.
- Nilai `0` pada fitur berarti "mencurigakan" atau "tidak pasti" pada aturan pengkodeannya. Itu adalah nilai fitur, bukan nilai target. Targetnya tetap hanya dua kelas.
- Ada 5.205 baris dengan nilai fitur identik, hal yang wajar karena tiap fitur hanya punya 3 nilai. Skor validasi bisa sedikit lebih optimis karena baris yang sama dapat muncul di data train dan validasi.
- Kolom `Index` dibuang dari fitur karena hanya nomor urut. Pada data ini, kelas `1` berarti website **sah**, jadi hati-hati saat membaca precision dan recall kelas 1 jika fokusnya mendeteksi phishing.

## Hasil baseline (5-fold cross-validation)

| Model | Accuracy | F1 | ROC-AUC |
|---|---|---|---|
| Logistic Regression | 0,927 | 0,935 | 0,979 |
| KNN | 0,939 | 0,945 | 0,983 |
| Decision Tree | 0,945 | 0,951 | 0,984 |
| Random Forest | 0,971 | 0,974 | 0,996 |
| Gradient Boosting | 0,967 | 0,971 | 0,996 |

Berbeda dari dua dataset lain, fitur di sini punya sinyal yang kuat. Fitur terpenting menurut Random Forest adalah `HTTPS`, `AnchorURL`, `WebsiteTraffic`, `SubDomains`, dan `LinksInScriptTags`.
