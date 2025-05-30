# Laporan Proyek Machine Learning - Briyan Bayu Pangestu

## Project Overview

Sistem rekomendasi telah menjadi komponen krusial dalam berbagai platform digital, termasuk layanan streaming film. Dalam konteks industri hiburan, memberikan saran film yang relevan bagi pengguna tidak hanya meningkatkan kepuasan tetapi juga memperpanjang waktu keterlibatan pengguna dalam platform. Oleh karena itu, proyek ini bertujuan membandingkan dua pendekatan utama dalam sistem rekomendasi, yaitu **Content-Based Filtering** dan **Collaborative Filtering**.

Masalah ini penting untuk diselesaikan karena banyak platform menghadapi tantangan seperti cold start, keterbatasan personalisasi, serta tingginya sparsitas data pengguna. Pendekatan yang optimal dapat memberikan dampak nyata terhadap engagement dan retensi pengguna.

## Business Understanding

### Problem Statements

1. Bagaimana mengidentifikasi film yang disukai pengguna berdasarkan konten film seperti genre dan judul?
2. Bagaimana memanfaatkan pola rating dari banyak pengguna untuk memprediksi preferensi pengguna baru?
3. Bagaimana menangani masalah cold start terutama pada pengguna atau film baru dengan sedikit data historis?

### Goals

1. Mengembangkan model Content-Based Filtering berbasis TF-IDF dan cosine similarity.
2. Membangun model Collaborative Filtering menggunakan pendekatan Neural Collaborative Filtering (NCF).
3. Membandingkan performa kedua model berdasarkan metrik yang relevan.

### Solution Statements

* **Content-Based Filtering**: menggunakan representasi teks (judul dan genre) dengan TF-IDF dan menghitung cosine similarity antar film.
* **Collaborative Filtering (NCF)**: model neural network dengan embedding user dan item untuk menangkap interaksi kompleks dalam data rating.

## Data Understanding

Dataset yang digunakan adalah **MovieLens 1M** dari GroupLens:

* Jumlah rating: 1.000.209
* Jumlah pengguna: 6.040
* Jumlah film: 3.883
* Sumber: [GroupLens](https://grouplens.org/datasets/movielens/1m/)

### Fitur utama:

* `UserID`, `MovieID`, `Rating`, `Timestamp`: data rating
* `Title`, `Genres`: informasi konten film
* `Gender`, `Age`, `Occupation`, `Zip-code`: informasi pengguna

### Insight Awal:

* Skala rating: 1 - 5
* Matrix sangat sparse (95.53%)
* Rating didominasi nilai tinggi (bias positif)
* Genre populer: Drama, Comedy

## Data Preparation

Langkah-langkah:

1. **Pembersihan judul film**: menghilangkan tahun produksi dari judul
2. **Ekstraksi fitur konten**: menggabungkan genre dan judul
3. **Normalisasi rating** untuk training model CF
4. **Filtering**: hanya user dan item dengan minimal 20 rating digunakan
5. **Split data**: training (70%), validation (15%), test (15%)

Alasan filtering adalah untuk mengurangi noise dari pengguna pasif dan film yang tidak cukup populer.

## Modeling

### 1. Content-Based Filtering

* Metode: TF-IDF + Cosine Similarity
* Fitur: Kombinasi genre dan judul film
* Output: Top-N rekomendasi berdasarkan film input pengguna
* Evaluasi berdasarkan precision, recall, dan F1-score

### 2. Collaborative Filtering (Neural CF)

* Arsitektur: Embedding user dan item (64 dimensi), 3 hidden layer, dropout, L2 regularisasi
* Optimizer: Adam, Loss: MSE
* Teknik: Early Stopping, ReduceLROnPlateau
* Output: Prediksi rating dan rekomendasi personal

## Evaluation

### Metrik Evaluasi:

* **Content-Based Filtering**:

  * Precision: 0.0343
  * Recall: 0.0125
  * F1-Score: 0.0184

* **Collaborative Filtering**:

  * Test RMSE: 0.2204 → Denormalized: 0.8818
  * Test MAE: 0.1735 → Denormalized: 0.6940

### Perbandingan:

| Aspek             | Content-Based | Collaborative Filtering |
| ----------------- | ------------- | ----------------------- |
| Akurasi           | Rendah        | Baik                    |
| Personalisasi     | Terbatas      | Tinggi                  |
| Cold Start - User | Tidak bisa    | Tidak bisa              |
| Cold Start - Item | Bisa          | Tidak bisa              |
| Diversity         | Rendah        | Sedang                  |
| Skalabilitas      | Tinggi        | Rendah (resource-heavy) |

# Referensi:

* \[1] [Movie Recommendation System (Content-Based Filtering) - IJIRSET.](https://www.ijirset.com/upload/2024/june/13_Movie.pdf)
* \[2] [What is collaborative filtering? - IBM.](https://www.ibm.com/think/topics/collaborative-filtering)
* \[3] [A Survey on Deep Neural Networks in Collaborative Filtering Recommendation Systems - arXiv](https://arxiv.org/abs/2412.01378)
* \[4] [Content-Based and Collaborative Filtering Models - GitHub.](https://github.com/khanhnamle1994/movielens/blob/master/Content_Based_and_Collaborative_Filtering_Models.ipynb)
* \[5] [What is content-based filtering? - IBM.](https://www.ibm.com/think/topics/content-based-filtering)
* \[6] [Content-based Filtering for Improving Movie Recommender System - Atlantis Press.](https://www.atlantis-press.com/article/125998090.pdf)
* \[7] [Enhancing Recommender Systems - MDPI.](https://www.mdpi.com/2076-3417/13/18/10041)
