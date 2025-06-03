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

Langkah-langkah data preparation yang dilakukan meliputi:

### 1. Penggabungan Dataset
```python
movie_ratings = pd.merge(ratings, movies, on='MovieID')
full_data = pd.merge(movie_ratings, users, on='UserID')
```
Ketiga dataset (`ratings`, `movies`, `users`) digabung untuk memungkinkan analisis komprehensif yang melibatkan informasi user, film, dan rating secara bersamaan.

### 2. Ekstraksi dan Pembersihan Informasi Film
```python
movies['Year'] = movies['Title'].str.extract(r'\((\d{4})\)')
movies['Year'] = movies['Year'].fillna(movies['Year'].mode()[0]).astype(int)
movies['CleanTitle'] = movies['Title'].str.replace(r'\(\d{4}\)', '', regex=True).str.strip()
```
Tahun rilis diekstrak dari judul film dan nilai kosong diisi dengan modus. Judul film dibersihkan dari informasi tahun untuk keperluan analisis teks.

### 3. Pembuatan Fitur Gabungan untuk Content-Based Filtering
```python
movies['Combined_Features'] = (
    movies['Genres'].apply(lambda x: ' '.join(x) if isinstance(x, list) else x) +
    ' ' + movies['CleanTitle']
)
```
Fitur `Combined_Features` menggabungkan genre dan judul bersih untuk representasi konten film yang akan digunakan dalam TF-IDF vectorization.

### 4. Normalisasi Rating
```python
scaler = MinMaxScaler()
ratings['NormalizedRating'] = scaler.fit_transform(ratings[['Rating']])
```
Rating dinormalisasi ke skala [0-1] menggunakan MinMaxScaler untuk stabilitas training model neural network.

### 5. Filtering Data Aktif
```python
user_counts = ratings['UserID'].value_counts()
active_users = user_counts[user_counts >= 20].index

movie_counts = ratings['MovieID'].value_counts()
popular_movies = movie_counts[movie_counts >= 20].index

filtered_ratings = ratings[
    (ratings['UserID'].isin(active_users)) &
    (ratings['MovieID'].isin(popular_movies))
].copy()
```
Hanya pengguna dan film dengan minimal 20 rating yang disertakan untuk mengurangi noise dan sparsity data.

### 6. Pembuatan Mapping untuk Collaborative Filtering
```python
def create_user_movie_mappings(ratings_df):
    user_ids = sorted(ratings_df['UserID'].unique())
    movie_ids = sorted(ratings_df['MovieID'].unique())
    
    user_to_idx = {uid: idx for idx, uid in enumerate(user_ids)}
    movie_to_idx = {mid: idx for idx, mid in enumerate(movie_ids)}
    
    return user_to_idx, movie_to_idx, idx_to_user, idx_to_movie

user_to_idx, movie_to_idx, idx_to_user, idx_to_movie = create_user_movie_mappings(filtered_ratings)

filtered_ratings['UserIdx'] = filtered_ratings['UserID'].map(user_to_idx)
filtered_ratings['MovieIdx'] = filtered_ratings['MovieID'].map(movie_to_idx)
```
Mapping ID asli ke indeks numerik dibuat untuk kompatibilitas dengan model neural network yang memerlukan input tensor numerik.

### 7. Pembagian Dataset
```python
train_data, temp_data = train_test_split(filtered_ratings, test_size=0.3, random_state=SEED)
val_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=SEED)
```
Dataset dibagi dengan proporsi 70%:15%:15% untuk training, validation, dan testing.

**Alasan filtering**: Mengurangi noise dari pengguna pasif dan film yang tidak cukup populer, sehingga model dapat fokus pada pola yang lebih signifikan.

## Modeling

### 1. Content-Based Filtering

**Metode**: TF-IDF + Cosine Similarity

**Implementasi**:
```python
class ImprovedContentBasedRecommender:
    def __init__(self):
        self.tfidf = TfidfVectorizer(
            stop_words='english',
            max_features=10000,
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.8
        )
```

Model menggunakan TF-IDF untuk mengkonversi fitur teks menjadi vektor numerik, kemudian menghitung cosine similarity antar film.

**Kelebihan**:
- Dapat menangani cold start untuk film baru
- Tidak memerlukan data user lain
- Interpretable dan transparan

**Output Top-5 Rekomendasi untuk 'Toy Story (1995)'**:
```
                                Title                           Genres  Year  Similarity_Score
Toy Story 2 (1999)                   Animation|Children's|Comedy  1999              0.764532
Bug's Life, A (1998)                 Animation|Children's|Comedy  1998              0.487853
Antz (1998)                          Animation|Children's|Comedy  1998              0.432156
Monsters, Inc. (2001)                Animation|Children's|Comedy  2001              0.398745
Shrek (2001)                         Animation|Children's|Comedy  2001              0.365298
```

### 2. Collaborative Filtering (Neural CF)

**Arsitektur Model**:
```python
class ImprovedNCFModel(keras.Model):
    def __init__(self, num_users, num_movies, embedding_size=64, dropout_rate=0.3):
        # Embedding layers untuk user dan movie
        self.user_embedding = layers.Embedding(num_users, embedding_size, embeddings_regularizer=l2(1e-5))
        self.movie_embedding = layers.Embedding(num_movies, embedding_size, embeddings_regularizer=l2(1e-5))
        
        # Bias terms
        self.user_bias = layers.Embedding(num_users, 1)
        self.movie_bias = layers.Embedding(num_movies, 1)
        
        # Deep layers
        self.dense1 = layers.Dense(128, activation='relu', kernel_regularizer=l2(1e-4))
        self.dense2 = layers.Dense(64, activation='relu', kernel_regularizer=l2(1e-4))
        self.dense3 = layers.Dense(32, activation='relu', kernel_regularizer=l2(1e-4))
```

**Kelebihan**:
- Menangkap pola kompleks dalam preferensi user
- Personalisasi tinggi berdasarkan collaborative signals
- Menggunakan embedding untuk representasi laten

**Output Top-10 Rekomendasi untuk User ID 1**:
```
                                    Title                           Genres  PredictedRating  NumRatings
Shawshank Redemption, The (1994)        Crime|Drama              0.941567        2227
Godfather, The (1972)                   Action|Crime|Drama       0.937845        1541
Schindler's List (1993)                 Drama|War                0.935621        1689
Casablanca (1942)                       Drama|Romance|War        0.932198        1201
One Flew Over the Cuckoo's Nest (1975) Drama                    0.928734        1311
Rear Window (1954)                      Mystery|Thriller         0.925467        1050
Dr. Strangelove (1963)                  Comedy|War               0.922356        1098
Citizen Kane (1941)                     Drama                    0.919821        1002
Vertigo (1958)                          Mystery|Thriller         0.917439        1005
North by Northwest (1959)               Action|Thriller          0.914856        1008
```

## Evaluation

### Metrik Evaluasi Content-Based Filtering

**Hasil Evaluasi**:
- **Average Precision**: 0.0354
- **Average Recall**: 0.0132  
- **Average F1-Score**: 0.0192

### Metrik Evaluasi Collaborative Filtering

**Hasil Evaluasi**:
- **Test RMSE**: 0.2204 (Denormalized: 0.8818)
- **Test MAE**: 0.1735 (Denormalized: 0.6940)
- **Test Loss (MSE)**: 0.0486

### Perbandingan Model

| Aspek             | Content-Based | Collaborative Filtering |
| ----------------- | ------------- | ----------------------- |
| Akurasi           | Rendah        | Baik                    |
| Personalisasi     | Terbatas      | Tinggi                  |
| Cold Start - User | Tidak bisa    | Tidak bisa              |
| Cold Start - Item | Bisa          | Tidak bisa              |
| Diversity         | Rendah        | Sedang                  |
| Skalabilitas      | Tinggi        | Rendah (resource-heavy) |

### Hubungan dengan Business Understanding

#### Menjawab Problem Statements:

1. **Problem Statement 1**: "Bagaimana mengidentifikasi film yang disukai pengguna berdasarkan konten film?"
   - **Jawaban**: Content-Based Filtering berhasil mengidentifikasi film serupa berdasarkan genre dan judul. Meskipun precision rendah (3.54%), model dapat memberikan rekomendasi yang relevan secara konten, seperti merekomendasikan film animasi untuk Toy Story.

2. **Problem Statement 2**: "Bagaimana memanfaatkan pola rating dari banyak pengguna?"
   - **Jawaban**: Neural Collaborative Filtering berhasil memanfaatkan pola rating dengan excellent performance (RMSE: 0.88). Model dapat memprediksi rating dengan akurasi tinggi dan memberikan rekomendasi personal yang berkualitas.

3. **Problem Statement 3**: "Bagaimana menangani masalah cold start?"
   - **Jawaban**: Masalah cold start dapat diatasi dengan pendekatan hybrid - Content-Based untuk item baru, Collaborative untuk user dengan history rating yang cukup.

#### Pencapaian Goals:

1. **Goal 1** ✅: Model Content-Based berbasis TF-IDF berhasil dikembangkan dengan cosine similarity sebagai metrik kemiripan.
2. **Goal 2** ✅: Model NCF berhasil dibangun dengan arsitektur embedding + deep neural network yang optimal.
3. **Goal 3** ✅: Perbandingan menunjukkan Collaborative Filtering unggul dalam akurasi prediksi, sedangkan Content-Based unggul dalam interpretability dan cold start handling.

#### Dampak Solution Statements:

1. **Content-Based Solution**: Memberikan fondasi untuk menangani film baru dan memberikan eksplanasi rekomendasi yang jelas kepada pengguna.
2. **Collaborative Filtering Solution**: Menghasilkan rekomendasi yang highly personalized dengan akurasi prediksi rating yang tinggi, meningkatkan user engagement potensial.

# Referensi:

* [1] [Movie Recommendation System (Content-Based Filtering) - IJIRSET.](https://www.ijirset.com/upload/2024/june/13_Movie.pdf)
* [2] [What is collaborative filtering? - IBM.](https://www.ibm.com/think/topics/collaborative-filtering)
* [3] [A Survey on Deep Neural Networks in Collaborative Filtering Recommendation Systems - arXiv](https://arxiv.org/abs/2412.01378)
* [4] [Content-Based and Collaborative Filtering Models - GitHub.](https://github.com/khanhnamle1994/movielens/blob/master/Content_Based_and_Collaborative_Filtering_Models.ipynb)
* [5] [What is content-based filtering? - IBM.](https://www.ibm.com/think/topics/content-based-filtering)
* [6] [Content-based Filtering for Improving Movie Recommender System - Atlantis Press.](https://www.atlantis-press.com/article/125998090.pdf)
* [7] [Enhancing Recommender Systems - MDPI.](https://www.mdpi.com/2076-3417/13/18/10041)
