### Laporan Proyek Machine Learning - Briyan Bayu Pangestu

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

Langkah-langkah yang diambil dalam persiapan data adalah sebagai berikut:

1. **Pembersihan judul film**: Menghilangkan tahun produksi dari judul film untuk mendapatkan nama film yang lebih bersih.
2. **Ekstraksi fitur konten**: Menggabungkan genre dan judul film untuk digunakan dalam model Content-Based Filtering.
3. **Normalisasi rating**: Skor rating dinormalisasi ke dalam skala \[0–1] menggunakan `MinMaxScaler` untuk kestabilan model, terutama untuk model deep learning Collaborative Filtering.
4. **Filtering data**: Hanya memilih pengguna yang memberikan minimal 20 rating dan film yang mendapatkan minimal 20 rating untuk memastikan data yang digunakan relevan dan memiliki informasi yang cukup.
5. **Split data**: Membagi dataset menjadi bagian training (70%), validation (15%), dan test (15%) untuk memastikan evaluasi yang adil.

Berikut adalah kode yang digunakan dalam data preparation:

```python
# Merge datasets
movie_ratings = pd.merge(ratings, movies, on='MovieID')
full_data = pd.merge(movie_ratings, users, on='UserID')

# Ekstrak tahun dari judul film
movies['Year'] = movies['Title'].str.extract(r'\((\d{4})\)')
movies['Year'] = movies['Year'].fillna(movies['Year'].mode()[0]).astype(int)

# Bersihkan judul film
movies['CleanTitle'] = movies['Title'].str.replace(r'\(\d{4}\)', '', regex=True).str.strip()

# Buat fitur gabungan untuk Content-Based Filtering
movies['Combined_Features'] = (
    movies['Genres'].apply(lambda x: ' '.join(x) if isinstance(x, list) else x) +
    ' ' + movies['CleanTitle']
)

# Normalisasi rating untuk deep learning
scaler = MinMaxScaler()
ratings['NormalizedRating'] = scaler.fit_transform(ratings[['Rating']])

# Hanya ambil user yang memberikan minimal 20 rating
user_counts = ratings['UserID'].value_counts()
active_users = user_counts[user_counts >= 20].index

# Hanya ambil film yang mendapat minimal 20 rating
movie_counts = ratings['MovieID'].value_counts()
popular_movies = movie_counts[movie_counts >= 20].index

# Filter dataset
filtered_ratings = ratings[
    (ratings['UserID'].isin(active_users)) &
    (ratings['MovieID'].isin(popular_movies))
].copy()
```

## Modeling

### 1. Content-Based Filtering

Untuk model **Content-Based Filtering**, kami menggunakan metode TF-IDF (Term Frequency-Inverse Document Frequency) untuk mengubah fitur teks (judul dan genre) menjadi representasi numerik. Cosine similarity digunakan untuk mengukur kedekatan antara film-film yang memiliki fitur serupa. Berikut adalah kode untuk implementasi model ini:

```python
class ImprovedContentBasedRecommender:
    def __init__(self):
        self.tfidf = TfidfVectorizer(
            stop_words='english',
            max_features=10000,
            ngram_range=(1, 2),  # Unigram dan bigram
            min_df=2,
            max_df=0.8
        )
        self.cosine_sim = None
        self.movies_data = None

    def fit(self, movies_df):
        """Melatih model Content-Based Filtering"""
        self.movies_data = movies_df.copy()

        # Buat TF-IDF matrix
        tfidf_matrix = self.tfidf.fit_transform(movies_df['Combined_Features'])

        # Hitung cosine similarity
        self.cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    
    def get_recommendations(self, title, n_recommendations=10):
        """Dapatkan rekomendasi berdasarkan judul film"""
        idx = self.movies_data[self.movies_data['Title'] == title].index[0]
        sim_scores = list(enumerate(self.cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        movie_indices = [i[0] for i in sim_scores[1:n_recommendations+1]]
        recommendations = self.movies_data.iloc[movie_indices][['Title', 'Genres', 'Year']]
        recommendations['Similarity_Score'] = [sim_scores[i+1][1] for i in range(n_recommendations)]
        return recommendations
```

### 2. Collaborative Filtering (Neural CF)

Untuk model **Collaborative Filtering**, kami menggunakan Neural Collaborative Filtering (NCF), yang merupakan model jaringan saraf dengan embedding untuk pengguna dan film. Model ini memungkinkan untuk menangkap interaksi kompleks antara pengguna dan film. Berikut adalah implementasi dasar dari model tersebut:

```python
class ImprovedNCFModel(keras.Model):
    def __init__(self, num_users, num_movies, embedding_size=64, dropout_rate=0.3):
        super().__init__()

        # Embedding layers
        self.user_embedding = layers.Embedding(input_dim=num_users, output_dim=embedding_size)
        self.movie_embedding = layers.Embedding(input_dim=num_movies, output_dim=embedding_size)
        self.user_bias = layers.Embedding(input_dim=num_users, output_dim=1)
        self.movie_bias = layers.Embedding(input_dim=num_movies, output_dim=1)

        # Hidden layers
        self.dropout1 = layers.Dropout(dropout_rate)
        self.dense1 = layers.Dense(128, activation='relu')
        self.dense2 = layers.Dense(64, activation='relu')
        self.output_layer = layers.Dense(1, activation='sigmoid')

    def call(self, inputs, training=None):
        user_id = inputs['user_id']
        movie_id = inputs['movie_id']
        user_vec = self.user_embedding(user_id)
        movie_vec = self.movie_embedding(movie_id)
        concat = layers.Concatenate()([user_vec, movie_vec])
        x = self.dropout1(concat, training=training)
        x = self.dense1(x)
        x = self.dense2(x)
        return self.output_layer(x)
```

## Evaluation

### Metrik Evaluasi:

**Content-Based Filtering**:

* Precision: 0.0354
* Recall: 0.0132
* F1-Score: 0.0192

**Collaborative Filtering**:

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
