import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns

# Setup halaman
st.set_page_config(page_title="Clustering Balita Stunting", layout="wide")

st.title("📊 Clustering Balita Stunting dengan K-Means")

# Fungsi load data
@st.cache_data
def load_data():
    # Coba baca file CSV
    try:
        df = pd.read_csv("DataBalitaStunting.csv")
        df['kecamatan'] = df['kemendagri_nama_kecamatan'].fillna(df['bps_nama_kecamatan'])
        df.drop(['bps_nama_kecamatan', 'kemendagri_nama_kecamatan'], axis=1, inplace=True)
        cols_drop = ['nama_provinsi', 'bps_nama_kabupaten_kota', 'bps_nama_desa_kelurahan',
                     'kemendagri_nama_desa_kelurahan', 'satuan', 'Filter']
        df.drop(columns=[col for col in cols_drop if col in df.columns], inplace=True)
        df = df.dropna(subset=['kecamatan', 'jumlah_balita_stunting', 'tahun'])
        df['jumlah_balita_stunting'] = pd.to_numeric(df['jumlah_balita_stunting'], errors='coerce')
        df['tahun'] = pd.to_numeric(df['tahun'], errors='coerce')
        df = df.dropna(subset=['jumlah_balita_stunting', 'tahun']).reset_index(drop=True)
    except FileNotFoundError:
        st.warning("📁 File CSV tidak ditemukan. Menggunakan data dummy.")
        data_dummy = {
            'kecamatan': ['Kec A', 'Kec B', 'Kec C', 'Kec A', 'Kec B', 'Kec C'],
            'jumlah_balita_stunting': [34, 45, 23, 31, 40, 19],
            'tahun': [2021, 2021, 2021, 2022, 2022, 2022]
        }
        df = pd.DataFrame(data_dummy)
    return df

# Load data
df = load_data()

# Sidebar konfigurasi
st.sidebar.header("🔧 Pengaturan Clustering")
n_clusters = st.sidebar.slider("Jumlah Klaster", min_value=2, max_value=10, value=3)

# Preprocessing
le = LabelEncoder()
df['kecamatan_enc'] = le.fit_transform(df['kecamatan'])
features = ['jumlah_balita_stunting', 'tahun', 'kecamatan_enc']
scaler = StandardScaler()
X = scaler.fit_transform(df[features])

# Clustering
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
df['cluster'] = kmeans.fit_predict(X)

# Visualisasi hasil clustering
st.subheader("🧬 Visualisasi Klaster")
fig, ax = plt.subplots()
sns.scatterplot(data=df, x='jumlah_balita_stunting', y='tahun', hue='cluster', palette='tab10', s=100)
ax.set_title("Hasil Clustering")
st.pyplot(fig)

# Elbow & Silhouette Score
if st.sidebar.checkbox("📈 Tampilkan Elbow & Silhouette"):
    wcss = []
    silhouette_scores = []
    K_range = range(2, 11)
    for k in K_range:
        model = KMeans(n_clusters=k, random_state=42)
        labels = model.fit_predict(X)
        wcss.append(model.inertia_)
        silhouette_scores.append(silhouette_score(X, labels))

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Elbow Method")
        fig1, ax1 = plt.subplots()
        ax1.plot(K_range, wcss, marker='o')
        ax1.set_xlabel("Jumlah Klaster")
        ax1.set_ylabel("WCSS")
        ax1.set_title("Elbow Curve")
        st.pyplot(fig1)

    with col2:
        st.markdown("#### Silhouette Score")
        fig2, ax2 = plt.subplots()
        ax2.plot(K_range, silhouette_scores, marker='o', color='green')
        ax2.set_xlabel("Jumlah Klaster")
        ax2.set_ylabel("Silhouette Score")
        ax2.set_title("Silhouette Curve")
        st.pyplot(fig2)

# Tabel hasil
st.subheader("📄 Data Hasil Clustering")
st.dataframe(df[['kecamatan', 'tahun', 'jumlah_balita_stunting', 'cluster']])
