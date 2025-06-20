import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Clustering Stunting Balita", layout="wide")
st.title("🧠 Clustering Stunting Balita dengan K-Means")

# ====================
# Upload CSV
# ====================
uploaded_file = st.file_uploader("📂 Unggah file CSV Data Balita Stunting", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    # ====================
    # Data Preprocessing
    # ====================
    with st.expander("🔍 Pratinjau dan Pembersihan Data"):
        # Gabung nama kecamatan
        df['kecamatan'] = df['kemendagri_nama_kecamatan'].fillna(df['bps_nama_kecamatan'])
        df.drop(['bps_nama_kecamatan', 'kemendagri_nama_kecamatan'], axis=1, inplace=True)

        # Hapus kolom tidak relevan
        drop_cols = ['nama_provinsi', 'bps_nama_kabupaten_kota', 'bps_nama_desa_kelurahan',
                     'kemendagri_nama_desa_kelurahan', 'satuan', 'Filter']
        df.drop(columns=[col for col in drop_cols if col in df.columns], inplace=True)

        # Hapus NaN dan ubah tipe data
        df = df.dropna(subset=['kecamatan', 'jumlah_balita_stunting', 'tahun'])
        df['jumlah_balita_stunting'] = pd.to_numeric(df['jumlah_balita_stunting'], errors='coerce')
        df['tahun'] = pd.to_numeric(df['tahun'], errors='coerce')
        df = df.dropna(subset=['jumlah_balita_stunting', 'tahun']).reset_index(drop=True)

        st.dataframe(df.head(10))

    # ====================
    # Encoding & Scaling
    # ====================
    df['kecamatan_enc'] = LabelEncoder().fit_transform(df['kecamatan'])
    features = ['jumlah_balita_stunting', 'tahun', 'kecamatan_enc']
    X = StandardScaler().fit_transform(df[features])

    # ====================
    # Sidebar: Clustering Config
    # ====================
    st.sidebar.header("⚙️ Pengaturan Clustering")
    n_clusters = st.sidebar.slider("Jumlah Klaster", min_value=2, max_value=10, value=3)

    # ====================
    # KMeans Clustering
    # ====================
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    df['cluster'] = kmeans.fit_predict(X)

    # ====================
    # Visualisasi Clustering
    # ====================
    st.subheader("📊 Visualisasi Clustering (Jumlah Balita vs Tahun)")
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.scatterplot(x=df['jumlah_balita_stunting'], y=df['tahun'], hue=df['cluster'], palette='tab10', ax=ax)
    ax.set_title("Hasil Clustering K-Means")
    ax.set_xlabel("Jumlah Balita Stunting")
    ax.set_ylabel("Tahun")
    st.pyplot(fig)

    # ====================
    # Evaluasi: Elbow & Silhouette
    # ====================
    with st.expander("📈 Evaluasi Klaster: Elbow & Silhouette Score"):
        wcss, silhouette = [], []
        range_k = range(2, 11)
        for k in range_k:
            km = KMeans(n_clusters=k, random_state=42)
            lbls = km.fit_predict(X)
            wcss.append(km.inertia_)
            silhouette.append(silhouette_score(X, lbls))

        fig1, ax1 = plt.subplots()
        ax1.plot(range_k, wcss, marker='o')
        ax1.set_title("Elbow Method")
        ax1.set_xlabel("Jumlah Klaster")
        ax1.set_ylabel("WCSS")
        st.pyplot(fig1)

        fig2, ax2 = plt.subplots()
        ax2.plot(range_k, silhouette, marker='o', color='green')
        ax2.set_title("Silhouette Score")
        ax2.set_xlabel("Jumlah Klaster")
        ax2.set_ylabel("Score")
        st.pyplot(fig2)

    # ====================
    # Tampilkan Data
    # ====================
    st.subheader("📄 Data Hasil Clustering")
    st.dataframe(df[['kecamatan', 'tahun', 'jumlah_balita_stunting', 'cluster']])

    # ====================
    # Download Data
    # ====================
    csv_out = df.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Unduh Data Hasil Clustering", data=csv_out, file_name="hasil_clustering.csv", mime="text/csv")

else:
    st.info("Silakan unggah file CSV untuk memulai.")
