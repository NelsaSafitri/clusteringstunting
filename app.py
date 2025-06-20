import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns

st.title("Clustering Stunting Balita dengan K-Means")

# Load data
@st.cache_data
def load_data():
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
    return df

df = load_data()

# Encoding & Scaling
le = LabelEncoder()
df['kecamatan_enc'] = le.fit_transform(df['kecamatan'])
features = ['jumlah_balita_stunting', 'tahun', 'kecamatan_enc']
scaler = StandardScaler()
X = scaler.fit_transform(df[features])

# Pilih jumlah klaster
st.sidebar.header("Pengaturan Clustering")
n_clusters = st.sidebar.slider("Jumlah Klaster", 2, 10, 3)

# KMeans
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
labels = kmeans.fit_predict(X)
df['cluster'] = labels

# Plot hasil clustering
fig, ax = plt.subplots()
sns.scatterplot(x=df['jumlah_balita_stunting'], y=df['tahun'], hue=df['cluster'], palette='tab10', ax=ax)
ax.set_title("Hasil Clustering")
st.pyplot(fig)

# Elbow & Silhouette
if st.sidebar.checkbox("Tampilkan Elbow & Silhouette Score"):
    wcss = []
    silhouette = []
    K_range = range(2, 11)
    for k in K_range:
        km = KMeans(n_clusters=k, random_state=42)
        lbls = km.fit_predict(X)
        wcss.append(km.inertia_)
        silhouette.append(silhouette_score(X, lbls))

    fig2, ax2 = plt.subplots()
    ax2.plot(K_range, wcss, marker='o')
    ax2.set_title("Elbow Method (WCSS)")
    ax2.set_xlabel("Jumlah Klaster")
    ax2.set_ylabel("WCSS")
    st.pyplot(fig2)

    fig3, ax3 = plt.subplots()
    ax3.plot(K_range, silhouette, marker='o', color='green')
    ax3.set_title("Silhouette Score")
    ax3.set_xlabel("Jumlah Klaster")
    ax3.set_ylabel("Silhouette Score")
    st.pyplot(fig3)

# Tampilkan data hasil cluster
st.subheader("Data Hasil Clustering")
st.dataframe(df[['kecamatan', 'tahun', 'jumlah_balita_stunting', 'cluster']])