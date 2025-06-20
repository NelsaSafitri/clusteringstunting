@st.cache_data
def load_data():
    try:
        df = pd.read_csv("DataBalitaStunting.csv")
    except FileNotFoundError:
        st.warning("⚠️ File 'DataBalitaStunting.csv' tidak ditemukan. Menggunakan data dummy sebagai gantinya.")
        data_dummy = {
            'kecamatan': ['Kec A', 'Kec B', 'Kec C', 'Kec A', 'Kec B', 'Kec C'],
            'jumlah_balita_stunting': [34, 45, 23, 31, 40, 19],
            'tahun': [2021, 2021, 2021, 2022, 2022, 2022]
        }
        df = pd.DataFrame(data_dummy)
    else:
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
