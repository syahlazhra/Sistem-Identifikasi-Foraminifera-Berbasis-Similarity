# app.py

```python
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(
    page_title="Identifikasi Foraminifera",
    page_icon="🔬",
    layout="wide"
)

# =========================
# CSS STYLE
# =========================
st.markdown(
    """
    <style>
    .main {
        background-color: #f5f7fa;
    }

    .title {
        text-align: center;
        font-size: 42px;
        font-weight: bold;
        color: #1f4e79;
        margin-bottom: 10px;
    }

    .subtitle {
        text-align: center;
        font-size: 18px;
        color: #555;
        margin-bottom: 30px;
    }

    .result-box {
        background-color: white;
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        margin-bottom: 20px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# =========================
# TITLE
# =========================
st.markdown('<div class="title">🔬 Sistem Identifikasi Foraminifera</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Metode Similarity Menggunakan Cosine Similarity</div>',
    unsafe_allow_html=True
)

# =========================
# IMAGE DATABASE
# =========================
image_urls = {
    "Globigerina_bulloides": "https://www.mikrotax.org/images/pf_cenozoic/Globigerinidae/Globigerina/Globigerina%20bulloides/K_S%201983%2006-4.JPG",
    "Globorotalia_menardii": "https://www.mikrotax.org/images/pf_cenozoic/Globorotaliidae/Globorotalia/menardii%20lineage/Globorotalia%20menardii/K_S%201983%2029-1.JPG",
    "Orbulina_universa": "https://www.mikrotax.org/images/pf_cenozoic/Globigerinidae/Orbulina/Orbulina%20universa/K_S%201983%2020-4.JPG",
    "Nummulites_sp": "https://www.mindat.org/imagecache/85/ed/00960130017362090929908.jpg",
    "Operculina_sp": "https://images.marinespecies.org/thumbs/119138_operculina-complanata-defrance-in-de-blainville-1822.jpg",
    "Fusulina_sp": "https://media.sketchfab.com/models/960ac3f9ee754d3a889ad9579c922ff7/thumbnails/7221b5ef600a4853be40eb458f2ebb1f/f793b4f964644c2ebd1a7a6f2ce8f4e1.jpeg",
    "Textularia_sp": "https://www.mikrotax.org/images/pf_cat/T/Textularia/Textularia%20globulosa/USNM%20%20264610-156.jpg",
    "Bolivina_sp": "https://www.mikrotax.org/images/pf_cat/B/Bolivina/Bolivina%20merecuanai/Sellier%20de%20Civrieux%201976%20pl09%20f05-8.JPG",
    "Ammonia_beccarii": "https://foraminifera.eu/singimg/ammonia-beccarii-stirone.jpg",
    "Elphidium_sp": "https://www.mikrotax.org/images/bf_main/Rotaliana/Rotalioidea/Elphidium/Elphidium%20crispum/Holbourn%20et%20al%202013%20f319.jpg",
    "Quinqueloculina_sp": "https://www.mikrotax.org/images/bf_main/Miliolida/Quinqueloculina/Quinqueloculina%20sp./Cushman%201946%20pl.%2014%20fig.%2012.jpg",
    "Spiroloculina_sp": "https://foraminifera.eu/singimg/sonx003.jpg",
    "Lagena_sp": "https://www.mikrotax.org/images/bf_main/Nodosariana/Nodosariida/Lagenidae/Lagena/Lagena%20sp./Hermelin%201989%20pl.%204%20fig.%2018.jpg",
    "Guttulina_sp": "https://www.mikrotax.org/images/bf_main/Nodosariana/Polymorphinida/Polymorphinidae/Guttulina/Guttulina%20trigonula/Bolli%20et%20al%201994%20pl33%20fig10-12.jpg",
    "Planorbulina_sp": "https://images.marinespecies.org/thumbs/173349_planorbulinella-larvata.jpg"
}

# =========================
# FILE UPLOAD
# =========================
st.sidebar.header("📂 Upload Dataset")

uploaded_file = st.sidebar.file_uploader(
    "Upload file CSV",
    type=["csv"]
)

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file, sep=';')

    # CLEANING
    df.columns = df.columns.str.strip()
    df.columns = df.columns.str.lower()
    df.columns = df.columns.str.replace(" ", "_")

    df['spesies'] = df['spesies'].str.strip()
    df['spesies'] = df['spesies'].str.replace(" ", "_")

    # ADD IMAGE
    df['image'] = df['spesies'].map(image_urls)

    # FEATURES
    features = [
        'jumlah_kamar',
        'ukuran',
        'bentuk_cangkang',
        'spiral',
        'tekstur',
        'ostia',
        'oskula',
        'spongocoel',
        'body_wall'
    ]

    df_features = df[features]

    # ENCODING
    df_encoded = pd.get_dummies(df_features)

    # SCALING
    scaler = MinMaxScaler()
    df_scaled = scaler.fit_transform(df_encoded)

    # =========================
    # INPUT FORM
    # =========================
    st.sidebar.header("🧪 Input Karakteristik")

    jumlah_kamar = st.sidebar.number_input("Jumlah Kamar", min_value=1, value=7)
    ukuran = st.sidebar.number_input("Ukuran (µm)", min_value=1, value=150)

    bentuk_cangkang = st.sidebar.selectbox(
        "Bentuk Cangkang",
        ['bulat', 'oval', 'memanjang']
    )

    spiral = st.sidebar.selectbox(
        "Spiral",
        ['trochoid', 'planispiral', 'serial', 'spherical']
    )

    tekstur = st.sidebar.selectbox(
        "Tekstur",
        ['halus', 'kasar']
    )

    ostia = st.sidebar.selectbox(
        "Ostia",
        ['kecil', 'besar']
    )

    oskula = st.sidebar.selectbox(
        "Oskula",
        ['ada', 'tidak_ada']
    )

    spongocoel = st.sidebar.selectbox(
        "Spongocoel",
        ['ada', 'tidak_ada']
    )

    body_wall = st.sidebar.selectbox(
        "Body Wall",
        ['tipis', 'sedang', 'tebal', 'agregat']
    )

    # =========================
    # BUTTON
    # =========================
    if st.sidebar.button("🔍 Identifikasi"):

        input_user = {
            'jumlah_kamar': jumlah_kamar,
            'ukuran': ukuran,
            'bentuk_cangkang': bentuk_cangkang,
            'spiral': spiral,
            'tekstur': tekstur,
            'ostia': ostia,
            'oskula': oskula,
            'spongocoel': spongocoel,
            'body_wall': body_wall
        }

        df_input = pd.DataFrame([input_user])

        # ENCODE INPUT
        df_input_encoded = pd.get_dummies(df_input)
        df_input_encoded = df_input_encoded.reindex(
            columns=df_encoded.columns,
            fill_value=0
        )

        # SCALE INPUT
        df_input_scaled = scaler.transform(df_input_encoded)

        # COSINE SIMILARITY
        similarity = cosine_similarity(df_input_scaled, df_scaled)[0]

        df['similarity'] = similarity * 100

        # TOP RESULT
        top5 = df.sort_values('similarity', ascending=False).head(5)

        st.success("✅ Identifikasi berhasil dilakukan")

        st.subheader("🏆 Top 5 Similarity")

        for i, row in top5.iterrows():

            col1, col2 = st.columns([1, 2])

            with col1:
                if pd.notnull(row['image']):
                    st.image(row['image'], width=220)
                else:
                    st.warning("Gambar tidak tersedia")

            with col2:
                st.markdown(f"### {row['spesies']}")
                st.progress(int(row['similarity']))
                st.write(f"Similarity: {row['similarity']:.2f}%")

            st.markdown("---")

    # =========================
    # DATA PREVIEW
    # =========================
    with st.expander("📊 Preview Dat
```
