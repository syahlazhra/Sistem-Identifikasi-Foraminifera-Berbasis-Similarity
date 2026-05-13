from google.colab import files
uploaded = files.upload()

import pandas as pd

file_name = list(uploaded.keys())[0]
df = pd.read_csv(file_name, sep=';')

print("=== DATA AWAL ===")
print(df.head())

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

df['spesies'] = df['spesies'].str.strip()
df['spesies'] = df['spesies'].str.replace(" ", "_")

df['image'] = df['spesies'].map(image_urls)

import pandas as pd

file_name = list(uploaded.keys())[0]
df = pd.read_csv(file_name, sep=';')

# autofix kolom
df.columns = df.columns.str.strip()
df.columns = df.columns.str.lower()
df.columns = df.columns.str.replace(" ", "_")

print("=== NAMA KOLOM ===")
print(df.columns)

# Clean the 'spesies' column to match image_urls keys
df['spesies'] = df['spesies'].str.strip()
df['spesies'] = df['spesies'].str.replace(" ", "_")

# Map image URLs to the DataFrame
df['image'] = df['spesies'].map(image_urls)

print("First 5 rows with image URLs:")
display(df[['spesies', 'image']].head())

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

print(df_features.head())

df_encoded = pd.get_dummies(df_features)

print(df_encoded.head())

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
df_scaled = scaler.fit_transform(df_encoded)

input_data = pd.DataFrame([{
    'jumlah_kamar': 7,
    'ukuran': 150,
    'bentuk_cangkang': 'bulat',
    'spiral': 'trochoid',
    'tekstur': 'halus',
    'ostia': 'kecil',
    'oskula': 'tidak_ada',
    'spongocoel': 'tidak_ada',
    'body_wall': 'sedang'
}])

input_encoded = pd.get_dummies(input_data)

input_encoded = input_encoded.reindex(
    columns=df_encoded.columns,
    fill_value=0
)

input_scaled = scaler.transform(input_encoded)
input_scaled = input_scaled[0]

import numpy as np

def euclidean(a, b):
    return np.sqrt(np.sum((a - b)**2))

distances = []

for row in df_scaled:
    distances.append(euclidean(input_scaled, row))

df['distance'] = distances

import ipywidgets as widgets
from IPython.display import display

jumlah_kamar = widgets.IntText(description='Jumlah kamar')
ukuran = widgets.IntText(description='Ukuran (µm)')

bentuk_cangkang = widgets.Dropdown(
    options=['bulat','oval','memanjang'],
    description='Bentuk'
)

spiral = widgets.Dropdown(
    options=['trochoid','planispiral','serial','spherical'],
    description='Spiral'
)

tekstur = widgets.Dropdown(
    options=['halus','kasar'],
    description='Tekstur'
)

ostia = widgets.Dropdown(options=['kecil','besar'], description='Ostia')
oskula = widgets.Dropdown(options=['ada','tidak_ada'], description='Oskula')
spongocoel = widgets.Dropdown(options=['ada','tidak_ada'], description='Spongocoel')
body_wall = widgets.Dropdown(options=['tipis','sedang','tebal','agregat'], description='Body wall')

btn = widgets.Button(description="Proses")

display(
    jumlah_kamar, ukuran, bentuk_cangkang, spiral,
    tekstur, ostia, oskula, spongocoel, body_wall, btn
)

from IPython.display import Image, display
from sklearn.metrics.pairwise import cosine_similarity
from IPython.display import clear_output

def on_button_click(b):
    clear_output(wait=True);


    display(
        jumlah_kamar, ukuran, bentuk_cangkang, spiral,
        tekstur, ostia, oskula, spongocoel, body_wall, btn
    )

    input_user = {
        'jumlah_kamar': jumlah_kamar.value,
        'ukuran': ukuran.value,
        'bentuk_cangkang': bentuk_cangkang.value,
        'spiral': spiral.value,
        'tekstur': tekstur.value,
        'ostia': ostia.value,
        'oskula': oskula.value,
        'spongocoel': spongocoel.value,
        'body_wall': body_wall.value
    }

    df_input = pd.DataFrame([input_user])
    df_input_encoded = pd.get_dummies(df_input)
    df_input_encoded = df_input_encoded.reindex(columns=df_encoded.columns, fill_value=0)

    df_input_scaled = scaler.transform(df_input_encoded)

    similarity = cosine_similarity(df_input_scaled, df_scaled)[0]
    df['similarity'] = similarity * 100

    top5 = df.sort_values('similarity', ascending=False).head(5)

    print("\n=== HASIL ===\n")

    for i, row in top5.iterrows():
        print(f"Spesies: {row['spesies']}")
        print(f"Similarity: {row['similarity']:.2f}%")

        if 'image' in row.index and isinstance(row['image'], str):
            display(Image(url=row['image'], width=200))
        else:
            print("❌ gambar tidak ada")

        print("-"*30)

btn.on_click(on_button_click)