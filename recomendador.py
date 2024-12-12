import streamlit as st
import pandas as pd
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Funciones reutilizadas
def recommend_by_content(tfidf_matrix, selected_movie_index, dataset, top_n=5):
    similarities = cosine_similarity(tfidf_matrix[selected_movie_index], tfidf_matrix).flatten()
    similar_indices = similarities.argsort()[-top_n-1:-1][::-1]
    return dataset.iloc[similar_indices][['title', 'genre']]

def recommend_by_user_profile(dataset, user_profile, top_n=5):
    dataset['Genre_Score'] = dataset['genre'].fillna('').apply(
        lambda g: sum([user_profile['Genres'].get(genre, 0) for genre in g.split(', ')])
    )
    recommendations = dataset.sort_values(by='Genre_Score', ascending=False).head(top_n)
    return recommendations[['title', 'genre', 'Genre_Score']]

# Cargar datos (aquí se utiliza un dataset simulado)
@st.cache_data
def load_data():
    return pd.read_csv("peliculas.csv")
dataset = load_data()

# Crear la matriz TF-IDF
@st.cache_data
def compute_tfidf_matrix(dataset):
    dataset['synopsis'] = dataset['synopsis'].fillna('')
    vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
    return vectorizer.fit_transform(dataset['synopsis'])
tfidf_matrix = compute_tfidf_matrix(dataset)

# Almacenar datos de usuario en un archivo JSON
def load_user_data():
    try:
        with open("user_data.json", "r") as file:
            return json.load(file)
    except FileNotFoundError:
        return {"genres": {}, "ratings": {}}

def save_user_data(data):
    with open("user_data.json", "w") as file:
        json.dump(data, file)

user_data = load_user_data()

# Configuración de la app de Streamlit
st.title("Sistema de Recomendación de Películas")

# Selección de tipo de recomendación
st.sidebar.title("Opciones de Recomendación")
method = st.sidebar.selectbox("Selecciona un enfoque", ("Similitud de Contenido", "Perfil del Usuario"))

if method == "Similitud de Contenido":
    movie_title = st.sidebar.selectbox("Selecciona una película", dataset['title'])
    movie_index = dataset[dataset['title'] == movie_title].index[0]
    top_n = st.sidebar.slider("Número de recomendaciones", 1, 10, 5)

    if st.button("Generar Recomendaciones"):
        recommendations = recommend_by_content(tfidf_matrix, movie_index, dataset, top_n)
        st.write("### Recomendaciones basadas en Similitud de Contenido")
        st.write(recommendations)

elif method == "Perfil del Usuario":
    recommendation_type = st.radio("Selecciona el tipo de recomendación", ("Por géneros favoritos", "Por valoraciones de películas"))

    if recommendation_type == "Por géneros favoritos":
        st.write("### Selecciona tus géneros favoritos")
        genres = dataset['genre'].str.split(', ').explode().unique()
        selected_genres = st.multiselect("Selecciona géneros", genres, default=list(user_data['genres'].keys()))

        for genre in selected_genres:
            user_data['genres'][genre] = user_data['genres'].get(genre, 0) + 1

        save_user_data(user_data)

        st.write("### Generando recomendaciones basadas en géneros seleccionados...")
        user_profile = {"Genres": user_data['genres']}
        top_n = st.sidebar.slider("Número de recomendaciones", 1, 10, 5)

        if st.button("Generar Recomendaciones por Géneros"):
            recommendations = recommend_by_user_profile(dataset, user_profile, top_n)
            st.write("### Recomendaciones basadas en tus géneros favoritos")
            st.write(recommendations)

    elif recommendation_type == "Por valoraciones de películas":
        st.write("### Valora las películas para mejorar las recomendaciones")
        selected_movie = st.selectbox("Busca y selecciona una película para valorar", dataset['title'].unique())
        if selected_movie:
            row = dataset[dataset['title'] == selected_movie].iloc[0]
            idx = row.name
            title = row['title']
            rating = st.slider(f"Valora {title}", 0, 5, int(user_data['ratings'].get(title, 0)), key=f"rating_{idx}")
            if rating > 0:
                user_data['ratings'][title] = rating

        save_user_data(user_data)

        st.write("### Generando recomendaciones basadas en valoraciones...")
        top_n = st.sidebar.slider("Número de recomendaciones", 1, 10, 5)

        if st.button("Generar Recomendaciones por Valoraciones"):
            high_rated_movies = [title for title, rating in user_data['ratings'].items() if rating >= 4]
            recommendations = pd.DataFrame()
            for movie_title in high_rated_movies:
                if movie_title in dataset['title'].values:
                    movie_index = dataset[dataset['title'] == movie_title].index[0]
                    recs = recommend_by_content(tfidf_matrix, movie_index, dataset, top_n)
                    recommendations = pd.concat([recommendations, recs]).drop_duplicates().head(top_n)

            st.write("### Recomendaciones basadas en tus valoraciones")
            st.write(recommendations)
    st.write("### Selecciona tus géneros favoritos")
    genres = dataset['genre'].str.split(', ').explode().unique()
    selected_genres = st.multiselect("Selecciona géneros", genres, default=list(user_data['genres'].keys()))

    for genre in selected_genres:
        user_data['genres'][genre] = user_data['genres'].get(genre, 0) + 1

    st.write("### Valora las películas para mejorar las recomendaciones")
    selected_movie = st.selectbox("Busca y selecciona una película para valorar", dataset['title'].unique())
    if selected_movie:
        row = dataset[dataset['title'] == selected_movie].iloc[0]
        idx = row.name
        title = row['title']
        rating = st.slider(f"Valora {title}", 0, 5, int(user_data['ratings'].get(title, 0)), key=f"rating_{idx}")
        if rating > 0:
            user_data['ratings'][title] = rating

    save_user_data(user_data)

    st.write("### Generando recomendaciones basadas en tu perfil...")
    user_profile = {"Genres": user_data['genres']}
    top_n = st.sidebar.slider("Número de recomendaciones", 1, 10, 5)

    if st.button("Generar Recomendaciones"):
        recommendations = recommend_by_user_profile(dataset, user_profile, top_n)
        st.write("### Recomendaciones basadas en tu perfil")
        st.write(recommendations)
