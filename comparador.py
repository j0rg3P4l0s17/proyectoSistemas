import ttkbootstrap as tb
import tkinter as tk
from ttkbootstrap.constants import *
from tkinter import ttk, messagebox
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd
import json
import os

# Archivo JSON para almacenar usuarios y valoraciones
USER_DATA_FILE = "usuarios.json"

# --------------------------
# Funciones de Manejo de JSON
# --------------------------
def load_users():
    """ Cargar datos de usuarios desde JSON """
    if os.path.exists(USER_DATA_FILE):
        with open(USER_DATA_FILE, 'r') as file:
            try:
                return json.load(file)
            except json.JSONDecodeError:
                return {}  # Si el archivo está vacío o corrupto
    return {}

def save_users(users):
    """ Guardar datos de usuarios en JSON """
    with open(USER_DATA_FILE, 'w') as file:
        json.dump(users, file, indent=4)

# --------------------------
# Cargar y Validar Datos CSV
# --------------------------
def load_data():
    file_path = 'peliculas.csv'
    if not os.path.exists(file_path):
        messagebox.showerror("Error", "El archivo peliculas.csv no existe.")
        exit()
    try:
        df = pd.read_csv(file_path)
        required_columns = {'title', 'genre', 'director'}
        if not required_columns.issubset(df.columns):
            raise ValueError("El archivo CSV debe contener las columnas: title, genre, director.")
        return df
    except Exception as e:
        messagebox.showerror("Error", f"Error al leer el archivo CSV: {e}")
        exit()

# --------------------------
# Actualizar Perfil del Usuario
# --------------------------
def update_user_profile(email, movie_title, rating):
    users = load_users()
    if email not in users:
        users[email] = {"ratings": {}, "profile": {"genres": {}}}

    # Verificar si la película existe
    matching_movies = df[df['title'].str.strip().str.lower() == movie_title.strip().lower()]
    if matching_movies.empty:
        messagebox.showerror("Error", f"No se encontró la película: '{movie_title}'.")
        return

    # Actualizar valoraciones
    movie = matching_movies.iloc[0]
    users[email]["ratings"][movie['title']] = rating

    # Actualizar perfil de géneros
    genres = movie['genre'].split(", ") if pd.notna(movie['genre']) else []
    for genre in genres:
        users[email]['profile']['genres'][genre] = users[email]['profile']['genres'].get(genre, 0) + rating

    save_users(users)

# --------------------------
# Generar Recomendaciones
# --------------------------
def recommend_movies_based_on_profile(email):
    """ Genera recomendaciones usando similitud por coseno entre el perfil del usuario y las películas. """
    users = load_users()
    if email not in users or not users[email]['profile']['genres']:
        messagebox.showinfo("Info", "No hay suficientes valoraciones para generar recomendaciones.")
        return pd.DataFrame()

    # Obtener el perfil de usuario como un vector de géneros
    user_profile = users[email]['profile']['genres']

    # Crear una lista de géneros únicos en el dataset
    all_genres = set(genre.strip() for genres in df['genre'].dropna() for genre in genres.split(','))

    # Crear un vector de usuario con los géneros ponderados por sus puntuaciones
    user_vector = np.array([user_profile.get(genre, 0) for genre in all_genres]).reshape(1, -1)

    # Función para crear el vector de cada película basado en géneros
    def movie_to_vector(genres):
        genre_list = genres.split(',') if pd.notna(genres) else []
        return np.array([1 if genre in genre_list else 0 for genre in all_genres])

    # Construcción de la matriz de películas
    movie_vectors = np.array([movie_to_vector(genres) for genres in df['genre'].fillna('')])

    # Calcular similitud por coseno entre usuario y películas
    similarities = cosine_similarity(user_vector, movie_vectors)[0]

    df['Cosine_Similarity'] = similarities

    # Excluir películas que el usuario ya ha valorado
    user_rated_movies = set(users[email]['ratings'].keys())
    recommendations = df[~df['title'].isin(user_rated_movies)].sort_values(by='Cosine_Similarity', ascending=False).head(10)

    return recommendations[['title', 'genre', 'director', 'Cosine_Similarity']]
# --------------------------
# Ventana para Valorar Películas
# --------------------------
def rate_movies(email):
    def submit_rating():
        selected_movie = movie_combobox.get().strip()  # Eliminar espacios extras
        rating = rating_slider.get()

        if not selected_movie:
            messagebox.showerror("Error", "Por favor, selecciona o escribe una película válida.")
            return

        # Búsqueda flexible: normalizar nombres de películas
        matching_movies = df[df['title'].str.strip().str.lower() == selected_movie.lower()]

        if matching_movies.empty:
            messagebox.showerror("Error", f"La película '{selected_movie}' no existe en la base de datos. Verifica el nombre.")
            return

        # Obtener el título exacto desde el DataFrame para evitar problemas
        exact_movie_title = matching_movies.iloc[0]['title']

        if rating <= 0:
            messagebox.showerror("Error", "La valoración debe ser mayor que 0.")
            return

        # Actualizar perfil del usuario
        update_user_profile(email, exact_movie_title, int(rating))
        messagebox.showinfo("Éxito", f"Valoración guardada: {exact_movie_title} -> {int(rating)} estrellas")
        rate_window.destroy()

    # Crear la ventana de valoración
    rate_window = tb.Toplevel(root)
    rate_window.title("Valorar Película")
    rate_window.geometry("800x500")

    ttk.Label(rate_window, text="Buscar o escribir una película:").pack(pady=10)

    # Configurar el Combobox para búsqueda flexible
    movie_titles = df['title'].dropna().unique().tolist()
    movie_combobox = ttk.Combobox(rate_window, values=movie_titles, width=40)
    movie_combobox.pack(pady=5)

    # Permitir autocompletado (opcional si no soportado en ttk.Combobox)
    movie_combobox.bind("<Return>", lambda event: submit_rating())

    ttk.Label(rate_window, text="Valora la película (1-5):").pack(pady=10)
    rating_slider = ttk.Scale(rate_window, from_=1, to=5, orient=HORIZONTAL, length=300)
    rating_slider.set(3)  # Valor inicial predeterminado
    rating_slider.pack(pady=5)

    # Botón para guardar la valoración
    tb.Button(rate_window, text="Guardar Valoración", command=submit_rating, bootstyle=SUCCESS).pack(pady=10)

# --------------------------
# Ventana Principal de Recomendaciones
# --------------------------
def main_recommendations_window(email):
    users = load_users()
    user_name = users[email]["name"]  # Recuperar el nombre del usuario desde el JSON

    def logout():
        main_window.destroy()
        root.deiconify()  # Volver a mostrar la ventana principal

    def generate_recommendations():
        recommendations = recommend_movies_based_on_profile(email)

        # Limpiar el contenido previo del scrollable frame
        for widget in scrollable_frame.winfo_children():
            widget.destroy()

        # Mostrar las recomendaciones
        if recommendations.empty:
            ttk.Label(scrollable_frame, text="No se encontraron recomendaciones. Valora más películas.",
                      font=("Arial", 12), foreground="red").pack(pady=5)
        else:
            for _, row in recommendations.iterrows():
                ttk.Label(scrollable_frame, text=f"🎬 {row['title']}", font=("Arial", 12, "bold")).pack(anchor="center", pady=5)
                ttk.Label(scrollable_frame, text=f"Géneros: {row['genre']}", font=("Arial", 10)).pack(anchor="center")
                ttk.Label(scrollable_frame, text=f"Director: {row['director']}", font=("Arial", 10)).pack(anchor="center")
                ttk.Separator(scrollable_frame, orient="horizontal").pack(fill="x", pady=5)

    # Crear ventana principal más grande
    main_window = tb.Toplevel(root)
    main_window.title("Recomendador Personalizado de Películas")
    main_window.geometry("1200x800")

    # Mostrar el nombre del usuario
    ttk.Label(main_window, text=f"🎥 Bienvenido, {user_name}", font=("Arial", 18)).pack(pady=20)

    # Botón para generar recomendaciones
    tb.Button(main_window, text="Generar Recomendaciones", command=generate_recommendations,
              bootstyle=PRIMARY, width=30).pack(pady=10)

    # Contenedor centrado con scrollable frame
    container = ttk.Frame(main_window)
    container.pack(expand=True, fill="both", padx=20, pady=20)

    canvas = tk.Canvas(container, highlightthickness=0)  # Sin bordes adicionales
    scrollbar = ttk.Scrollbar(container, orient="vertical", command=canvas.yview)
    
    scrollable_frame = ttk.Frame(canvas)
    scrollable_frame.bind(
        "<Configure>",
        lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
    )

    # Centrar contenido horizontalmente dentro del canvas
    canvas.create_window((0, 0), window=scrollable_frame, anchor="n", width=1100)

    canvas.configure(yscrollcommand=scrollbar.set)

    # Ajustar el canvas y la barra de desplazamiento dentro del contenedor
    canvas.pack(side="left", fill="both", expand=True)
    scrollbar.pack(side="right", fill="y")

    # Botón para valorar películas
    tb.Button(main_window, text="Valorar Películas", command=lambda: rate_movies(email),
              bootstyle=INFO, width=30).pack(pady=10)

    # Botón para cerrar sesión
    tb.Button(main_window, text="Cerrar Sesión", command=logout, bootstyle=SECONDARY, width=30).pack(pady=10)

    # Botón para salir
    tb.Button(main_window, text="Salir", command=main_window.destroy, bootstyle=DANGER, width=30).pack(pady=10)

# --------------------------
# Funciones de Login y Registro
# --------------------------
def login():
    def validate_login():
        email = entry_email.get()
        password = entry_password.get()
        users = load_users()

        if email in users and users[email]["password"] == password:
            messagebox.showinfo("Éxito", f"Bienvenido {users[email]['name']}!")
            login_window.destroy()
            root.withdraw()  # Ocultar ventana principal
            main_recommendations_window(email)
        else:
            messagebox.showerror("Error", "Correo o contraseña incorrectos.")

    login_window = tb.Toplevel(root)
    login_window.title("Iniciar Sesión")
    login_window.geometry("600x400")

    ttk.Label(login_window, text="Correo:").pack(pady=5)
    entry_email = ttk.Entry(login_window)
    entry_email.pack(pady=5)

    ttk.Label(login_window, text="Contraseña:").pack(pady=5)
    entry_password = ttk.Entry(login_window, show="*")
    entry_password.pack(pady=5)

    tb.Button(login_window, text="Iniciar Sesión", command=validate_login, bootstyle=SUCCESS).pack(pady=10)

def create_user():
    def save_new_user():
        name = entry_name.get()
        email = entry_email.get()
        password = entry_password.get()
        users = load_users()

        if email in users:
            messagebox.showerror("Error", "El correo ya está registrado.")
        else:
            users[email] = {"name": name, "password": password, "ratings": {}, "profile": {"genres": {}}}
            save_users(users)
            messagebox.showinfo("Éxito", "Usuario registrado correctamente.")
            create_user_window.destroy()

    create_user_window = tb.Toplevel(root)
    create_user_window.title("Crear Usuario")
    create_user_window.geometry("300x250")

    ttk.Label(create_user_window, text="Nombre:").pack(pady=5)
    entry_name = ttk.Entry(create_user_window)
    entry_name.pack(pady=5)

    ttk.Label(create_user_window, text="Correo:").pack(pady=5)
    entry_email = ttk.Entry(create_user_window)
    entry_email.pack(pady=5)

    ttk.Label(create_user_window, text="Contraseña:").pack(pady=5)
    entry_password = ttk.Entry(create_user_window, show="*")
    entry_password.pack(pady=5)

    tb.Button(create_user_window, text="Guardar", command=save_new_user, bootstyle=SUCCESS).pack(pady=10)

# --------------------------
# Ventana Principal
# --------------------------
root = tb.Window(themename="superhero")
root.title("Sistema de Recomendación")
root.geometry("600x400")

ttk.Label(root, text="Sistema de Recomendación de Películas", font=("Arial", 12)).pack(pady=10)
tb.Button(root, text="Crear Usuario", command=create_user, bootstyle=INFO).pack(pady=5)
tb.Button(root, text="Iniciar Sesión", command=login, bootstyle=SUCCESS).pack(pady=5)
tb.Button(root, text="Salir", command=root.quit, bootstyle=DANGER).pack(pady=5)

# Cargar datos
df = load_data()
root.mainloop()
