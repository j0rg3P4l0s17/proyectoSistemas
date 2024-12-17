import tkinter as tk
from tkinter import ttk, messagebox
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
import json
import os

# Archivo JSON para almacenar usuarios
USER_DATA_FILE = "usuarios.json"

# Funciones para manejo de usuarios
def load_users():
    if os.path.exists(USER_DATA_FILE):
        with open(USER_DATA_FILE, 'r') as file:
            return json.load(file)
    return {}

def save_users(users):
    with open(USER_DATA_FILE, 'w') as file:
        json.dump(users, file, indent=4)

def create_user():
    def save_new_user():
        name = entry_name.get()
        email = entry_email.get()
        password = entry_password.get()
        users = load_users()
        if email in users:
            messagebox.showerror("Error", "El correo ya está registrado.")
        else:
            users[email] = {"name": name, "password": password}
            save_users(users)
            messagebox.showinfo("Éxito", "Usuario registrado correctamente.")
            create_user_window.destroy()

    create_user_window = tk.Toplevel(root)
    create_user_window.title("Crear Usuario")
    create_user_window.geometry("300x200")

    ttk.Label(create_user_window, text="Nombre:").pack(pady=5)
    entry_name = ttk.Entry(create_user_window)
    entry_name.pack(pady=5)

    ttk.Label(create_user_window, text="Correo:").pack(pady=5)
    entry_email = ttk.Entry(create_user_window)
    entry_email.pack(pady=5)

    ttk.Label(create_user_window, text="Contraseña:").pack(pady=5)
    entry_password = ttk.Entry(create_user_window, show="*")
    entry_password.pack(pady=5)

    ttk.Button(create_user_window, text="Guardar", command=save_new_user).pack(pady=10)

def login():
    def validate_login():
        email = entry_email.get()
        password = entry_password.get()
        users = load_users()
        if email in users and users[email]['password'] == password:
            messagebox.showinfo("Éxito", f"Bienvenido {users[email]['name']}!")
            login_window.destroy()
            main_recommendations_window()
        else:
            messagebox.showerror("Error", "Correo o contraseña incorrectos.")

    login_window = tk.Toplevel(root)
    login_window.title("Iniciar Sesión")
    login_window.geometry("300x150")

    ttk.Label(login_window, text="Correo:").pack(pady=5)
    entry_email = ttk.Entry(login_window)
    entry_email.pack(pady=5)

    ttk.Label(login_window, text="Contraseña:").pack(pady=5)
    entry_password = ttk.Entry(login_window, show="*")
    entry_password.pack(pady=5)

    ttk.Button(login_window, text="Iniciar Sesión", command=validate_login).pack(pady=10)

# Datos simulados y configuración inicial
def load_data():
    """ Cargar y preprocesar los datos. """
    global genre_matrix, director_matrix, collection_matrix, df, user_profile_vector, combined_features
    file_path = 'peliculas.csv'
    df = pd.read_csv(file_path)

    df_attributes = df[['genre', 'director', 'view_the_collection']].fillna('')
    for col in ['genre', 'director', 'view_the_collection']:
        df_attributes[col] = df_attributes[col].apply(lambda x: [i.strip() for i in x.split(',') if i])

    mlb_genre, mlb_director, mlb_collection = MultiLabelBinarizer(), MultiLabelBinarizer(), MultiLabelBinarizer()
    genre_matrix = mlb_genre.fit_transform(df_attributes['genre'])
    director_matrix = mlb_director.fit_transform(df_attributes['director'])
    collection_matrix = mlb_collection.fit_transform(df_attributes['view_the_collection'])

    combined_features = np.hstack([genre_matrix, director_matrix, collection_matrix])
    user_profile_vector = np.random.rand(1, combined_features.shape[1])

def recommend_movies():
    cosine_similarities = cosine_similarity(user_profile_vector, combined_features)
    similarity_scores = cosine_similarities.flatten()
    recommended_indices = similarity_scores.argsort()[::-1]
    top_movies = df.loc[recommended_indices, ['title', 'genre', 'director']].head(10)

    results_text.set('')
    for _, row in top_movies.iterrows():
        results_text.set(results_text.get() + f"{row['title']} \nGéneros: {row['genre']} \nDirector: {row['director']}\n\n")

def main_recommendations_window():
    main_window = tk.Toplevel(root)
    main_window.title("Recomendador de Películas")
    main_window.geometry("600x400")

    welcome_label = ttk.Label(main_window, text="Generador de Recomendaciones", font=("Arial", 14))
    welcome_label.pack(pady=10)

    recommend_button = ttk.Button(main_window, text="Generar Recomendaciones", command=recommend_movies)
    recommend_button.pack(pady=10)

    global results_text
    results_text = tk.StringVar()
    results_label = ttk.Label(main_window, textvariable=results_text, wraplength=550, justify="left")
    results_label.pack(padx=10, pady=10)

    exit_button = ttk.Button(main_window, text="Salir", command=main_window.destroy)
    exit_button.pack(pady=10)

# Configuración GUI principal
root = tk.Tk()
root.title("Sistema de Usuarios y Recomendaciones")
root.geometry("300x200")

# Botones principales
ttk.Button(root, text="Crear Usuario", command=create_user).pack(pady=10)
ttk.Button(root, text="Iniciar Sesión", command=login).pack(pady=10)
ttk.Button(root, text="Salir", command=root.quit).pack(pady=10)

# Cargar datos
load_data()
root.mainloop()
