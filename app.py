import streamlit as st
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# Chargement des données
@st.cache_data
def load_data():
    movies = pd.read_csv("data/movies.csv", sep=";")
    ratings = pd.read_csv("data/rating.csv", sep=";")
    return movies, ratings

movies, ratings = load_data()

# Création de la matrice collaborative
@st.cache_data
def build_similarity(ratings):
    movie_user_matrix = ratings.pivot_table(
        index='movieId',
        columns='userId',
        values='rating'
    ).fillna(0)

    similarity = cosine_similarity(movie_user_matrix)
    
    similarity_df = pd.DataFrame(
        similarity,
        index=movie_user_matrix.index,
        columns=movie_user_matrix.index
    )
    return similarity_df

movie_similarity_df = build_similarity(ratings)

# Fonction de recommandation

def recommend_movies_collaborative(movie_title, n=10):

    if movie_title not in movies['title'].values:
        return None

    movie_id = movies[movies['title'] == movie_title]['movieId'].values[0]

    sim_scores = movie_similarity_df[movie_id].sort_values(ascending=False)
    sim_scores = sim_scores.iloc[1:n+1]

    return movies[movies['movieId'].isin(sim_scores.index)][['title', 'genres']]


# Interface Streamlit

st.set_page_config(page_title="Movie Recommender", layout="centered")

st.title("Système de recommandation de films")
st.write("Basé sur les **notes des utilisateurs** (Collaborative Filtering)")

movie_selected = st.selectbox(
    "Choisis un film :",
    sorted(movies['title'].unique())
)

if st.button("Recommander"):
    recommendations = recommend_movies_collaborative(movie_selected)

    if recommendations is None:
        st.error("Film non trouvé")
    else:
        st.subheader("Films recommandés")
        st.dataframe(recommendations.reset_index(drop=True))
