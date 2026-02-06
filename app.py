import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Chargement des données
@st.cache_data
def load_data():
    movies = pd.read_csv("data/movies.csv", sep=";")
    ratings = pd.read_csv("data/rating.csv", sep=";")
    return movies, ratings

movies, ratings = load_data()

@st.cache_data
def build_tfidf(movies):
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(movies['genres'])

    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    return cosine_sim

cosine_sim = build_tfidf(movies)

@st.cache_data
def build_similarity(ratings):
    user_movie_matrix = ratings.pivot_table(
        index='userId',
        columns='movieId',
        values='rating'
    ).fillna(0)

    movie_similarity = cosine_similarity(user_movie_matrix.T)

    movie_similarity_df = pd.DataFrame(
        movie_similarity,
        index=user_movie_matrix.columns,
        columns=user_movie_matrix.columns
    )
    return movie_similarity_df

movie_similarity_df = build_similarity(ratings)

# Fonction de recommandation

def recommend_movies_hybrid(movie_title, n=10, w_content=0.5, w_collaborative=0.5):

    if movie_title not in movies['title'].values:
        return "Film non trouvé"

    # index pour content-based
    movie_idx = movies[movies['title'] == movie_title].index[0]

    # movieId pour collaborative
    movie_id = movies.loc[movie_idx, 'movieId']

    #  SCORES CONTENT 
    content_scores = pd.Series(
        cosine_sim[movie_idx],
        index=movies['movieId']
    )

    #  SCORES COLLABORATIVE 
    if movie_id in movie_similarity_df.columns:
        collaborative_scores = movie_similarity_df[movie_id]
    else:
        collaborative_scores = pd.Series(0, index=movies['movieId'])

    #  SCORE HYBRIDE 
    hybrid_scores = (
        w_content * content_scores +
        w_collaborative * collaborative_scores
    )

    # enlever le film lui-même
    hybrid_scores = hybrid_scores.drop(movie_id)

    # top N
    top_movies = hybrid_scores.sort_values(ascending=False).head(n)

    return movies[movies['movieId'].isin(top_movies.index)][['title', 'genres']]


# Interface Streamlit

st.set_page_config(page_title="Movie Recommender", layout="centered")

st.title("Système de recommandation de films")
st.write("Basé sur les **notes des utilisateurs** (Collaborative Filtering)")

movie_selected = st.selectbox(
    "Choisis un film :",
    sorted(movies['title'].unique())
)

if st.button("Recommander"):
    recommendations = recommend_movies_hybrid(movie_selected)

    if recommendations is None:
        st.error("Film non trouvé")
    else:
        st.subheader("Films recommandés")
        st.dataframe(recommendations.reset_index(drop=True))
