import pandas as pd
import streamlit as st
from sentence_transformers import SentenceTransformer, util
import requests


# ------------------- CONFIG -------------------
st.set_page_config(page_title="🎬 Smart Movie Recommender", layout="centered")

# ------------------- STYLES -------------------
st.markdown("""
    <style>
    body {
        background-color: #f0f2f6;
    }
    .main {
        padding-top: 2rem;
    }
    h1 {
        font-family: 'Helvetica', sans-serif;
        font-size: 40px;
        color: #FF4B4B;
        text-align: center;
    }
    .stTextInput>div>div>input {
        font-size: 16px;
    }
    .recommendation {
        background-color: #00000;
        padding: 10px 20px;
        border-radius: 12px;
        margin-bottom: 10px;
        box-shadow: 0 4px 6px rgba(255, 75, 75, 0.1);
    }
    </style>
""", unsafe_allow_html=True)

# ------------------- LOADERS -------------------
@st.cache_data
def load_data():
    movies = pd.read_csv("Movies_Cleaned_1.csv")
    movies['tags'] = movies['tags'].str.lower()
    return movies

@st.cache_resource
def load_model_and_embeddings(movies):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(movies['tags'].tolist(), convert_to_tensor=True)
    return model, embeddings

def load_lottie_url(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

# ------------------- RECOMMENDATION -------------------
def recommend_with_genre_filter(movie_input, model, embeddings, movies):
    movie_input = movie_input.lower().strip()
    matched_movies = movies[movies['title'].str.lower() == movie_input]

    if matched_movies.empty:
        input_text = movie_input
        input_genres = []
    else:
        input_text = matched_movies.iloc[0]['tags']
        input_genres = matched_movies.iloc[0]['genres']

    input_embedding = model.encode(input_text, convert_to_tensor=True)
    scores = util.pytorch_cos_sim(input_embedding, embeddings)[0]
    top_indices = scores.argsort(descending=True).tolist()

    recommendations = []
    for idx in top_indices:
        candidate_genres = movies.iloc[idx]['genres']
        if candidate_genres and input_genres:
            overlap = set(input_genres).intersection(set(candidate_genres))
            if len(overlap) == 0:
                continue
        title = movies.iloc[idx]['title']
        if title.lower() != movie_input:
            recommendations.append(title)
        if len(recommendations) == 5:
            break
    return recommendations

# ------------------- MAIN APP -------------------
movies = load_data()
model, embeddings = load_model_and_embeddings(movies)

with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/812/812807.png", width=100)
    st.title("🎥 About")
    st.markdown("""
    This is a **Smart Movie Recommender** powered by:
    - AI embeddings (Sentence Transformers)
    - Semantic similarity

    **Instructions:**
    - Enter a movie title (works best with Hollywood titles)
    - Optional: Filter by genre

    """)

with st.container():
    st.markdown("<h1>🎬 Smart Movie Recommender</h1>", unsafe_allow_html=True)

st.markdown("---")

# ------------------- INPUT -------------------
movie_input = st.text_input("📽️ Enter a movie name or concept (e.g., 'Harry Potter')")

if st.button("🔍 Recommend"):
    if not movie_input:
        st.warning("⚠ Please enter a movie title or keyword.")
    else:
        with st.spinner("🔎 Finding similar movies..."):
            results = recommend_with_genre_filter(movie_input, model, embeddings, movies)

        if results:
            st.success("You might also like:")
            for idx, title in enumerate(results, 1):
                st.markdown(f"<div class='recommendation'><strong>{idx}. {title}</strong></div>", unsafe_allow_html=True)
        else:
            st.warning("Sorry, no similar movies found.")
