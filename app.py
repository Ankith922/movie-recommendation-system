import streamlit as st
import pandas as pd
import math
import re
from collections import Counter
import os

# Set page configuration
st.set_page_config(page_title="Movie Recommendation System", layout="wide")

st.title("🎬 Movie Recommendation System")
st.write("Select or enter a movie you like, and we'll recommend 5 similar movies based on content similarity!")

def compute_tfidf_cosine_sim(corpus, selected_idx):
    """
    Pure Python implementation of TF-IDF and Cosine Similarity
    to avoid C-extension / DLL loading issues on restricted systems.
    """
    # 1. Basic tokenization
    def tokenize(text):
        return re.findall(r'\b\w+\b', text.lower())
    
    tokenized_corpus = [tokenize(doc) for doc in corpus]
    
    # 2. Compute Term Frequency (TF)
    tf_list = []
    doc_freq = Counter()
    for tokens in tokenized_corpus:
        counts = Counter(tokens)
        tf_list.append(counts)
        for term in counts.keys():
            doc_freq[term] += 1
            
    num_docs = len(corpus)
    
    # 3. Compute IDF (Inverse Document Frequency)
    idf = {}
    for term, df in doc_freq.items():
        # Using standard smoothing: log((1+N)/(1+df)) + 1
        idf[term] = math.log((1 + num_docs) / (1 + df)) + 1
        
    # 4. Compute TF-IDF vectors
    tfidf_vectors = []
    for tf in tf_list:
        vec = {}
        norm = 0.0
        for term, count in tf.items():
            val = count * idf[term]
            vec[term] = val
            norm += val * val
        norm = math.sqrt(norm)
        
        # Normalize vector
        if norm > 0:
            for term in vec:
                vec[term] /= norm
        tfidf_vectors.append(vec)
        
    # 5. Compute Cosine Similarity between selected_idx and all others
    selected_vec = tfidf_vectors[selected_idx]
    sim_scores = []
    for idx, vec in enumerate(tfidf_vectors):
        dot_product = sum(selected_vec.get(term, 0.0) * val for term, val in vec.items())
        sim_scores.append((idx, dot_product))
        
    return sim_scores

# Load the dataset
@st.cache_data
def load_data():
    file_path = 'movies.csv'
    if not os.path.exists(file_path):
        return None
    
    try:
        df = pd.read_csv(file_path)
        # Handle missing values by replacing them with empty strings
        features = ['Genres', 'Keywords', 'Overview', 'Cast', 'Director']
        for feature in features:
            if feature in df.columns:
                df[feature] = df[feature].fillna('')
            else:
                # If a column is missing, add it as empty to prevent errors
                df[feature] = ''
                
        # Combine relevant features into a single string for each movie
        df['combined_features'] = df['Genres'] + ' ' + df['Keywords'] + ' ' + df['Overview'] + ' ' + df['Cast'] + ' ' + df['Director']
        return df
    except Exception as e:
        st.error(f"Error loading dataset: {e}")
        return None

df = load_data()

if df is None or df.empty:
    st.error("The dataset (`movies.csv`) could not be loaded or is empty. Please make sure the file exists and is in the correct format.")
else:
    if 'Movie title' not in df.columns:
        st.error("The dataset must contain a 'Movie title' column.")
    else:
        # Create a list of movie titles
        movie_titles = df['Movie title'].tolist()
        
        # Movie selection search box
        selected_movie = st.selectbox("Select a movie from the dropdown:", [''] + movie_titles)
        
        # Recommend Movies button
        if st.button("Recommend Movies"):
            if not selected_movie:
                st.warning("Please select a movie first.")
            else:
                with st.spinner("Finding similar movies..."):
                    corpus = df['combined_features'].tolist()
                    
                    # Find the index of the selected movie
                    movie_idx = df[df['Movie title'] == selected_movie].index[0]
                    
                    # Compute similarities using pure Python mathematical implementation
                    sim_scores = compute_tfidf_cosine_sim(corpus, movie_idx)
                    
                    # Sort movies based on similarity scores in descending order
                    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
                    
                    # Get top 5 similar movies (excluding the movie itself at index 0)
                    top_5_movies = sim_scores[1:6]
                    
                    if not top_5_movies:
                        st.info("No recommendations found.")
                    else:
                        st.subheader(f"Top 5 movies similar to '{selected_movie}':")
                        for idx, score in top_5_movies:
                            recommended_title = df.iloc[idx]['Movie title']
                            # Display title and similarity score
                            st.write(f"- **{recommended_title}** *(Similarity: {score:.2f})*")
