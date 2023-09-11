import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# Step 1: Load the movie dataset (assuming you have a CSV file)
# Replace 'movies.csv' with your actual dataset file.
df = pd.read_csv('movies.csv')

# Step 2: Data Preprocessing (if needed)

# Step 3: Feature Engineering (TF-IDF on movie genres)
tfidf_vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf_vectorizer.fit_transform(df['movie_genre'])

# Step 4: Similarity Calculation (cosine similarity)
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

# Step 5: User Input
user_movie = "The Birth of a Nation" #example

# Step 6: Get the index of the user's input movie
movie_index = df[df['movie_name'] == user_movie].index[0]

# Step 7: Get movie recommendations based on similarity
similar_movies_indices = list(enumerate(cosine_sim[movie_index]))
similar_movies_indices = sorted(similar_movies_indices, key=lambda x: x[1], reverse=True)
top_similar_movies_indices = similar_movies_indices[1:11]  # Exclude the user's input

# Step 8: Display top recommended movies
print(f"Top 10 Movies Similar to '{user_movie}':")
for idx, sim_score in top_similar_movies_indices:
    print(f"{df.iloc[idx]['movie_name']} (Similarity Score: {sim_score:.2f})")

