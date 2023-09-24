import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import hstack

# Load the dataset
dataset = pd.read_csv("movies_dataset.csv")

# Vectorize text features: genre, director, and cast
tfidf_vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix_genre = tfidf_vectorizer.fit_transform(dataset['genre'])
tfidf_matrix_director = tfidf_vectorizer.fit_transform(dataset['director'])
tfidf_matrix_cast = tfidf_vectorizer.fit_transform(dataset['cast'])

# Combine vectorized features into one matrix
tfidf_matrix_combined = hstack((tfidf_matrix_genre, tfidf_matrix_director, tfidf_matrix_cast))

# Compute cosine similarity matrix
cosine_sim = cosine_similarity(tfidf_matrix_combined, tfidf_matrix_combined)

#save
cosine_sim_df = pd.DataFrame(cosine_sim, columns=dataset['title'], index=dataset['title'])
cosine_sim_df.to_csv('cosine_similarity_matrix.csv')

# Function to get top N similar movies for a given movie title
def get_similar_movies(movie_title, n=10):
    movie_index = dataset[dataset['title'] == movie_title].index[0]
    similar_scores = list(enumerate(cosine_sim[movie_index]))
    similar_scores = sorted(similar_scores, key=lambda x: x[1], reverse=True)
    similar_movies = [dataset.iloc[score[0]]['title'] for score in similar_scores[1:n+1]]
    return similar_movies

# Example: Find top 10 similar movies to 'The Godfather'
similar_movies = get_similar_movies('Stoned', n=10)
print("Top 10 similar movies to 'Stoned':")
for movie in similar_movies:
    print(movie)
