"I started with Movie Recommendation because all other item recommendations share similar approaches and vary in features."

# Features
**Cosine Similarity**: The system calculates the cosine similarity between movies to determine their similarity.

**Vectorization**: Text features like genre, director, and cast are vectorized using TF-IDF (Term Frequency-Inverse Document Frequency) to create feature representations.

**Customizable**: You can easily customize the dataset and add more features as needed.

# Requirements
Python 3.x
Required Python libraries: pandas, scikit-learn, numpy

# Usage
Clone this repository to your local machine.
git clone [GitHub]( https://github.com/your-username/movie-recommendation.git)

Install the required Python libraries if you haven't already.

```python
pip install pandas scikit-learn numpy
```

Prepare your movie dataset in a **CSV file** named movies_dataset.csv with columns: title, genre, year of release, IMDb rate, director, and cast.

Run the recommendation system script.

```python
python movie_recommendation.py
```

Enter a movie title when prompted, and the system will display the top 10 similar movies.

# Example


```python
Enter a movie title: The Godfather

Top 10 similar movies to 'The Godfather':
1. The Godfather Part II
2. Goodfellas
3. Scarface
4. Casino
5. The Untouchables
6. Donnie Brasco
7. Heat
8. Once Upon a Time in America
9. The Departed
10. A Bronx Tale
