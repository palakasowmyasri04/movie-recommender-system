import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load dataset
movies = pd.read_csv('data/tmdb_5000_movies.csv')

# Select important columns
movies = movies[['title', 'overview']]

# Fill missing values
movies.fillna('', inplace=True)

# Convert text into vectors
cv = CountVectorizer(max_features=5000, stop_words='english')
vectors = cv.fit_transform(movies['overview']).toarray()

# Calculate similarity
similarity = cosine_similarity(vectors)

# Recommendation function
def recommend(movie):
    movie = movie.lower()

    if movie not in movies['title'].str.lower().values:
        print("Movie not found")
        return

    index = movies[movies['title'].str.lower() == movie].index[0]

    distances = list(enumerate(similarity[index]))

    movies_list = sorted(distances, reverse=True, key=lambda x: x[1])[1:6]

    print("\nRecommended Movies:")
    for i in movies_list:
        print(movies.iloc[i[0]].title)

# Test recommendation
movie_name = input("Enter movie name: ")
recommend(movie_name)