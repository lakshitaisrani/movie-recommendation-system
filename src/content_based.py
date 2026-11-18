
import pickle
from pathlib import Path


# --------------------------------------------------
# 1. Loading the saved model files
# --------------------------------------------------

# Find the main project directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Define paths to the saved files
MOVIES_PATH = BASE_DIR / "models" / "movies.pkl"
SIMILARITY_PATH = BASE_DIR / "models" / "similarity.pkl"


# Load the processed movie dataset
with open(MOVIES_PATH, "rb") as file:
    movies = pickle.load(file)

# Load the cosine similarity matrix
with open(SIMILARITY_PATH, "rb") as file:
    similarity_matrix = pickle.load(file)


# --------------------------------------------------
# 2. Recommendation function
# --------------------------------------------------

def recommend(movie):

    # Check whether the movie exists in our dataset
    if movie not in movies["title"].values:
        print("Movie not found in our dataset!")
        return []

    # Find the index of the selected movie
    movie_index = movies[movies["title"] == movie].index[0]

    # Get similarity scores for the selected movie
    distances = similarity_matrix[movie_index]

    # Sort movies according to their similarity scores
    movie_list = sorted(
        list(enumerate(distances)),
        reverse=True,
        key=lambda x: x[1]
    )[1:11]

    # Store the recommended movie titles
    recommendations = []

    for i in movie_list:
        recommendations.append(movies.iloc[i[0]]["title"])

    return recommendations


# --------------------------------------------------
# 3. Testing the recommendation function
# --------------------------------------------------

if __name__ == "__main__":

    movie_name = "The Dark Knight"

    recommended_movies = recommend(movie_name)

    print(f"\nMovies similar to {movie_name}:\n")

    for movie in recommended_movies:
        print(movie)

