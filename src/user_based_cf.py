
import pandas as pd
import pickle


class UserBasedCF:

    def __init__(self, user_movie_matrix, user_similarity):
        # Store the training user-movie matrix
        self.user_movie_matrix = user_movie_matrix

        # Store the precomputed user similarity matrix
        self.user_similarity = user_similarity

    def recommend(self, user_id, n_neighbours=300, n_recommendations=10):

        # Check whether the user exists
        if user_id not in self.user_movie_matrix.index:
            raise ValueError("User not found in training data.")

        # Get similarities of the target user with other users
        similarities = self.user_similarity.loc[user_id]

        # Exclude the target user
        similarities = similarities.drop(user_id)

        # Keep only positive similarities
        similarities = similarities[similarities > 0]

        # Select the top N neighbours
        neighbours = similarities.sort_values(
            ascending=False
        ).head(n_neighbours)

        # Get movies already rated by the target user
        user_ratings = self.user_movie_matrix.loc[user_id]

        # Identify unrated movies
        unrated_movies = user_ratings[user_ratings.isna()].index

        # Get ratings given by neighbours
        neighbour_ratings = self.user_movie_matrix.loc[neighbours.index]

        # Calculate similarity-weighted ratings
        weighted_ratings = neighbour_ratings.mul(neighbours, axis=0)

        numerator = weighted_ratings.sum(axis=0)

        denominator = neighbour_ratings.notna().mul(
            neighbours.abs(), axis=0
        ).sum(axis=0)

        # Calculate predicted ratings
        predicted_ratings = numerator / denominator

        # Keep only movies not already rated by the user
        predicted_ratings = predicted_ratings.loc[unrated_movies]

        # Remove movies without predictions
        predicted_ratings = predicted_ratings.dropna()

        # Count how many neighbours rated each movie
        support_count = neighbour_ratings.notna().sum(axis=0)

        # Require at least two neighbours to support a recommendation
        predicted_ratings = predicted_ratings[
            support_count.loc[predicted_ratings.index] >= 2
        ]

        # Select the highest predicted ratings
        recommendations = predicted_ratings.sort_values(
            ascending=False
        ).head(n_recommendations)

        return recommendations


def load_user_based_cf():

    # Load saved matrices
    with open("models/user_movie_matrix.pkl", "rb") as file:
        user_movie_matrix = pickle.load(file)

    with open("models/user_similarity.pkl", "rb") as file:
        user_similarity = pickle.load(file)

    return UserBasedCF(user_movie_matrix, user_similarity)
