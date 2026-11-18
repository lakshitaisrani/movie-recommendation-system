
# Movie Recommendation System

A movie recommendation system built using Python and Machine Learning techniques. This project explores different recommendation approaches, starting with content-based filtering and gradually expanding towards collaborative and hybrid recommendation systems.

## Project Overview

The goal of this project is to build a movie recommendation system that suggests movies to users based on their similarities.

The first phase implements **Content-Based Filtering**, which recommends movies by analysing information such as:

- Movie overview
- Genres
- Keywords
- Cast
- Directors

## Current Implementation: Content-Based Filtering

The current model uses the following workflow:

1. Load and explore the TMDB 5000 Movies and Credits datasets.
2. Clean and preprocess the data.
3. Extract movie genres, keywords, cast and directors.
4. Combine movie information into a single text feature called `tags`.
5. Convert text into numerical vectors using `CountVectorizer`.
6. Apply text stemming using NLTK's `PorterStemmer`.
7. Calculate movie similarity using cosine similarity.
8. Recommend the top 10 similar movies based on a selected movie.

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- NLTK
- Jupyter Notebook
- Pickle

## Project Structure

```text
movie-recommendation-system/
│
├── data/
│   ├── tmdb_5000_movies.csv
│   └── tmdb_5000_credits.csv
│
├── notebooks/
│   ├── 01_content_based_learning.ipynb
│   └── 02_content_based_clean.ipynb
│
├── models/
│   ├── movies.pkl
│   └── similarity.pkl
│
├── src/
│   └── content_based.py
│
├── requirements.txt
├── README.md
└── .gitignore
```

## Dataset

This project uses the TMDB 5000 Movies Dataset.

The dataset contains movie information, including titles, overviews, genres, keywords, cast and crew.

Dataset source: [TMDB 5000 Movie Dataset on Kaggle](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata)

Download the dataset files and place them inside the `data/` directory.

## Installation and Setup

### 1. Clone the repository

```bash
git clone <your-repository-url>
cd movie-recommendation-system
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Prepare the datasets

Download the TMDB datasets and place the CSV files inside the `data/` folder.

### 4. Generate the model files

Run the clean notebook:

`notebooks/02_content_based_clean.ipynb`

Execute its cells to preprocess the data, generate movie similarity scores and save the model files inside the `models/` directory.

### 5. Run the recommendation system

From the project root directory:

```bash
python -m src.content_based
```

## Future Improvements

This project is being developed in multiple phases:

- [x] Content-Based Recommendation System
- [ ] Streamlit Web Application
- [ ] Collaborative Filtering
- [ ] Neural Network-Based Recommendations
- [ ] Hybrid Recommendation System
- [ ] Model Evaluation and Performance Comparison
- [ ] Deployment

## Learning Objective

This project is also a hands-on learning journey focused on understanding recommendation systems, text preprocessing, feature extraction, similarity measures and practical machine learning implementation.
