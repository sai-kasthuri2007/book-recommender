# book-recommender
Book Recommendation System using Machine Learning and Flask
# Book Recommender System

A machine learning based Book Recommender System developed using Python, Flask, Pandas, SciPy and Scikit-learn.

## Features

- Search for books using an autocomplete search box
- Recommend similar books
- Display book title and author
- Machine learning based recommendations
- Flask web application

## Technologies Used

- Python
- Pandas
- SciPy
- Scikit-learn
- Flask
- HTML
- CSS
- JavaScript

## Machine Learning

The recommendation system uses the Nearest Neighbors algorithm with book-rating data from the Book-Crossing dataset.

The book ratings are represented using a sparse matrix to reduce storage requirements.

## Dataset

Book-Crossing Dataset from Kaggle.
## How to Run

Install the required libraries:

pip install flask pandas scipy scikit-learn

Run the Flask application:

python app.py

Open the application in a browser:

http://127.0.0.1:5000

## Project Structure

```text
book-recommender/
│
├── app.py
│
├── artifacts/
│   ├── model.pkl
│   ├── book_sparse.npz
│   ├── book_titles.pkl
│   └── book_info.csv
│
└── templates/
    └── index.html

