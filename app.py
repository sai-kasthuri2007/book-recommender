from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd
from scipy.sparse import load_npz

app = Flask(__name__)

# Load the trained ML model
model = pickle.load(open('artifacts/model.pkl', 'rb'))

# Load the sparse book-rating matrix
book_sparse = load_npz('artifacts/book_sparse.npz')

# Load book titles
book_titles = pickle.load(open('artifacts/book_titles.pkl', 'rb'))

# Load book information
book_info = pd.read_csv('artifacts/book_info.csv')


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/books')
def get_books():

    return jsonify(book_titles)


@app.route('/recommend', methods=['POST'])
def recommend():

    book_name = request.json['book_name']

    if book_name not in book_titles:
        return jsonify({
            'error': 'Book not found'
        }), 404

    # Find the row number of the selected book
    book_index = book_titles.index(book_name)

    # Find similar books
    distance, suggestion = model.kneighbors(
        book_sparse.getrow(book_index),
        n_neighbors=6
    )

    recommendations = []

    for i in suggestion[0][1:]:

        title = book_titles[i]

        matching_books = book_info[
            book_info['Book-Title'] == title
        ]

        if matching_books.empty:
            continue

        book = matching_books.iloc[0]

        recommendations.append({
            'title': title,
            'author': book['Book-Author'],
            'image_url': book['Image-URL-M']
        })

    return jsonify(recommendations)


if __name__ == '__main__':
    app.run(debug=True)