import os
import pickle
import pandas as pd

from scipy.sparse import csr_matrix, save_npz
from sklearn.neighbors import NearestNeighbors

# 1. Create artifacts folder
os.makedirs('artifacts', exist_ok=True)

# 2. Load Books dataset

books = pd.read_csv(
    '/kaggle/input/datasets/ra4u12/bookrecommendation/BX-Books.csv',
    sep=';',
    encoding='latin-1',
    on_bad_lines='skip',
    engine='python'
)

# 3. Load Ratings dataset

ratings = pd.read_csv(
    '/kaggle/input/datasets/ra4u12/bookrecommendation/BX-Book-Ratings.csv',
    sep=';',
    encoding='latin-1',
    on_bad_lines='skip',
    engine='python'
)

# 4. Merge ratings with book information

ratings_with_books = ratings.merge(
    books,
    on='ISBN'
)

# 5. Convert book titles and users into numerical codes
book_codes = pd.Categorical(
    ratings_with_books['Book-Title']
)

user_codes = pd.Categorical(
    ratings_with_books['User-ID']
)

# 6. Create sparse book-rating matrix

book_sparse = csr_matrix(
    (
        ratings_with_books['Book-Rating'].values,
        (book_codes.codes, user_codes.codes)
    )
)

book_titles = list(book_codes.categories)


print("Sparse matrix shape:", book_sparse.shape)
print("Number of books:", len(book_titles))

# 7. Save sparse matrix

save_npz(
    'artifacts/book_sparse.npz',
    book_sparse
)

print("book_sparse.npz saved successfully!")

# 8. Save book titles

with open(
    'artifacts/book_titles.pkl',
    'wb'
) as file:
    pickle.dump(book_titles, file)

print("book_titles.pkl saved successfully!")


# 9. Create K-Nearest Neighbors model

model = NearestNeighbors(
    algorithm='brute',
    metric='cosine'
)

model.fit(book_sparse)


print("Model trained successfully!")
print("Model features:", model.n_features_in_)

# 10. Save trained model

with open(
    'artifacts/model.pkl',
    'wb'
) as file:
    pickle.dump(model, file)

print("model.pkl saved successfully!")


# 11. Create book information file

small_book_info = books[
    ['Book-Title', 'Book-Author']
].drop_duplicates('Book-Title')


# Keep only books that exist in the recommendation matrix
small_book_info = small_book_info[
    small_book_info['Book-Title'].isin(book_titles)
]

# 12. Save book information

small_book_info.to_csv(
    'artifacts/book_info.csv',
    index=False
)

print("book_info.csv saved successfully!")

# 13. Display generated files

print("\nGenerated artifacts:")

for file in os.listdir('artifacts'):
    size = os.path.getsize(
        os.path.join('artifacts', file)
    )

    print(
        file,
        "=",
        round(size / (1024 * 1024), 2),
        "MB"
    )


print("\nNumber of books in book_info.csv:",
      len(small_book_info))

print("\nML pipeline completed successfully!")
