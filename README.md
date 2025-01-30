## Book-Recommendation-Engine-using-KNN
# Code for Book recommendation using machine learning through python 



import pandas as pd
from sklearn.neighbors import NearestNeighbors
import numpy as np

# Load dataset
books = pd.read_csv("book_ratings.csv")  # Ensure you have a dataset with user-book ratings

# Pivot table: Users as rows, books as columns
book_matrix = books.pivot_table(index="user_id", columns="book_title", values="rating").fillna(0)

# Fit KNN model
model = NearestNeighbors(metric="cosine", algorithm="brute")
model.fit(book_matrix.T)

def get_recommends(book_title):
    book_idx = list(book_matrix.columns).index(book_title)
    distances, indices = model.kneighbors([book_matrix.T.iloc[book_idx]], n_neighbors=6)

  recommendations = [[book_matrix.columns[i], 1 - distances.flatten()[j]] for j, i in enumerate(indices.flatten())][1:]
    
  return [book_title, recommendations]

# Example usage
print(get_recommends("The Queen of the Damned (Vampire Chronicles (Paperback))"))
