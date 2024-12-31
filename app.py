from flask import Flask, render_template, request
import pickle
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

app = Flask(__name__)

# Load preprocessed models and data
with open("recipemodel_tfidf.pkl", "rb") as tfidf_file:
    tfidf_vectorizer = pickle.load(tfidf_file)
with open("recipemodel_scaler.pkl", "rb") as scaler_file:
    scaler = pickle.load(scaler_file)
with open("recipemodel_combined_features.pkl", "rb") as combined_file:
    combined_features = pickle.load(combined_file)
recipes = pd.read_csv("recipes.csv")

# Function for weighted recommendations
def recommend_weighted(input_ingredients, input_minutes, top_n=6):
    # Calculate similarity for ingredients
    input_ingredient_vector = tfidf_vectorizer.transform([input_ingredients])
    ingredient_sim = cosine_similarity(input_ingredient_vector, combined_features[:, 1:])

    # Calculate similarity for minutes
    scaled_minutes = scaler.transform([[input_minutes]])
    minute_sim = cosine_similarity(scaled_minutes[:, 0].reshape(-1, 1), combined_features[:, 0].reshape(-1, 1))

    # Weighted similarity
    combined_sim = (0.8 * ingredient_sim) + (0.2 * minute_sim)

    # Get top recommendations
    top_indices = combined_sim[0].argsort()[-top_n:][::-1]
    recommendations = recipes.iloc[top_indices].copy()
    recommendations['similarity_score'] = combined_sim[0][top_indices]

    # Clean ingredients to remove brackets and format as a string
    recommendations['ingredients'] = recommendations['ingredients'].apply(lambda x: ', '.join(eval(x)) if isinstance(x, str) and x.startswith('[') else x)

    # Clean steps to remove brackets and format as a string
    recommendations['steps'] = recommendations['steps'].apply(lambda x: '. '.join(eval(x)) + '.' if isinstance(x, str) and x.startswith('[') else x)

    return recommendations[['name', 'ingredients', 'minutes', 'similarity_score', 'description', 'steps']]


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Get user inputs from the form
        input_minutes = float(request.form['minutes'])
        input_ingredients = request.form['ingredients']

        # Generate recommendations
        recommendations = recommend_weighted(input_ingredients, input_minutes)

        return render_template(
            'index.html',
            recommendations=recommendations.to_dict(orient='records')
        )

    return render_template('index.html', recommendations=[])

if __name__ == '__main__':
    app.run(debug=True)
