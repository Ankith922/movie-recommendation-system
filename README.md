# Movie Recommendation System

## Project Description
A simple, web-based Movie Recommendation System built with Python. This application allows users to select a movie and receive 5 similar movie recommendations based on content similarity (genres, keywords, overview, cast, and director). It provides an easy-to-use interface built with Streamlit.

## Features
- Content-based filtering using Natural Language Processing (NLP).
- Simple and intuitive web interface built with Streamlit.
- Provides similarity scores for the recommended movies.
- Handles missing dataset values gracefully without crashing.
- Beginner-friendly code structure with a **pure Python implementation** of TF-IDF and Cosine Similarity to avoid strict Windows Security Policies blocking C-extensions.

## Technologies Used
- **Python**: Core programming language.
- **Pandas**: Data manipulation and handling.
- **Streamlit**: Framework for building the interactive web application.
- *(Note: Scikit-learn was intentionally excluded to ensure maximum compatibility across restricted systems.)*

## How the Recommendation System Works
1. **Data Preprocessing**: The system reads movie data (`movies.csv`) and extracts important features such as genres, keywords, overview, cast, and director. It also handles any missing values by replacing them with empty strings.
2. **Feature Combination**: Relevant text features are combined into a single text string for each movie.
3. **TF-IDF Vectorization**: The combined text is converted into numerical vectors using a custom pure-Python mathematical implementation. This process gives more weight to unique words and less weight to common words.
4. **Cosine Similarity**: The system calculates the cosine of the angle between the vectors representing the movies. A higher similarity score (closer to 1.0) means the movies are more similar in content.
5. **Recommendation**: When a user selects a movie, the system fetches the top 5 movies with the highest similarity scores compared to the selected movie.

## Installation Steps

1. Clone this repository or download the source code.
2. Ensure you have Python installed.
3. Create a virtual environment (optional but highly recommended):
   ```bash
   python -m venv venv
   ```
4. Activate the virtual environment:
   - On **Windows**:
     ```bash
     venv\Scripts\activate
     ```
   - On **macOS/Linux**:
     ```bash
     source venv/bin/activate
     ```
5. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## How to Run the Project

Run the following command in your terminal within the project directory:

```bash
streamlit run app.py
```

The application will start and open automatically in your default web browser (typically at `http://localhost:8501`).

## Example Usage
1. Open the application in your browser.
2. Select a movie from the dropdown menu (e.g., "The Matrix").
3. Click the "Recommend Movies" button.
4. View the top 5 similar movies and their similarity scores displayed below.

## Future Improvements
- Integrate with the TMDB API to fetch live movie data and display movie posters.
- Add user-based collaborative filtering for more personalized recommendations.
- Improve the UI with better layouts, images, and additional movie information (e.g., ratings, release year).
