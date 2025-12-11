# Text Sentiment Analyzer (Naive Bayes)

This project is an end-to-end Machine Learning solution for classifying movie review text into two categories: **Positive** or **Negative**. It uses a trained Multinomial Naive Bayes model and is packaged with a runnable Gradio web interface for instant demonstration.

This project is a perfect example of a fast, client-ready ML deployment artifact.

---

## 🎯 Project Goal & Results

The primary goal was to build a quick and accurate sentiment classifier using classical ML techniques.

* **Model Used:** Multinomial Naive Bayes (MultinomialNB)
* **Vectorizer:** CountVectorizer (Bag-of-Words) with a maximum of 5,000 features.
* **Dataset:** IMDB Dataset of 50,000 Movie Reviews.
* **Final Accuracy:** 84.71%

## ⚙️ How to Run the Demo Locally

This project is designed for immediate execution, demonstrating a professional ML delivery pipeline.

### Prerequisites

1.  Python (3.7+)
2.  Install required libraries using the provided `requirements.txt` file:
    ```bash
    pip install -r requirements.txt
    ```

### Execution Steps

1.  **Download all files** from this repository (the `.ipynb`, `.py`, `.txt`, and the two `.pkl` files).
2.  Open your terminal or command prompt in the project directory.
3.  Run the application script:
    ```bash
    python app.py
    ```

A link will appear in your terminal (usually `http://127.0.0.1:7860`). Open that link in your browser to interact with the model instantly!

---

## 📁 Repository Contents

| File | Description | Purpose |
| :--- | :--- | :--- |
| **`Text_Sentiment_Analyzer.ipynb`** | The full Jupyter Notebook. | Documents the complete training, preprocessing, and evaluation process. |
| **`app.py`** | The main execution script. | Loads the model artifacts and launches the Gradio UI. **The client-facing runnable file.** |
| **`multinomial_nb_model.pkl`** | The saved, trained Naive Bayes model. | Enables instant predictions without time-consuming re-training. |
| **`count_vectorizer.pkl`** | The saved, fitted vocabulary (Vectorizer). | **CRITICAL** for converting new text input into the numerical format the model expects. |
| **`requirements.txt`** | List of all necessary Python dependencies. | Ensures easy, reproducible setup for any user. |
| **`README.md`** | This project overview. | Professional documentation and instructions. |

## 🔗 Dataset Details

The dataset used for training this model (50,000 IMDB movie reviews) is publicly available:
* **Source Link:** [IMDB Dataset of 50k Movie Reviews on Kaggle](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)

---