
import joblib
import gradio as gr
import re
import os 


def clean_text(text):

    text = re.sub(r'<[^>]+>', ' ', text)

    text = text.lower()
 
    text = re.sub(r'[^a-z\\s]', ' ', text)

    text = re.sub(r'\\s+', ' ', text).strip()
    return text


def gradio_predict_sentiment(text):
    
    def get_sentiment_label(prediction):
        return 'Positive 🎉' if prediction == 1 else 'Negative 😠'
    
    cleaned = clean_text(text)
    
    vec = loaded_vectorizer.transform([cleaned])
    
    pred_label_encoded = loaded_model.predict(vec)[0]
   
    return get_sentiment_label(pred_label_encoded)


try:
   
    loaded_model = joblib.load('multinomial_nb_model.pkl')
    loaded_vectorizer = joblib.load('count_vectorizer.pkl')
    print("Successfully loaded model and vectorizer artifacts.")
    


    iface = gr.Interface(
        fn=gradio_predict_sentiment, 
        inputs=gr.Textbox(lines=5, label="EnterText Here"), 
        outputs="text",
        title="Text Sentiment Analyzer (Naive Bayes)",
        description="Built on Movies Review Data set ... Type in a comment or paragraph and get an instant sentiment prediction (Positive or Negative).",
        examples=[
            ["The special effects were terrible, and the plot made no sense."],
            ["This movie was so heartwarming and perfectly acted, a true masterpiece!"],
        ]
    )

    iface.launch()

except FileNotFoundError:
    print("\n--- ERROR: Files Not Found ---")
    print("Please ensure 'multinomial_nb_model.pkl' and 'count_vectorizer.pkl' are in the same directory as app.py.")
    print("Check your local folder and try again.")