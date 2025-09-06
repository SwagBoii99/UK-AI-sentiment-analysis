import pandas as pd
from transformers import pipeline
from bertopic import BERTopic
import torch

# Check for GPU, fall back to CPU if not available
device = 0 if torch.cuda.is_available() else -1
print(f"Using device: {'GPU' if device == 0 else 'CPU'}")

def perform_analysis(filepath):
    df = pd.read_csv(filepath)
    docs = df['cleaned_text'].astype(str).tolist()

    # --- 1. Topic Modeling (Aspect Extraction) ---
    print("Starting topic modeling... This may take a while.")
    topic_model = BERTopic(verbose=True, min_topic_size=15)
    topics, _ = topic_model.fit_transform(docs)
    df['topic'] = topics
    
    topic_info = topic_model.get_topic_info()
    topic_map = topic_info.set_index('Topic')['Name'].to_dict()
    df['topic_name'] = df['topic'].map(topic_map)
    print("Topic modeling complete.")

    # --- 2. Sentiment Analysis ---
    print("Starting sentiment analysis...")
    sentiment_pipeline = pipeline(
        "sentiment-analysis", 
        model="cardiffnlp/twitter-roberta-base-sentiment-latest",
        device=device
    )
    
    sentiments = []
    # We pass truncation arguments directly to the pipeline
    # It's more efficient and handles everything internally
    # The try-except block will skip any text that still causes a rare error
    for doc in docs:
        try:
            result = sentiment_pipeline(doc, truncation=True, max_length=512)
            sentiments.append(result[0])
        except Exception as e:
            print(f"Skipping a post due to sentiment analysis error: {e}")
            # Append a placeholder if there's an error
            sentiments.append({'label': 'error', 'score': 0.0})

    df['sentiment'] = [s['label'] for s in sentiments]
    df['sentiment_score'] = [s['score'] for s in sentiments]
    print("Sentiment analysis complete.")

    # --- 3. Emotion Analysis ---
    print("Starting emotion analysis...")
    emotion_pipeline = pipeline(
        "text-classification",
        model="SamLowe/roberta-base-go_emotions",
        device=device
    )

    emotions = []
    for doc in docs:
        try:
            result = emotion_pipeline(doc, truncation=True, max_length=512)
            emotions.append(result[0])
        except Exception as e:
            print(f"Skipping a post due to emotion analysis error: {e}")
            emotions.append({'label': 'error', 'score': 0.0})

    df['emotion'] = [e['label'] for e in emotions]
    df['emotion_score'] = [e['score'] for e in emotions]
    print("Emotion analysis complete.")

    # Save final analyzed data
    df.to_csv('ai_sentiment_analyzed_data.csv', index=False)
    print("Full analysis complete. Results saved to 'ai_sentiment_analyzed_data.csv'")

if __name__ == '__main__':
    perform_analysis('ai_sentiment_cleaned_data.csv')