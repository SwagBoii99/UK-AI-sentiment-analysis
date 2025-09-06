import pandas as pd
import re

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\@\w+|\#','', text)
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

if __name__ == '__main__':
    df = pd.read_csv('ai_sentiment_raw_data.csv')
    
    df.dropna(subset=['text'], inplace=True)
    
    df['cleaned_text'] = df['text'].apply(clean_text)
    
    df = df[df['cleaned_text'].str.len() > 20]
    
    df.to_csv('ai_sentiment_cleaned_data.csv', index=False)
    print("Data cleaning complete. Clean data saved to 'ai_sentiment_cleaned_data.csv'")