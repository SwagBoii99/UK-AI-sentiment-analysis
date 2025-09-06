import pandas as pd
import praw
from ntscraper import Nitter
import time

# --- Reddit Data Collection ---
def get_reddit_data(client_id, client_secret, user_agent, keywords, subreddits, limit=100):
    reddit = praw.Reddit(client_id=client_id,
                         client_secret=client_secret,
                         user_agent=user_agent)
    
    posts_data = []
    print("Collecting Reddit data...")
    for sub in subreddits:
        subreddit = reddit.subreddit(sub)
        for keyword in keywords:
            for post in subreddit.search(keyword, limit=limit):
                posts_data.append({
                    'source': 'Reddit',
                    'text': f"{post.title}. {post.selftext}",
                    'date': pd.to_datetime(post.created_utc, unit='s'),
                    'source_link': post.permalink
                })
    print(f"Collected {len(posts_data)} posts from Reddit.")
    return pd.DataFrame(posts_data)

# --- X (Twitter) Data Collection ---
def get_twitter_data(keywords, limit=500):
    scraper = Nitter()
    tweets_data = []
    print("Collecting Twitter data...")
    
    search_terms = [f'{kw} lang:en' for kw in keywords]
    
    try:
        # The 'number' parameter specifies the total limit across all keywords
        tweets = scraper.get_tweets(search_terms, mode='term', number=limit)
        
        # Check if the 'tweets' key exists and is a list
        if 'tweets' in tweets and isinstance(tweets['tweets'], list):
            for tweet in tweets['tweets']:
                 tweets_data.append({
                    'source': 'X',
                    'text': tweet['text'],
                    'date': pd.to_datetime(tweet['date']),
                    'source_link': tweet['link']
                })
        else:
            print("Warning: Twitter scraping returned no tweet data.")
            
    except Exception as e:
        # This will now catch the error and print a friendly message
        print(f"\n--- WARNING ---")
        print(f"An error occurred while scraping Twitter: {e}")
        print("This is often a temporary issue. The program will continue using only the Reddit data.")
        print("-----------------\n")

    print(f"Collected {len(tweets_data)} tweets.")
    return pd.DataFrame(tweets_data)

# --- Main Execution ---
if __name__ == '__main__':
    # --- CONFIGURATION ---
    KEYWORDS = ["Artificial Intelligence", "AI jobs", "AI safety", "UK AI strategy", "DeepMind"]
    
    # Reddit Config
    REDDIT_CLIENT_ID = 'PabOwNxmN2pYUVzHLysyWw'
    REDDIT_CLIENT_SECRET = 'n0fig7hihSVQzPnSG-NLwD4SzI2pZw'
    REDDIT_USER_AGENT = 'uk_ai_sentiment_scraper by u/Potential_Meaning662'
    SUBREDDITS = ['unitedkingdom', 'ukpolitics', 'technology']
    
    # --- RUN COLLECTION ---
    reddit_df = get_reddit_data(REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET, REDDIT_USER_AGENT, KEYWORDS, SUBREDDITS)
    twitter_df = get_twitter_data(KEYWORDS)
    
    # Combine and save
    if not twitter_df.empty:
        final_df = pd.concat([reddit_df, twitter_df], ignore_index=True)
    else:
        final_df = reddit_df
    
    # Save the raw data
    final_df.to_csv('ai_sentiment_raw_data.csv', index=False)
    print("Data collection complete. Raw data saved to 'ai_sentiment_raw_data.csv'")