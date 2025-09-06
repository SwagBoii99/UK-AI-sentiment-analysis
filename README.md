# 🤖 UK AI Sentiment Analysis Dashboard

This repository contains the code for "Mind the Gap," an interactive dashboard that analyzes public sentiment towards Artificial Intelligence in the United Kingdom. The project collects real-time data from Reddit, performs advanced NLP analysis, and presents the findings through a web-based dashboard built with Streamlit.

## 📊 Key Findings

The analysis of nearly 1,000 Reddit posts revealed several key insights:
* **Overwhelmingly Cautious Sentiment:** The public conversation around AI is largely driven by neutral (informative/questioning) and negative emotions, with an overall positive sentiment of only **5.4%**.
* **Explosion in Conversation Volume:** The "Sentiment Over Time" chart clearly shows a massive spike in discussion volume from 2023 onwards, perfectly aligning with the mainstream adoption of generative AI tools like ChatGPT.
* **Topic-Specific Emotion:** Sentiment varies dramatically by topic. While discussions around "AI in the NHS" show pockets of optimism, conversations about "AI & Jobs" are dominated by fear and concern.

## 🛠️ Technical Stack

* **Data Collection:** Python (`praw`, `ntscraper`)
* **Data Analysis:** Pandas, Hugging Face Transformers
* **NLP Models:**
    * **Topic Modeling:** `BERTopic`
    * **Sentiment Analysis:** `cardiffnlp/twitter-roberta-base-sentiment-latest`
    * **Emotion Classification:** `SamLowe/roberta-base-go_emotions`
* **Dashboard:** Streamlit
* **Deployment:** Streamlit Community Cloud

## 🚀 Live Demo

(https://uk-ai-sentiment-analysis.streamlit.app/)


