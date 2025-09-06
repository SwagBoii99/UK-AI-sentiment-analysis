import streamlit as st
import pandas as pd
import plotly.express as px

# --- Page Configuration ---
st.set_page_config(
    page_title="UK AI Sentiment Dashboard",
    page_icon="🤖",
    layout="wide"
)

# --- Load Data ---
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('ai_sentiment_analyzed_data.csv')
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        # Clean up topic names for better display
        df['topic_name'] = df['topic_name'].apply(lambda x: ' '.join(str(x).split('_')[1:]))
        return df
    except FileNotFoundError:
        return None

df = load_data()

# --- Main Title ---
st.title("🤖 Mind the Gap: UK Public Sentiment on AI")
st.markdown("An interactive dashboard analyzing the UK's conversation around Artificial Intelligence.")

if df is None:
    st.error("Error: The analyzed data file ('ai_sentiment_analyzed_data.csv') was not found. Please run the analysis script first.")
else:
    # Since we only have one data source, we no longer need the filter.
    # We will use the entire dataframe for the charts.
    df_filtered = df

    # --- Key Metrics ---
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Posts Analyzed", f"{len(df_filtered):,}")
    positive_sentiment = len(df_filtered[df_filtered['sentiment'] == 'positive']) / len(df_filtered) if len(df_filtered) > 0 else 0
    col2.metric("Overall Positive Sentiment", f"{positive_sentiment:.1%}")
    dominant_emotion = df_filtered['emotion'].mode()[0] if not df_filtered.empty else "N/A"
    col3.metric("Dominant Emotion", dominant_emotion.capitalize())
    st.markdown("---")


    # --- Visualizations ---
    col1, col2 = st.columns((1, 1))

    with col1:
        st.subheader("Sentiment by Topic (Aspect)")
        sentiment_by_topic = df_filtered.groupby('topic_name')['sentiment'].value_counts(normalize=True).unstack().fillna(0)
        fig_sentiment = px.bar(sentiment_by_topic,
                               barmode='stack',
                               labels={'value': 'Percentage', 'topic_name': 'Topic'},
                               title="How positive/negative is the conversation on each topic?")
        st.plotly_chart(fig_sentiment, use_container_width=True)

    with col2:
        st.subheader("Emotion Distribution by Topic")
        top_emotions = df_filtered['emotion'].value_counts().nlargest(7).index
        emotion_by_topic = df_filtered[df_filtered['emotion'].isin(top_emotions)]
        emotion_by_topic_grouped = emotion_by_topic.groupby('topic_name')['emotion'].value_counts(normalize=True).unstack().fillna(0)
        
        fig_emotion = px.bar(emotion_by_topic_grouped,
                             barmode='stack',
                             labels={'value': 'Percentage', 'topic_name': 'Topic'},
                             title="What are the dominant emotions for each topic?")
        st.plotly_chart(fig_emotion, use_container_width=True)

    st.subheader("Sentiment Over Time")
    df_filtered.dropna(subset=['date'], inplace=True)
    df_filtered['month'] = df_filtered['date'].dt.to_period('M').astype(str)
    sentiment_over_time = df_filtered.groupby(['month', 'sentiment']).size().reset_index(name='count')
    fig_time = px.line(sentiment_over_time,
                       x='month',
                       y='count',
                       color='sentiment',
                       labels={'month': 'Month', 'count': 'Number of Posts'},
                       title="How has sentiment evolved over time?")
    st.plotly_chart(fig_time, use_container_width=True)

    # --- Display Raw Data ---
    st.subheader("Explore the Data")
    st.dataframe(df_filtered[['date', 'source', 'text', 'topic_name', 'sentiment', 'emotion']].head(100))