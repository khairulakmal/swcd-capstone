import streamlit as st
import json
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
import calendar
from datetime import datetime
from collections import defaultdict, Counter
from transformers import pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
import torch

st.set_page_config(layout="wide")
st.title("📊 Tweet Emotion Analysis Dashboard")

# ========== CACHING FUNCTIONS ==========
@st.cache_data(show_spinner="🔍 Classifying tweets...")
def classify_tweets(tweets):
    pipe_cardiff = pipeline(
        "text-classification",
        model="cardiffnlp/twitter-roberta-large-emotion-latest",
        top_k=None,
        device=0 if torch.cuda.is_available() else -1
    )
    pipe_hartmann = pipeline(
        "text-classification",
        model="j-hartmann/emotion-english-distilroberta-base",
        top_k=None,
        device=0 if torch.cuda.is_available() else -1
    )

    classified = []
    for tweet in tweets:
        text = tweet["text"]

        pred_cardiff = pipe_cardiff(text)[0]
        if pred_cardiff:
            top_cardiff = max(pred_cardiff, key=lambda x: x["score"])
            tweet["emotion_cardiff"] = top_cardiff["label"]
            tweet["emotion_cardiff_conf"] = round(top_cardiff["score"], 4)
        else:
            tweet["emotion_cardiff"] = None
            tweet["emotion_cardiff_conf"] = None

        pred_hartmann = pipe_hartmann(text)[0]
        top_hartmann = sorted(pred_hartmann, key=lambda x: x["score"], reverse=True)[0]
        tweet["emotion_hartmann"] = top_hartmann["label"]
        tweet["emotion_hartmann_conf"] = round(top_hartmann["score"], 4)

        classified.append(tweet)

    return classified


@st.cache_data
def process_classified(classified):
    top1_records = []
    mood_scores = []
    emotion_buckets = defaultdict(list)

    for tweet in classified:
        emotion = tweet.get("emotion_cardiff")
        conf = tweet.get("emotion_cardiff_conf")
        date = tweet.get("date")

        if emotion and conf and date:
            dt = datetime.strptime(date, "%a %b %d %H:%M:%S +0000 %Y")
            top1_records.append({"date": dt, "emotion": emotion})
            emotion_buckets[dt.date()].append(emotion)

            score = {
                "anger": -1.0, "anticipation": 0.3, "disgust": -0.7, "fear": -0.8, "joy": 1.0,
                "love": 0.8, "optimism": 0.7, "pessimism": -0.6, "sadness": -0.9,
                "surprise": 0.5, "trust": 0.6
            }.get(emotion, 0.0) * conf
            mood_scores.append({"date": dt, "score": score})

    df_top1 = pd.DataFrame(top1_records)
    df_mood = pd.DataFrame(mood_scores)
    return df_top1, df_mood, emotion_buckets


# ========== FILE UPLOAD + CLASSIFICATION ==========
uploaded_file = st.file_uploader("Upload a JSON file with tweets", type=["json"])
if uploaded_file:
    try:
        tweets = json.load(uploaded_file)
        if not isinstance(tweets, list) or not all("text" in t and "date" in t for t in tweets):
            st.error("Uploaded file must be a JSON list of tweets with 'text' and 'date' keys.")
            st.stop()
        st.success(f"Loaded {len(tweets)} tweets")
    except json.JSONDecodeError:
        st.error("Invalid JSON file. Please upload a properly formatted JSON.")
        st.stop()

    classified = classify_tweets(tweets)
    df_top1, df_mood, emotion_buckets = process_classified(classified)
else:
    st.stop()


# ========== TAB UI ========== #
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🗓️ Daily Emotion Calendar",
    "📈 Weekly Mood Trend",
    "🗺️ Weekly Emotion Frequency Heatmap",
    "🚨 Emotion Spikes",
    "📝 Spike Keywords"
])


# === TAB 1: Calendar
with tab1:
    st.subheader("🗓️ Daily Emotion Calendar")
    st.markdown("*Label = most frequent emotion(s) from that day’s tweets. Color = average mood score based on all detected emotions.*")

    selected_month = st.selectbox("Select Month", sorted(df_top1["date"].dt.to_period("M").astype(str).unique()))
    year, month = map(int, selected_month.split("-"))
    cal = calendar.Calendar(firstweekday=0)
    all_days = list(cal.itermonthdates(year, month))
    days_in_month = [d for d in all_days if d.month == month]

    df_mood["date_only"] = df_mood["date"].dt.date
    mood_dict = df_mood.groupby("date_only")["score"].mean().to_dict()
    
    def get_top_emotions(series):
        counts = series.value_counts()
        top_count = counts.iloc[0]
        top_emotions = counts[counts == top_count].index.tolist()
        return ', '.join(top_emotions)

    top_emo = df_top1.groupby(df_top1["date"].dt.date)["emotion"].agg(get_top_emotions).to_dict()

    col1, col2 = st.columns([1.4, 0.6])
    with col1:
        fig, ax = plt.subplots(figsize=(5, 3))
        cmap = plt.get_cmap("RdYlGn")
        norm = lambda x: (x + 1) / 2

        date_to_pos = {}
        row = 0
        col = 0
        for d in cal.itermonthdates(year, month):
            if d.month == month:
                date_to_pos[d] = (row, col)
            col += 1
            if col == 7:
                col = 0
                row += 1

        for d, (r, c) in date_to_pos.items():
            mood_score = mood_dict.get(d, None)
            color = cmap(norm(mood_score)) if mood_score is not None else "#f0f0f0"
            rect = patches.Rectangle((c, r), 1, 1, linewidth=1, edgecolor='black', facecolor=color)
            ax.add_patch(rect)
            ax.text(c + 0.1, r + 0.15, f"{d.day}", ha='left', va='top', fontsize=5, weight='bold')
            if d in top_emo:
                emotions = top_emo[d].split(', ')
                line_spacing = 0.18
                font_size = 4

                total_height = line_spacing * (len(emotions) - 1)
                start_y = r + 0.5 - total_height / 2

                for idx, emo in enumerate(emotions):
                    ax.text(
                        c + 0.5,
                        start_y + idx * line_spacing,
                        emo,
                        ha='center',
                        va='center',
                        fontsize=font_size
                    )

        for i, day in enumerate(['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']):
            ax.text(i + 0.5, -0.5, day, ha='center', va='center', fontsize=5, weight='bold')

        ax.set_xlim(0, 7)
        ax.set_ylim(row, -1)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"{calendar.month_name[month]} {year}", fontsize=8)
        ax.set_aspect('equal')
        st.pyplot(fig)

    with col2:
        st.markdown("### 📋 Emotion Details")

        click_day = st.date_input(
            "Pick a date",
            value=datetime(year, month, 1).date(),
            min_value=datetime(year, month, 1).date(),
            max_value=datetime(year, month, calendar.monthrange(year, month)[1]).date()
        )

        # Get all tweets from the selected date
        tweets_on_day = [
            tweet for tweet in classified
            if datetime.strptime(tweet["date"], "%a %b %d %H:%M:%S +0000 %Y").date() == click_day
        ]

        if tweets_on_day:
            # Count top-1 emotions from Cardiff
            emotion_list = [tweet.get("emotion_cardiff") for tweet in tweets_on_day if tweet.get("emotion_cardiff")]
            freq = Counter(emotion_list)

            # Show summary
            st.write(f"**{click_day}**: {len(tweets_on_day)} tweets")
            st.markdown("**Emotion Count in Tweet:**")
            st.table(pd.DataFrame(freq.items(), columns=["Emotion", "Count"]))

            # Show tweets
            with st.expander(f"📜 View {len(tweets_on_day)} tweets from {click_day}"):
                for i, tweet in enumerate(tweets_on_day, 1):
                    emotion = tweet.get("emotion_cardiff") or "No emotion"
                    conf = tweet.get("emotion_cardiff_conf")
                    st.markdown(f"""
                    **{i}.** {tweet['text']}  
                    _Emotion_: `{emotion}` ({conf})
                    """)
        else:
            st.info("No tweets recorded on this day.")


# === TAB 2: Weekly Mood
with tab2:
    st.subheader("📈 Weekly Mood Trend")
    st.markdown("_Mood Score is calculated from weighted emotion values, averaged across all tweets per week. Joy and love are positive; sadness and fear are negative._")
    df_mood['week_start'] = df_mood['date'] - pd.to_timedelta(df_mood['date'].dt.weekday, unit='D')
    df_mood['week_range'] = df_mood['week_start'].dt.strftime('%b %d') + ' – ' + (df_mood['week_start'] + pd.Timedelta(days=6)).dt.strftime('%b %d')
    weekly_mood = df_mood.groupby("week_range")["score"].mean().reset_index()
    fig_mood, ax_mood = plt.subplots(figsize=(10, 5))
    sns.lineplot(data=weekly_mood, x="week_range", y="score", marker="o", ax=ax_mood)
    ax_mood.set_xticklabels(ax_mood.get_xticklabels(), rotation=45)
    ax_mood.set_title("Average Mood Index per Week")
    ax_mood.set_ylabel("Mood Score")
    ax_mood.set_xlabel("Week")
    ax_mood.grid(True)
    st.pyplot(fig_mood)


# === TAB 3: Weekly Emotion Heatmap
with tab3:
    st.subheader("🗺️ Weekly Emotion Frequency Heatmap")
    df_top1["week_start"] = df_top1["date"] - pd.to_timedelta(df_top1["date"].dt.weekday, unit='D')
    df_top1["week_range"] = df_top1["week_start"].dt.strftime('%b %d') + ' – ' + (df_top1["week_start"] + pd.Timedelta(days=6)).dt.strftime('%b %d')
    weekly_counts = df_top1.groupby(["week_range", "emotion"]).size().unstack(fill_value=0)
    fig_heat, ax_heat = plt.subplots(figsize=(12, 6))

    # Custom color palette (same order each time)
    emotion_palette = sns.color_palette("coolwarm", n_colors=len(weekly_counts.columns))
    sns.heatmap(weekly_counts.T, cmap="coolwarm", linewidths=0.5, ax=ax_heat, cbar_kws={"label": "Tweet Count"})
    ax_heat.set_title("Emotion Frequency per Week")
    ax_heat.set_xlabel("Week")
    ax_heat.set_xticklabels(ax_heat.get_xticklabels(), rotation=45, ha='right')
    ax_heat.set_ylabel("Emotion")
    st.pyplot(fig_heat)


# === TAB 4: Emotion Spikes
with tab4:
    st.subheader("🚨 Detected Emotion Spikes")
    spike_threshold = 5
    spikes = []
    for emotion in weekly_counts.columns:
        series = weekly_counts[emotion]
        prev = None
        for week, val in series.items():
            if prev is not None and abs(val - prev) >= spike_threshold:
                spikes.append({
                    "week": week,
                    "emotion": emotion,
                    "count": val,
                    "change": val - prev
                })
            prev = val
    spike_df = pd.DataFrame(spikes)
    st.dataframe(spike_df)


# === TAB 5: Spike Keywords
with tab5:
    st.subheader("📝 Keywords from Spike Weeks")

    spike_texts = defaultdict(list)

    for _, row in spike_df.iterrows():
        week = row["week"]
        emotion = row["emotion"]

        for tweet in classified:
            try:
                dt = datetime.strptime(tweet["date"], "%a %b %d %H:%M:%S +0000 %Y")
                tweet_week_start = dt - pd.to_timedelta(dt.weekday(), unit='d')
                tweet_week_range = tweet_week_start.strftime('%b %d') + ' – ' + (tweet_week_start + pd.Timedelta(days=6)).strftime('%b %d')
            except Exception:
                continue

            if tweet.get("emotion_cardiff") == emotion and tweet_week_range == week:
                spike_texts[f"{week}_{emotion}"].append(tweet["text"])

    def extract_keywords(docs, top_n=10):
        vectorizer = TfidfVectorizer(stop_words="english", max_features=1000)
        X = vectorizer.fit_transform(docs)
        indices = X.sum(axis=0).A1.argsort()[::-1][:top_n]
        return [vectorizer.get_feature_names_out()[i] for i in indices]

    if not spike_texts:
        st.info("No significant spike keywords were extracted.")
    else:
        for key, texts in spike_texts.items():
            if len(texts) >= 5:
                st.markdown(f"**Top keywords for {key}**")
                keywords = extract_keywords(texts)
                st.write(", ".join(keywords))
