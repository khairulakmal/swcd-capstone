# 📊 TweetSense: Emotion Analytics Dashboard for Tweets

TweetSense is a Streamlit-based interactive dashboard that performs emotion classification on tweets and visualizes mood trends over time. The transformer models provides a nuanced analysis of public sentiment through calendar heatmaps, weekly emotion trends, spike detection, and keyword insights.

## 🚀 Features

- 🔍 Emotion classification using:
  - [`cardiffnlp/twitter-roberta-large-emotion-latest`](https://huggingface.co/cardiffnlp/twitter-roberta-large-emotion-latest)
  - [`j-hartmann/emotion-english-distilroberta-base`](https://huggingface.co/j-hartmann/emotion-english-distilroberta-base)
- 📅 Daily emotion calendar with color-coded mood scores
- 📈 Weekly mood trend line based on weighted polarity scores
- 🗺️ Emotion frequency heatmap (weekly)
- 🚨 Emotion spike detection based on sudden frequency changes
- 📝 Keyword extraction from spike periods using TF-IDF
- ⚠️ JSON validation and error handling for malformed inputs

## 📂 Input Format

Upload a `.json` file containing tweets in the following format:

[
  {
    "id": "123456789",
    "text": "I feel great about this new feature!",
    "date": "Wed Apr 10 10:35:21 +0000 2024"
  },
  ...
]

Each object **must** contain a `"text"` and `"date"` field.

## 🧪 Mood Score Calculation

Each predicted emotion is mapped to a fixed polarity value, then multiplied by the model’s confidence score. Example:

| Emotion     | Polarity |
|-------------|----------|
| joy         | +1.0     |
| love        | +0.8     |
| anger       | -1.0     |
| sadness     | -0.9     |
| ...         | ...      |

This produces a continuous **Mood Index**, which is averaged daily and weekly.

## 🖥️ How to Run

1. Clone this repo:
   git clone https://github.com/khairulakmal/swcd-capstone.git
   cd swcd-capstone

2. Install dependencies (Python 3.10+ recommended):
   pip install -r requirements.txt

3. Run the Streamlit app:
   streamlit run app_streamlit.py

4. Upload your tweet dataset and explore the dashboard

## 📊 Visualizations

- **Tab 1**: Daily Emotion Calendar  
- **Tab 2**: Weekly Mood Trend  
- **Tab 3**: Emotion Frequency Heatmap  
- **Tab 4**: Emotion Spikes Table  
- **Tab 5**: Spike Keyword Extractor  

## 📌 Limitations

- Requires manual `.json` uploads (no Twitter API streaming)
- English-only emotion classification

## 📚 Related Work

- Hedonometer: Real-time sentiment tracking using word-level happiness scores
  https://hedonometer.org/
- GoEmotions Dataset: Fine-grained emotion dataset used to train many models
  https://arxiv.org/abs/2010.12421

## 📄 License

This project is for academic use and research. Please respect Twitter's Terms of Service when collecting or analyzing tweet data.

---

Feel free to fork, extend, or contribute!
