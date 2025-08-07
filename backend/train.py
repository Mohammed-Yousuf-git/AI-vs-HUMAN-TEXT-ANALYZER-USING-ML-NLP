import pandas as pd
import numpy as np
import re
import pickle
import os
import textstat
import language_tool_python
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, accuracy_score
from scipy.sparse import hstack, csr_matrix

# === Ensure model directory exists ===
os.makedirs("backend/model", exist_ok=True)

# === Load dataset ===
df = pd.read_csv("/Users/usufahmed/Desktop/AIvsHUMAN/ai_human_content_detection_dataset 2.csv")
df["text_content"] = df["text_content"].astype(str)
y = df["label"]

# === Grammar tool setup ===
tool = language_tool_python.LanguageTool('en-US')

common_words = set([
    "the", "and", "is", "in", "it", "you", "that", "he", "was", "for",
    "on", "are", "with", "as", "I", "his", "they", "be", "at", "one", "have", "this"
])

def extract_engineered_features(text):
    sentences = re.split(r'[.!?]+', text)
    words = re.findall(r'\b\w+\b', text)
    num_words = len(words)
    num_sentences = len(sentences) if sentences[0] != '' else 0
    characters = len(text)
    lexical_diversity = len(set(words)) / num_words if num_words != 0 else 0
    avg_sentence_length = num_words / num_sentences if num_sentences != 0 else 0
    avg_word_length = sum(len(word) for word in words) / num_words if num_words != 0 else 0
    punctuation_ratio = len(re.findall(r'[^\w\s]', text)) / characters if characters != 0 else 0
    flesch_reading_ease = textstat.flesch_reading_ease(text)
    gunning_fog_index = textstat.gunning_fog(text)
    grammar_errors = len(tool.check(text))
    passive_voice_ratio = len(re.findall(r'\b(is|was|were|been|being|are|am)\b\s+\w+ed', text)) / num_sentences if num_sentences != 0 else 0
    predictability_score = sum([1 for word in words if word.lower() in common_words]) / num_words if num_words != 0 else 0
    burstiness = lexical_diversity * avg_sentence_length
    sentiment_score = textstat.flesch_kincaid_grade(text)

    return [
        num_words, characters, num_sentences, lexical_diversity,
        avg_sentence_length, avg_word_length, punctuation_ratio,
        flesch_reading_ease, gunning_fog_index, grammar_errors,
        passive_voice_ratio, predictability_score, burstiness,
        sentiment_score
    ]

# === Extract features ===
engineered_features = df["text_content"].apply(extract_engineered_features)
X_engineered = pd.DataFrame(engineered_features.tolist())
engineered_dim = X_engineered.shape[1]  # Save this!

# === TF-IDF ===
vectorizer = TfidfVectorizer(max_features=1000, stop_words="english")
X_tfidf = vectorizer.fit_transform(df["text_content"])

# === Combine ===
X_combined = hstack([csr_matrix(X_engineered.values), X_tfidf])

# === Train/test split ===
X_train, X_test, y_train, y_test = train_test_split(X_combined, y, test_size=0.2, random_state=42)

# === Train ===
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# === Evaluate ===
y_pred = model.predict(X_test)
print("✅ Accuracy:", accuracy_score(y_test, y_pred))
print("📊 Classification Report:\n", classification_report(y_test, y_pred))

# ✅ Save model + vectorizer + feature dim in one file
bundle = {
    "model": model,
    "vectorizer": vectorizer,
    "engineered_dim": engineered_dim
}

with open("backend/model/ai_model_bundle.pkl", "wb") as f:
    pickle.dump(bundle, f)

print("✅ Model bundle saved to backend/model/ai_model_bundle.pkl")
