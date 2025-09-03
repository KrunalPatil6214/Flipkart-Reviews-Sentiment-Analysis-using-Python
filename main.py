# 1. Importing Necessary Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from wordcloud import WordCloud
import re
import string
from collections import Counter

# Machine Learning Libraries
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.metrics import precision_score, recall_score

# NLTK for text preprocessing
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

# Suppress warnings
warnings.filterwarnings('ignore')

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('punkt_tab')

# Set style for plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

print("="*80)
print("FLIPKART REVIEWS SENTIMENT ANALYSIS PROJECT")
print("="*80)

# 2. Loading and Exploring the Dataset
print("\n1. LOADING AND EXPLORING THE DATASET")
print("-" * 50)

# Load the dataset
df = pd.read_csv('flipkart_data.csv')

print(f"Dataset Shape: {df.shape}")
print(f"Columns: {list(df.columns)}")
print("\nFirst 5 rows:")
print(df.head())

print("\nDataset Info:")
print(df.info())

print("\nMissing Values:")
print(df.isnull().sum())

print("\nRating Distribution:")
print(df['rating'].value_counts().sort_index())

# 3. Data Preprocessing
print("\n\n2. DATA PREPROCESSING")
print("-" * 50)

# Remove duplicates
print(f"Duplicate rows before removal: {df.duplicated().sum()}")
df = df.drop_duplicates()
print(f"Duplicate rows after removal: {df.duplicated().sum()}")

# Remove rows with missing values
df = df.dropna()
print(f"Final dataset shape: {df.shape}")

# Create sentiment labels based on ratings
# Ratings 1-3: Negative (0), Ratings 4-5: Positive (1)
def create_sentiment_labels(rating):
    if rating <= 3:
        return 0  # Negative
    else:
        return 1  # Positive

df['sentiment'] = df['rating'].apply(create_sentiment_labels)

print("\nSentiment Distribution:")
sentiment_counts = df['sentiment'].value_counts()
print(f"Negative (0): {sentiment_counts[0]}")
print(f"Positive (1): {sentiment_counts[1]}")

# Text preprocessing function
def preprocess_text(text):
    """
    Comprehensive text preprocessing function
    """
    # Convert to lowercase
    text = text.lower()
    
    # Remove 'READ MORE' from reviews
    text = re.sub(r'read more', '', text)
    
    # Remove URLs, mentions, and hashtags
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'@\w+|#\w+', '', text)
    
    # Remove punctuation and special characters
    text = re.sub(r'[^\w\s]', '', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Remove numbers
    text = re.sub(r'\d+', '', text)
    
    return text

# Initialize stemmer and stop words
stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

def advanced_preprocess_text(text):
    """
    Advanced preprocessing with tokenization, stopword removal, and stemming
    """
    # Basic preprocessing
    text = preprocess_text(text)
    
    # Tokenization
    tokens = word_tokenize(text)
    
    # Remove stopwords and perform stemming
    processed_tokens = []
    for token in tokens:
        if token not in stop_words and len(token) > 2:  # Remove short words
            stemmed_token = stemmer.stem(token)
            processed_tokens.append(stemmed_token)
    
    return ' '.join(processed_tokens)

# Apply preprocessing
print("\nApplying text preprocessing...")
df['cleaned_review'] = df['review'].apply(advanced_preprocess_text)

# Remove empty reviews after preprocessing
df = df[df['cleaned_review'].str.len() > 0]
print(f"Dataset shape after preprocessing: {df.shape}")

# 4. Exploratory Data Analysis (EDA)
print("\n\n3. EXPLORATORY DATA ANALYSIS")
print("-" * 50)

# Create visualizations
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# Sentiment distribution
sentiment_counts = df['sentiment'].value_counts()
axes[0, 0].pie(sentiment_counts.values, labels=['Positive', 'Negative'], autopct='%1.1f%%', 
               colors=['lightgreen', 'lightcoral'])
axes[0, 0].set_title('Sentiment Distribution')

# Rating distribution
df['rating'].value_counts().sort_index().plot(kind='bar', ax=axes[0, 1], color='skyblue')
axes[0, 1].set_title('Rating Distribution')
axes[0, 1].set_xlabel('Rating')
axes[0, 1].set_ylabel('Count')

# Review length analysis
df['review_length'] = df['cleaned_review'].str.len()
df.boxplot(column='review_length', by='sentiment', ax=axes[1, 0])
axes[1, 0].set_title('Review Length by Sentiment')
axes[1, 0].set_xlabel('Sentiment (0: Negative, 1: Positive)')
axes[1, 0].set_ylabel('Review Length')

# Review length distribution
axes[1, 1].hist(df['review_length'], bins=50, alpha=0.7, color='purple')
axes[1, 1].set_title('Distribution of Review Lengths')
axes[1, 1].set_xlabel('Review Length (characters)')
axes[1, 1].set_ylabel('Frequency')

plt.tight_layout()
plt.show()

# Word Cloud Analysis
print("\nGenerating Word Clouds...")

# Word cloud for positive reviews
positive_reviews = df[df['sentiment'] == 1]['cleaned_review'].str.cat(sep=' ')
wordcloud_positive = WordCloud(width=800, height=400, background_color='white', 
                              colormap='Greens').generate(positive_reviews)

# Word cloud for negative reviews
negative_reviews = df[df['sentiment'] == 0]['cleaned_review'].str.cat(sep=' ')
wordcloud_negative = WordCloud(width=800, height=400, background_color='white', 
                              colormap='Reds').generate(negative_reviews)

# Display word clouds
fig, axes = plt.subplots(1, 2, figsize=(16, 8))

axes[0].imshow(wordcloud_positive, interpolation='bilinear')
axes[0].set_title('Most Common Words in Positive Reviews', fontsize=16)
axes[0].axis('off')

axes[1].imshow(wordcloud_negative, interpolation='bilinear')
axes[1].set_title('Most Common Words in Negative Reviews', fontsize=16)
axes[1].axis('off')

plt.tight_layout()
plt.show()

# Most common words analysis
def get_top_words(text_series, n=20):
    """Get top n words from text series"""
    all_text = ' '.join(text_series)
    words = all_text.split()
    return Counter(words).most_common(n)

print("\nTop 10 words in positive reviews:")
positive_words = get_top_words(df[df['sentiment'] == 1]['cleaned_review'], 10)
for word, count in positive_words:
    print(f"{word}: {count}")

print("\nTop 10 words in negative reviews:")
negative_words = get_top_words(df[df['sentiment'] == 0]['cleaned_review'], 10)
for word, count in negative_words:
    print(f"{word}: {count}")

# 5. Feature Engineering with TF-IDF
print("\n\n4. FEATURE ENGINEERING WITH TF-IDF")
print("-" * 50)

# TF-IDF Vectorization
tfidf_vectorizer = TfidfVectorizer(
    max_features=5000,  # Limit to top 5000 features
    ngram_range=(1, 2),  # Include unigrams and bigrams
    min_df=2,  # Ignore terms that appear in less than 2 documents
    max_df=0.95  # Ignore terms that appear in more than 95% of documents
)

# Prepare features and target
X = tfidf_vectorizer.fit_transform(df['cleaned_review'])
y = df['sentiment']

print(f"TF-IDF Feature Matrix Shape: {X.shape}")
print(f"Number of samples: {X.shape[0]}")
print(f"Number of features: {X.shape[1]}")

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training set size: {X_train.shape[0]}")
print(f"Test set size: {X_test.shape[0]}")
print(f"Training set sentiment distribution:")
print(y_train.value_counts(normalize=True))

# 6. Model Training and Selection
print("\n\n5. MODEL TRAINING AND SELECTION")
print("-" * 50)

# Dictionary to store models and their performance
models = {}
results = {}

# 1. Logistic Regression
print("Training Logistic Regression...")
lr_model = LogisticRegression(random_state=42, max_iter=1000)
lr_model.fit(X_train, y_train)
lr_pred = lr_model.predict(X_test)
models['Logistic Regression'] = lr_model

# 2. Naive Bayes
print("Training Naive Bayes...")
nb_model = MultinomialNB()
nb_model.fit(X_train, y_train)
nb_pred = nb_model.predict(X_test)
models['Naive Bayes'] = nb_model

# 3. Random Forest
print("Training Random Forest...")
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)
models['Random Forest'] = rf_model

# 4. Support Vector Machine
print("Training SVM...")
svm_model = SVC(kernel='linear', random_state=42, probability=True)
svm_model.fit(X_train, y_train)
svm_pred = svm_model.predict(X_test)
models['SVM'] = svm_model

# Store predictions
predictions = {
    'Logistic Regression': lr_pred,
    'Naive Bayes': nb_pred,
    'Random Forest': rf_pred,
    'SVM': svm_pred
}

# 7. Model Evaluation
print("\n\n6. MODEL EVALUATION")
print("-" * 50)

# Calculate metrics for all models
for model_name, pred in predictions.items():
    accuracy = accuracy_score(y_test, pred)
    precision = precision_score(y_test, pred)
    recall = recall_score(y_test, pred)
    f1 = f1_score(y_test, pred)
    
    results[model_name] = {
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1
    }
    
    print(f"\n{model_name} Performance:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")

# Create results dataframe for comparison
results_df = pd.DataFrame(results).T
print("\n" + "="*60)
print("MODEL COMPARISON SUMMARY")
print("="*60)
print(results_df.round(4))

# Find best model
best_model_name = results_df['F1-Score'].idxmax()
best_model = models[best_model_name]
best_predictions = predictions[best_model_name]

print(f"\nBest Model: {best_model_name}")
print(f"Best F1-Score: {results_df.loc[best_model_name, 'F1-Score']:.4f}")

# Detailed evaluation of best model
print(f"\n\nDETAILED EVALUATION OF {best_model_name.upper()}")
print("-" * 50)

# Classification report
print("Classification Report:")
print(classification_report(y_test, best_predictions, 
                          target_names=['Negative', 'Positive']))

# Confusion Matrix
cm = confusion_matrix(y_test, best_predictions)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Negative', 'Positive'],
            yticklabels=['Negative', 'Positive'])
plt.title(f'Confusion Matrix - {best_model_name}')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

# Model comparison visualization
plt.figure(figsize=(12, 8))
results_df.plot(kind='bar', ax=plt.gca())
plt.title('Model Performance Comparison')
plt.xlabel('Models')
plt.ylabel('Score')
plt.legend(loc='lower right')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 8. Feature Importance Analysis (for tree-based models)
if 'Random Forest' in best_model_name:
    print(f"\n\nFEATURE IMPORTANCE ANALYSIS")
    print("-" * 50)
    
    # Get feature names
    feature_names = tfidf_vectorizer.get_feature_names_out()
    
    # Get feature importances
    importances = best_model.feature_importances_
    
    # Create feature importance dataframe
    feature_importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values('importance', ascending=False)
    
    print("Top 20 Most Important Features:")
    print(feature_importance_df.head(20))
    
    # Plot feature importance
    plt.figure(figsize=(12, 8))
    top_features = feature_importance_df.head(20)
    plt.barh(range(len(top_features)), top_features['importance'])
    plt.yticks(range(len(top_features)), top_features['feature'])
    plt.xlabel('Feature Importance')
    plt.title('Top 20 Most Important Features (Random Forest)')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()

# 9. Prediction Function for New Reviews
def predict_sentiment(review_text, model=best_model, vectorizer=tfidf_vectorizer):
    """
    Predict sentiment for a new review
    """
    # Preprocess the text
    cleaned_text = advanced_preprocess_text(review_text)
    
    # Vectorize the text
    text_vector = vectorizer.transform([cleaned_text])
    
    # Make prediction
    prediction = model.predict(text_vector)[0]
    
    # Get probability if available
    if hasattr(model, 'predict_proba'):
        probability = model.predict_proba(text_vector)[0]
        confidence = max(probability)
    else:
        # For models without predict_proba, use decision function if available
        if hasattr(model, 'decision_function'):
            decision_score = model.decision_function(text_vector)[0]
            # Convert decision score to probability-like confidence
            confidence = 1 / (1 + np.exp(-abs(decision_score)))
        else:
            confidence = 1.0  # Default confidence
    
    sentiment = "Positive" if prediction == 1 else "Negative"
    
    return sentiment, confidence

# 10. Testing with Sample Reviews
print("\n\n7. TESTING WITH SAMPLE REVIEWS")
print("-" * 50)

# Test reviews
test_reviews = [
    "This product is absolutely amazing! Great quality and fast delivery.",
    "Terrible quality, waste of money. Would not recommend to anyone.",
    "Average product, nothing special but does the job.",
    "Outstanding performance! Exceeded my expectations completely.",
    "Poor customer service and defective product. Very disappointed."
]

print("Sample Predictions:")
for i, review in enumerate(test_reviews, 1):
    sentiment, confidence = predict_sentiment(review)
    print(f"\nReview {i}: \"{review[:50]}...\"")
    print(f"Predicted Sentiment: {sentiment}")
    print(f"Confidence: {confidence:.4f}")

# 11. Save the trained model and vectorizer
print("\n\n8. SAVING THE MODEL")
print("-" * 50)

import pickle

# Save the best model
with open('best_sentiment_model.pkl', 'wb') as f:
    pickle.dump(best_model, f)

# Save the vectorizer
with open('tfidf_vectorizer.pkl', 'wb') as f:
    pickle.dump(tfidf_vectorizer, f)

print("Model and vectorizer saved successfully!")
print(f"Best model ({best_model_name}) saved as 'best_sentiment_model.pkl'")
print("TF-IDF vectorizer saved as 'tfidf_vectorizer.pkl'")

# 12. Summary and Insights
print("\n\n" + "="*80)
print("PROJECT SUMMARY")
print("="*80)

print(f"""
Dataset Analysis:
- Total reviews analyzed: {len(df):,}
- Positive reviews: {sentiment_counts[1]:,} ({sentiment_counts[1]/len(df)*100:.1f}%)
- Negative reviews: {sentiment_counts[0]:,} ({sentiment_counts[0]/len(df)*100:.1f}%)

Best Model Performance:
- Model: {best_model_name}
- Accuracy: {results_df.loc[best_model_name, 'Accuracy']:.4f}
- F1-Score: {results_df.loc[best_model_name, 'F1-Score']:.4f}
- Precision: {results_df.loc[best_model_name, 'Precision']:.4f}
- Recall: {results_df.loc[best_model_name, 'Recall']:.4f}
""")

print("\n" + "="*80)
print("PROJECT COMPLETED SUCCESSFULLY!")
print("="*80)
