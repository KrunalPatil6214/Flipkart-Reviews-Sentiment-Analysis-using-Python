# 🛍️ Flipkart Reviews Sentiment Analysis Using Python

A comprehensive machine learning project that analyzes Flipkart product reviews to classify customer sentiment as positive or negative using various ML algorithms.

## 📋 Project Overview

**Objective:** [Develop a Machine Learning model to analyze Flipkart product reviews and classify them as positive or negative based on user sentiment.](Flipkart_Reviews_Sentiment_Analysis_using_Python.pdf)

**Dataset:** [Flipkart Reviews Dataset](flipkart_data.csv)


## 🎯 Key Features

- ✅ **Data Preprocessing** - Cleaning, tokenization, stemming, and stopword removal, Remove missing values and duplicate entries.
- ✅ **TF-IDF Vectorization** - Converting text to numerical features for ML models.
- ✅ **Multiple ML Models** - Logistic Regression, Naive Bayes, Random Forest, SVM.
- ✅ **Comprehensive EDA** - Data visualization and sentiment distribution analysis.
- ✅ **Word Cloud Analysis** - Visual representation of most common words.
- ✅ **Model Comparison** - Performance evaluation using multiple metrics.
- ✅ **Real-time Prediction** - Function to classify new reviews instantly.

## 🛠️ Technologies Used
- **Python 3.8+**
- **Pandas** - Data manipulation and analysis.
- **Scikit-learn** - Machine learning algorithms and evaluation.
- **NLTK** - Natural language processing.
- **Matplotlib & Seaborn** - Data visualization.
- **WordCloud** - Text visualization.
- **Numpy** - Numerical computations.


## 📈 Project Workflow

### 1. Data Loading & Exploration
- Load Flipkart reviews dataset
- Explore data structure and quality
- Analyze rating distributions

### 2. Data Preprocessing
- Remove duplicates and missing values
- Create sentiment labels (Positive: 4-5 stars, Negative: 1-3 stars)
- Clean and preprocess text data

### 3. Text Processing & Feature Engineering
- Convert text to lowercase
- Remove punctuation, special characters, and stopwords
- Apply stemming and tokenization
- Generate TF-IDF features

### 4. Exploratory Data Analysis
- Visualize sentiment distribution
- Create word clouds for positive/negative reviews
- Analyze review length patterns
- Generate insightful visualizations

### 5. Model Training & Comparison
- Train multiple ML models:
  - 🔄 **Logistic Regression**
  - 🔄 **Naive Bayes**
  - 🔄 **Random Forest**
  - 🔄 **Support Vector Machine (SVM)**
- Compare performance using accuracy, precision, recall, F1-score

### 6. Model Evaluation
- Detailed analysis of best performing model
- Confusion matrix visualization
- Performance metrics calculation

### 7. Sentiment Prediction
- Create prediction function for new reviews
- Test with sample reviews
- Interactive sentiment analysis

## 📊 Expected Results

The project typically achieves:

- **Accuracy**: 85-95% on test data
- **F1-Score**: High performance across different models
- **Best Model**: Usually Logistic Regression or SVM performs best
- **Insights**: Clear patterns in positive vs negative review language


## 📈 Key Insights & Findings

- **📊 Dataset Balance**: Analysis reveals distribution of positive vs negative reviews
- **🔤 Word Patterns**: Common words in positive reviews include "good", "excellent", "amazing"
- **📏 Review Length**: Positive reviews tend to be more descriptive
- **🎯 Model Performance**: Multiple models achieve high accuracy with proper preprocessing

## 🔮 Future Enhancements

### Phase 2 Improvements:
- 🧠 **Deep Learning Models** (LSTM, BERT, Transformers)
- 🎭 **Multi-class Classification** (5-star rating prediction)
- 🔍 **Aspect-based Sentiment Analysis**
- 📱 **Web API Development** for real-time analysis
- ⚡ **Performance Optimization** for large-scale processing

### Business Applications:
- 🏪 **E-commerce Integration** for real-time review monitoring
- 📊 **Dashboard Development** for business insights
- 🔔 **Alert Systems** for negative sentiment detection
- 📈 **Trend Analysis** for product performance tracking

## 🤝 Contributing

Contributions are welcome! Feel free to:

1. **Fork** the repository
2. **Create** a feature branch
3. **Commit** your changes
4. **Push** to the branch
5. **Create** a Pull Request

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

## 👨‍💻 Author

**Data Science Project**
- 📧 Email: krunal6214@gmail.com
- 💼 LinkedIn: www.linkedin.com/in/krunal-patil-080aa2315

## 🙏 Acknowledgments

- **Flipkart** for providing the review dataset
- **Scikit-learn** community for excellent ML tools
- **NLTK** team for natural language processing capabilities
- **Open source community** for various visualization libraries

---

### 🎉 Ready to Analyze Sentiment? 

**Start exploring customer emotions through data!** 

This project demonstrates the power of machine learning in understanding customer feedback and can be extended for various business applications in e-commerce, product management, and customer service.

**Happy Analyzing! 📊✨**
