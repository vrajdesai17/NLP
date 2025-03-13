# NLP

# NLP Projects Collection

## 1. NLP Sentiment Analysis on Amazon Reviews

**Description**  
Perform sentiment analysis on Amazon Office Product reviews. Predict review sentiments based on star ratings.

**Key Features**  
- Data cleaning and preprocessing  
- Text normalization (lemmatization, stop-word removal)  
- Sentiment classification using ML models  

**Dataset**  
- Amazon Reviews Dataset (Office Products)  

**Libraries Used**  
- Pandas  
- NumPy  
- NLTK  
- Scikit-learn  

**How to Run**  
1. Open `NLP_Sentiment_Analysis_Amazon_Reviews.ipynb`  
2. Run all cells to preprocess data and train models  


## 2. Probabilistic Sequence Modeling - HMM POS Tagging

**Description**  
Implements a Hidden Markov Model (HMM) for Part-of-Speech tagging. Builds vocabulary and trains HMM from scratch.

**Key Features**  
- Vocabulary creation with frequency filtering  
- HMM model training with transition and emission probabilities  
- OOV handling using `<unk>` token  

**Libraries Used**  
- Python Standard Libraries (collections, JSON)  

**How to Run**  
1. Open `ProbabilisticSequenceModeling_HMM_POS.ipynb`  
2. Provide training files  
3. Run notebook to build vocabulary and train HMM  


## 3. Word2Vec and TF-IDF Review Classifier

**Description**  
Combines Word2Vec and TF-IDF features for sentiment classification on Amazon reviews.

**Key Features**  
- Word embeddings using Word2Vec  
- TF-IDF feature extraction  
- Classification using Logistic Regression and other ML models  

**Libraries Used**  
- Gensim  
- Scikit-learn  
- NLTK  

**How to Run**  
1. Open `Word2Vec_Tfidf_ReviewClassifier.ipynb`  
2. Run all cells to preprocess data and train models  


## Requirements

- Python 3.x  
- Jupyter Notebook  

```
pip install pandas numpy nltk gensim scikit-learn beautifulsoup4
```

## Author

Vraj Desai  
