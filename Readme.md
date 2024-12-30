**Yelp Review Classification with KNN**

**Overview:**

             This project implements a k-Nearest Neighbor (KNN) classifier to predict the sentiment of Yelp restaurant reviews. Reviews are classified as either positive (+1) or negative (-1). The training dataset consists of 18,000 labeled reviews, and the test dataset includes 18,000 unlabeled reviews.

The project involves text preprocessing, feature engineering, model building, cross-validation, and testing. The goal is to achieve the highest possible accuracy while adhering to the rules and requirements provided.

**Features:**

1) Text Preprocessing

* Tokenization and lowercasing of text.  
* Removal of stop words and punctuation.  
* Lemmatization for better representation of words.  
* Conversion to document-term matrix using CountVectorizer.  
* Transformation to TF-IDF representation using TfidfTransformer.


2) Feature Selection

* Chi-squared statistical tests were used to select the top 3000 features for training.


3) Distance/Similarity Measures

* Implementation of multiple distance metrics: Euclidean distance, Manhattan distance, and Cosine similarity (optimized for text data).


4) Model Training

* Custom implementation of the KNN algorithm.  
* Cross-validation to tune hyperparameters such as k (number of neighbors).  
* Optimal performance achieved with k=13 and Cosine similarity.


5) Evaluation

* Accuracy measured using the formula:  
              **Accuracy \= (True Positives \+ True Negatives) / Total Predictions**

​

* Confusion matrix used to evaluate model performance.


6) Testing and Results

* Test predictions submitted for leaderboard evaluation.  
* Achieved an accuracy of 78% on the test set with optimal parameters.

## **Getting Started**

### **Prerequisites**

* Python 3.8 or later  
* Required Python libraries:  
  * numpy  
  * pandas  
  * scikit-learn  
  * nltk  
  * matplotlib  
    

## **Results**

* **Accuracy**: 78% on the test set.  
* **Optimal Parameters**:  
  * Number of Neighbors (`k`): 13  
  * Similarity Metric: Cosine Similarity  
* **Feature Reduction**: Top 3000 features selected using Chi-squared tests.

## **Acknowledgments:**

## This project is part of the CS 584 (Data Mining) course at George Mason University, Fall 2023\.

