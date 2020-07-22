# import libraries
import sys
import re
#%matplotlib inline
import matplotlib.pyplot as plt
import nltk
nltk.download(['punkt', 'wordnet', 'stopwords'])
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import numpy as np
import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sqlalchemy import create_engine

def load_data(database_filepath):
    """Load date from the sqlite database

    Args:
        database_filepath (String): Database filename

    Returns:
        X , Y, category_names
    """
    # load data from database
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    df = pd.read_sql_table("MessageCategories", con=engine)
    X = df["message"]
    y = df[df.columns[4:]]
    category_names = y.columns
    return X, y, category_names


def tokenize(text):
    """Function to tokenize test messages

    Args:
        text (String): String to be tokenized

    Returns:
        clean_tokens (List): tokenized text as a list Object
    """
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)
    return clean_tokens


def build_model():
    """Create pipeline and run a gridsearch based on parameters

    Returns:
        model: Returns the model type obtained by gridsearch
    """
    moc = MultiOutputClassifier(RandomForestClassifier(n_estimators=100))

    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', moc)
        ])
    parameters = {'clf__estimator__max_depth': [10, 50, None],
              'clf__estimator__min_samples_leaf':[2, 5, 10]}

    cv = GridSearchCV(pipeline, parameters, cv=3)
    return cv

def evaluate_model(model, X_test, Y_test, category_names):
    y_pred = model.predict(X_test)

    for i, col in enumerate(Y_test):
        print(col)
        print(classification_report(Y_test[col], y_pred[:, i]))
    # print accuracy score
    print('Accuracy: {}'.format(np.mean(Y_test.values == y_pred)))

def save_model(model, model_filepath):
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()