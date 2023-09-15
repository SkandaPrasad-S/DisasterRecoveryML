import sys
# import libraries
import nltk
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')
#nltk.download()
import sys
import re
import nltk
import pandas as pd
import numpy as np
import pickle
from sqlalchemy import create_engine
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline


def load_data(database_filepath):
    """
    Function that loads messages and categories from database using database_filepath as a filepath and sqlalchemy as library
    Returns two dataframes X and y

    """
    engine = create_engine('sqlite:///' + str (database_filepath))
    df = pd.read_sql ('SELECT * FROM MessagesCategories', engine)
    X = df ['message']
    y = df.iloc[:,4:]

    return X, y


def tokenize(text,url_place_holder_string="urlplaceholder"):
    """
    Tokenize the text function
    
    Arguments:
        text -> Text message which needs to be tokenized
    Output:
        clean_tokens -> List of tokens extracted from the provided text
    """
    
    # Replace all urls with a urlplaceholder string
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    
    # Extract all the urls from the provided text 
    detected_urls = re.findall(url_regex, text)
    
    # Replace url with a url placeholder string
    for detected_url in detected_urls:
        text = text.replace(detected_url, url_place_holder_string)

    # Extract the word tokens from the provided text
    tokens = nltk.word_tokenize(text)
    
    #Lemmanitizer to remove inflectional and derivationally related forms of a word
    lemmatizer = nltk.WordNetLemmatizer()

    # List of clean tokens
    clean_tokens = [lemmatizer.lemmatize(w).lower().strip() for w in tokens]
    return clean_tokens


def build_model():
    #setting pipeline
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ]) 
    from sklearn.model_selection import GridSearchCV

    # Define the parameter grid for grid search
    param_grid = {
        'clf__estimator__min_samples_split': [2, 5, 10],
        'clf__estimator__min_samples_leaf': [1, 2, 4]
    }

    # Create a grid search instance
    cv = GridSearchCV(pipeline, param_grid, cv=3, n_jobs=-1)

    return cv
    #return pipeline

def evaluate_model(model, X_test, Y_test):
    """Evaluate the model on the test set and print classification reports."""
    Y_pred_test = model.predict(X_test)
    for i, column in enumerate(Y_test.columns):
        print(f"Category: {column}\n")
        print("Test Set:\n", classification_report(Y_test[column], Y_pred_test[:, i]))
        print("=" * 80)

def save_model(model, model_filepath):
    """Saving model using pickle."""
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        model2 = model.best_estimator_
        save_model(model2, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()