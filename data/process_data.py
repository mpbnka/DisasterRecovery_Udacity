# import libraries
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import sys


def load_data(messages_filepath, categories_filepath):
    """Load data from the csv files

    Args:
        messages_filepath (String): Path to messages csv
        categories_filepath (String): Path to categories csv

    Returns:
        df(pandas.DataFrame): pandas dataframe object
    """
    # load messages dataset
    messages = pd.read_csv('disaster_messages.csv')
    # load categories dataset
    categories = pd.read_csv('disaster_categories.csv')
    # merge datasets
    df = messages.merge(categories, on=["id"])
    return df

def clean_data(df):
    """Clean the dataframe and remove dulpicates

    Args:
        df (pandas.DataFrame): [description]
    """
    categories = df["categories"].str.split(pat=';', expand=True)   # create a dataframe of the 36 individual category columns
    row = categories.head(1)    # select the first row of the categories dataframe
    # use this row to extract a list of new column names for categories.
    # one way is to apply a lambda function that takes everything 
    # up to the second to last character of each string with slicing
    category_colnames = row.applymap(lambda x: x[:-2]).iloc[0, :].tolist()
    
    categories.columns = category_colnames  # rename the columns of `categories`
    for column in categories:
        categories[column] = categories[column].astype(str).str[-1] # set each value to be the last character of the string
        categories[column] = categories[column].astype(int) # convert column from string to numeric
    
    df.drop("categories", axis=1, inplace=True) # drop the original categories column from `df`
    df = pd.concat([df, categories], axis=1)    # concatenate the original dataframe with the new `categories` dataframe
    df.drop_duplicates(inplace=True)   # drop duplicates
    return df

def save_data(df, database_filename):
    """Save the dataframe as a sqlite db

    Args:
        df (pandas.DataFrame): DataFrame to be saved
        database_filename (String): Name of Database to be saved as
    """
    engine = create_engine('sqlite:///' + database_filename)
    df.to_sql('MessageCategories', engine, index=False)  


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()