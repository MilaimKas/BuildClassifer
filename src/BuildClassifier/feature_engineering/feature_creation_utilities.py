"""
Contains functions to produce encoded features
"""

# common packages
import numpy as np
import pandas as pd

# ML related
from sklearn.preprocessing import OneHotEncoder

# geo features
from collections import Counter
import country_converter as coco

# string processing
from nostril import nonsense
import string
import re

# import decorators and helpers
from BuildClassifier.feature_engineering.decorators import *

# string processing
import spacy
nlp_en = spacy.load("en_core_web_sm") # Load the English model
nlp_de = spacy.load("de_core_news_sm") # Load the German model

from loguru import logger


# String feature extraction
# -------------------------------------------------------------------------------------------------

# name of the meta features
meta_feat = ["length", "special_to_char_ratio", "uppercase_ratio", "vowel_to_consonant_ratio",
        "digit_to_char_ratio", "special_to_char_ratio", "words_ratio"
        ]

@handle_exceptions()
def extract_isnonsense(s: str) -> bool:
    """
    returns True is the string has no sense, False otherwise.
    retruns None if str cannot be proccesed (ex: too short)
    """
    try:
        res = nonsense(s)
    except:
        res = None
    return res

@handle_exceptions(default=[0]*7)
def extract_meta_features(text: str) -> list:
    """
    Extracts meta information from a string.
    
    Args:
        text (str): The input string.
        
    Returns:
        dict: A dictionary containing meta features of the string.
    """
    # Preprocessing: Remove spaces to focus on content
    stripped_text = text.replace(" ", "")

    if not stripped_text:
        return [0]*7
    
    length = len(stripped_text)
    num_vowels = sum(1 for char in stripped_text.lower() if char in 'aeiou')
    num_consonants = sum(1 for char in stripped_text.lower() if char.isalpha() and char not in 'aeiou')
    digit_to_char_ratio = sum(1 for char in stripped_text if char.isalnum()) / length if length > 0 else 0
    special_to_char_ratio = sum(1 for char in stripped_text if char in string.punctuation) / length if length > 0 else 0
    uppercase_ratio = sum(1 for char in stripped_text if char.isupper()) / length if length > 0 else 0
    vowel_to_consonant_ratio = num_vowels / num_consonants if num_consonants > 0 else 0
    words_ratio = len(text.split()) / length if length > 0 else 0
    
    return [
        length, special_to_char_ratio, uppercase_ratio, vowel_to_consonant_ratio,
        digit_to_char_ratio, special_to_char_ratio, words_ratio
    ]

@validate_column
def create_meta_features(df, text_column, pre_fix=""):
    df_new = df.copy()
    df_new[[pre_fix+f for f in meta_feat]]= df[text_column].apply(lambda x: extract_meta_features(x)).apply(pd.Series)
    return df_new

# string processing
@handle_exceptions()
def preprocess_text(text:str, thresh_token_len=2, lang="en"):
    if lang == "en":
        nlp = nlp_en
    elif lang == "de":
        nlp = nlp_de
    else:
        raise ValueError("Only english (en) and german (de) are supported")
    doc = nlp(str(text))
    return [token.lemma_ for token in doc if not token.is_stop and not token.is_punct and len(token)>thresh_token_len]

def generate_uni_di_tri_grams(text, lang="en"):
    
    tokens = preprocess_text(text, lang=lang)
    
    # Unigram frequencies
    unigram_freq = Counter(tokens)
    
    # Bigram frequencies
    bigrams = generate_ngrams(tokens, n=2)
    bigram_freq = Counter(bigrams)
    
    # Trigram frequencies
    trigrams = generate_ngrams(tokens, n=3)
    trigram_freq = Counter(trigrams)
    
    return unigram_freq, bigram_freq, trigram_freq


# Time related features
# ----------------------------------------------------------------------------------------------

def encode_day_time(ts):
    s = str(ts).split(":")
    try:
        res = (float(s[0])*3600 + float(s[1])*60 +float(s[0]))/86399
    except:
        res = None
    return res

def create_day_time(date_serie):
    return pd.to_datetime(date_serie).dt.time.apply(encode_day_time)

def create_day_year(date_serie):
    return pd.to_datetime(date_serie).dt.to_period("D").dt.dayofyear/366

def create_weekday(date_serie):
    return pd.to_datetime(date_serie).dt.dayofweek


# One hot encoding helper
# --------------------------------------------------------------------------------------------

def onehot_encode_with_vocab(df:pd.DataFrame, categorical_vocab:dict):
    """
    One-hot encode selected categorical features with fixed vocabularies.
    
    Args:
        df (pd.DataFrame): Input DataFrame to encode.
        categorical_vocab (dict): Dictionary of the form
                                  {feature_name: list_of_allowed_categories}.
    
    Returns:
        pd.DataFrame: One-hot encoded DataFrame with consistent columns.
    """
    encoded_parts = []

    for feature, allowed_categories in categorical_vocab.items():
        if feature not in df.columns:
            raise ValueError(f"Feature '{feature}' not found in input DataFrame.")
        
        # Extract unique categories in the data
        observed_categories = df[feature].dropna().unique()
        unexpected = set(observed_categories) - set(allowed_categories)
        if unexpected:
            logger.warning(f"Feature '{feature}' contains unknown categories: {unexpected}. These will be ignored.")

        # Clip categories to allowed set
        safe_col = df[feature].where(df[feature].isin(allowed_categories))

        # One-hot encode with allowed categories
        dummies = pd.get_dummies(safe_col, prefix=feature)

        # Ensure all dummy columns are present
        expected_cols = [f"{feature}_{cat}" for cat in allowed_categories]
        dummies = dummies.reindex(columns=expected_cols, fill_value=0)

        encoded_parts.append(dummies)

    return pd.concat(encoded_parts, axis=1)


# Other usuefull functions
# --------------------------------------------------------------------------------------------

def calculate_entropy(values):
    """
    Calculate entropy for categorical values.

    Parameters:
        values (list): A list of n categorical values.

    Returns:
        float: Entropy score.
    """
    # Count the frequency of each unique value
    counts = Counter(values)
    
    # Total number of values
    total = sum(counts.values())
    
    # Calculate probabilities
    probabilities = [count / total for count in counts.values()]
    
    # Compute entropy
    entropy = -sum(p * np.log2(p) for p in probabilities)/np.log(len(values))
    
    return np.abs(entropy)
