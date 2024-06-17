
import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
import re
from tqdm import tqdm
import regex as re

def preprocess_text(text):
    stemmer = SnowballStemmer("english")
    words = text.split()
    words = [stemmer.stem(word) for word in words]
    return ' '.join(words)

def stop_words(texts):
    stop_words = set(stopwords.words('english'))
    regex = re.compile('[^a-zA-Z]')
    preprocess_texts = []

    for text in tqdm(texts):
        text = text.lower()
        text = regex.sub(' ', text)
        word_tokens = nltk.word_tokenize(text)
        filtered_sentence = [w for w in word_tokens if (w not in stop_words) & (w != 'url')  & (w != 'username') & (w != 'via') & (w != 'ha') & (w != 'user')  & (w != 'number')]

        processed = []
        for word in filtered_sentence:
            word = re.sub(r'^.+@[^\.].*\.[a-z]{2,}$', 'emailaddress', word)
            word = re.sub(r'^http\://[a-zA-Z0-9\-\.]+\.[a-zA-Z]{2,3}(/\S*)?$', 'webaddress', word)
            word = re.sub(r'^\(?[\d]{3}\)?[\s-]?[\d]{3}[\s-]?[\d]{4}$', 'phonenumbr', word)
            processed.append(word)

        preprocess_texts.append(' '.join(processed))

    return preprocess_texts

def preprocess(data):
    
    # data['text'] = data['text'].apply(preprocess_text)
    data['text'] = stop_words(data['text'])

    return data