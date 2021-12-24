import pandas as pd
import re
import nltk
import string 
import spacy
pd.options.mode.chained_assignment = None


full_df = pd.read_csv('./dataset.csv')
df = full_df[["citationStringAnnotated","articleType"]]
df["citationStringAnnotated"] = df["citationStringAnnotated"].astype(str)


# Remove URLs
def remove_urls(text):
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    return url_pattern.sub(r'', text)
df["stopurls"] = df["citationStringAnnotated"].apply(lambda text: remove_urls(text))
print("\n", df["stopurls"])

# Remove HTML tags
from bs4 import BeautifulSoup
def remove_html(text):
    return BeautifulSoup(text, "lxml").text
df["stophtml"] = df["stopurls"].apply(lambda text: remove_html(text))
print("\n", df["stophtml"])

# lower casting
df["lowertext"] = df["stophtml"].str.lower()
print("\n",df["lowertext"].head())

# Remove punctuations
PUNCT = string.punctuation
def remove_punctuation(text):
    return text.translate(str.maketrans('', '', PUNCT))
df["nopuntext"] = df["lowertext"].apply(lambda text: remove_punctuation(text))
print("\n",df["nopuntext"].head())

# Remove stopwords
nltk.download('stopwords')
from nltk.corpus import stopwords
", ".join(stopwords.words('english'))
STOPW = set(stopwords.words('english'))
def remove_stopwords(text):
    return " ".join([word for word in str(text).split() if word not in STOPW])
df["stopwtext"] = df["nopuntext"].apply(lambda text: remove_stopwords(text))
print("\n",df["stopwtext"].head())

# Remove frequent words
from collections import Counter
cnt = Counter()
for text in df["stopwtext"].values:
    for word in text.split():
        cnt[word] += 1
cnt.most_common(10)
FREQWS = set([w for (w,wc) in cnt.most_common(10)])
def remove_freqwords(text):
    return " ".join([word for word in str(text).split() if word not in FREQWS])
df["freqwtext"] = df["stopwtext"].apply(lambda text: remove_freqwords(text))
print("\n", df["freqwtext"].head())

# Stemming
from nltk.stem.porter import PorterStemmer
stemmer = PorterStemmer()
def stem_words(text):
    return " ".join([stemmer.stem(word) for word in text.split()])
df["stemtext"] = df["freqwtext"].apply(lambda text: stem_words(text))
print("\n", df["stemtext"].head())

# Lemmatization
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
def lemmatize_words(text):
    return " ".join([lemmatizer.lemmatize(word) for word in text.split()])
df["lemmatext"] = df["freqwtext"].apply(lambda text: lemmatize_words(text))
print("\n", df["lemmatext"])


pd.DataFrame({'articleType' : df['articleType'],
            'citationStringAnnotated' : df['citationStringAnnotated'], 
            'Lower casting' : df['lowertext'],
            'Remove punctuations' : df['nopuntext'],
            'Remove stopwords': df['stopwtext'],
            'Remove frequent words': df['freqwtext'],
            'Stemming' : df['stemtext'],
            'Lemmatiztion' : df['lemmatext'],
            'Remove URLs': df['stopurls'],
            'Remove HTML tags' : df['stophtml']})
df.to_csv("preprocessing_data.csv")
