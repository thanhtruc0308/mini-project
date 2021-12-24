import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np


full_df = pd.read_csv('./preprocessing_data.csv')
df = full_df[["stopwtext"]]
 


vectorizer = TfidfVectorizer()
doc_vec = vectorizer.fit_transform(df["stopwtext"])
 
# Create dataFrame
feature_names = vectorizer.get_feature_names_out()

df2 = pd.DataFrame(doc_vec.toarray().transpose(), index = feature_names)


print(df2)
df2.to_csv("LSA.csv")

