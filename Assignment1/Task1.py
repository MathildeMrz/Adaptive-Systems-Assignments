#!/usr/bin/env python

import re
import csv
import datetime

from gensim import corpora
from gensim import models
from gensim import similarities
from pprint import pprint  # pretty-printer

from nltk import PorterStemmer
from nltk.corpus import stopwords

init_t: datetime = datetime.datetime.now()

# TASK 1
# Implement the following pseudocode to calculate the variable ratio_quality using the TFIDF vectors:<br>
# total_goods = 0<br>
# For every article (a) on topic "Food and Drink":<br>
#    Obtain the top-10 most similar articles (top-10) in Corpus to a<br>
#    Count how many articles in top-10 are related to topic "Food and Drink" (goods)<br>
#    total_goods = total_goods + goods<br>
# ratio_quality = total_goods/(num_articles_food_and_drink*10)<br>
# And measure the execution times separately for the following two subprocesses: <br>
# Creating the model (from the program begin to the call similarities.MatrixSimilarity(tfidf_vectors))<br>
# Implementation of the pseudocode above.<br>
# 

# Implementation of the pseudocode
def ratio_quality_using_TFIDF(topic_Food_and_Drink, goods, num_articles_food_and_drink):
    total_goods = 0
    for a in topic_Food_and_Drink :
        total_goods = total_goods + goods
        #    Obtain the top-10 most similar articles (top-10) in Corpus to a<br>
        #    Count how many articles in top-10 are related to topic "Food and Drink" (goods)<br>
    ratio_quality = total_goods/(num_articles_food_and_drink * 10)
    return ratio_quality

# import our csv file and read it
csv_file_path = './news.csv'
csv_full_text = ""

with open(csv_file_path, newline='') as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        row_text = ' '.join(row)
        csv_full_text += row_text + '\n'
# print the text 
print(csv_full_text)


porter = PorterStemmer()

# remove common words and tokenize
stoplist = stopwords.words('english')
texts = [
    [porter.stem(word) for word in csv_full_text.lower().split() if word not in stoplist]
    for csv_full_text in csv_full_text
]


# create mapping keyword-id
dictionary = corpora.Dictionary(texts)
print()
print("Mapping keyword-id:")
pprint(dictionary.token2id)

id2token = dict(dictionary.items())


# create the vector for each doc
model_bow = [dictionary.doc2bow(text) for text in texts]


# create the LDA model from bow vectors
lda = models.LdaModel(model_bow, num_topics=2, id2word=dictionary, random_state=30)
# random_state: forced to always obtain the same results in all the executions
lda_vectors = []
for v in model_bow:
    lda_vectors.append(lda[v])

print()
print("LDA vectors for docs (in terms of topics):")
i = 0
for v in lda_vectors:
    print(v, documents[i])
    i += 1

matrix_lda = similarities.MatrixSimilarity(lda_vectors)
print()
print("Matrix similarities")
print(matrix_lda)

def convert(match):
    return dictionary.id2token[int(match.group(0)[1:-1])]

print("LDA Topics:")
for t in lda.print_topics(num_words=30):
    print(re.sub('"[0-9]+"', convert, str(t)))

end_creation_model_t: datetime = datetime.datetime.now()

print()


# obtain LDA vector for the following doc<br>
# doc = "Human computer interaction"

doc = "trees graph human"
doc_s = [porter.stem(word) for word in doc.lower().split() if word not in stoplist]

vec_bow = dictionary.doc2bow(doc_s)
vec_lda = lda[vec_bow]


# calculate similarities between doc and each doc of texts using lda vectors and cosine
sims = matrix_lda[vec_lda]


# sort similarities in descending order
sims = sorted(enumerate(sims), key=lambda item: -item[1])

print()
print("Given the doc: " + doc)
print("whose LDA vector is: " + str(vec_lda))
print()
print("The Similarities between this doc and the documents of the corpus are:")
for doc_position, doc_score in sims:
    print(doc_score, documents[doc_position])

end_t: datetime = datetime.datetime.now()


# get execution time
elapsed_time_model_creation: datetime = end_creation_model_t - init_t
elapsed_time_comparison: datetime = end_t - end_creation_model_t
print()
print('Execution time model:', elapsed_time_model_creation, 'seconds')
print('Execution time comparison:', elapsed_time_comparison, 'seconds')