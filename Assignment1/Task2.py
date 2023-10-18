import datetime
import csv

from gensim import corpora
from gensim import models
from gensim import similarities

from nltk.corpus import stopwords
from nltk import PorterStemmer

# Initiating the time for the execution time calculation
init_t: datetime = datetime.datetime.now()

# CSV extraction
news_file = "news.csv"

all_news = []
descriptions = []
food_drink_descriptions = []

# Read CSV
with open(news_file, 'r', newline='', encoding='utf-8') as csv_file:
    reader_csv = csv.reader(csv_file)
    for line in reader_csv:
        all_news.append(line)
        descriptions.append(line[3])

        if "Food & Drink" in line[2]:
            food_drink_descriptions.append(line[3])

num_articles_food_and_drink = len(food_drink_descriptions)

# Delete the first row 'description' name
descriptions.pop(0)

porter = PorterStemmer()
# Remove common words and tokenize
stoplist = stopwords.words('english')

# Array of list of words for each document
texts = [
    [porter.stem(word) for word in document.lower().split() if word not in stoplist]
    for document in descriptions
]

# create mapping keyword-id
dictionary = corpora.Dictionary(texts)

model_bow = [dictionary.doc2bow(text) for text in texts]

# Create the LDA model from bow vectors
# Use 30 topics, two passes, and a random state parameter
lda = models.LdaModel(model_bow, num_topics=30, id2word=dictionary, random_state=30, passes=2)

# random_state: forced to always obtain the same results in all the executions
lda_vectors = []
for v in model_bow:
    lda_vectors.append(lda[v])

matrix_lda = similarities.MatrixSimilarity(lda_vectors)

end_creation_model_t: datetime = datetime.datetime.now()

total_goods = 0

# Pseudo-code
for food_drink_description in food_drink_descriptions:
    # Filtering the food and drink descriptions with stopwords and other regex expressions
    doc_s = [porter.stem(word) for word in food_drink_description.lower().split() if word not in stoplist]

    vec_bow = dictionary.doc2bow(doc_s)
    vec_tfidf = lda[vec_bow]

    # Calculating similarities between doc and each doc of texts using tfidf vectors and cosine
    sims = matrix_lda[vec_tfidf]  # sims is a list a similarities

    # Sorting similarities in descending order
    sims = sorted(enumerate(sims), key=lambda item: -item[1])

    # Selecting the 10 most similar elements
    top_10_similar_elements = sims[:10]
    goods = 0

    # Checking if these 10 elements are part of Drink & Food topics
    for doc_position, doc_score in top_10_similar_elements:
        print("Score")
        print(doc_score)
        print("Topic")
        print(all_news[doc_position][2])
        if all_news[doc_position][2] == "Food & Drink":
            print("Good ! ")
            goods = goods + 1

    total_goods = total_goods + goods
    print(total_goods)

ratio_quality = total_goods / (num_articles_food_and_drink * 10)
print("total_goods = ", total_goods)
print("num_articles_food_and_drink = ", num_articles_food_and_drink)
print("ratio_quality = ", ratio_quality)

end_t: datetime = datetime.datetime.now()  # to mark the end of the program

# Get execution time
elapsed_time_model_creation: datetime = end_creation_model_t - init_t
elapsed_time_comparison: datetime = end_t - end_creation_model_t
print()
print('Execution time model:', elapsed_time_model_creation, 'seconds')
print('Execution time comparison:', elapsed_time_comparison, 'seconds')
