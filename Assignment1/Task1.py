import datetime
import csv

from gensim import corpora
from gensim import models
from gensim import similarities

from nltk.corpus import stopwords
from nltk import PorterStemmer


### PART 1
# Data Preprocessing: 
# - Measure the starting execution time for the model
# - CSV import, extraction and saving all the results in different variables
# - Remove all the stopwords inside the 'description' column 
# - Create a dictionary with the words and the relative ids
### 

# Starting time for the subprocess 'model_creation'
init_t: datetime = datetime.datetime.now()

# CSV extraction
news_file = "./news.csv"

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

# Delete the first CSV row of column names
all_news.pop(0)
descriptions.pop(0)

porter = PorterStemmer()
# Remove common words using stopwords 
stoplist = stopwords.words('english')
# Array of list of words for each document
texts = [
    [porter.stem(word) for word in document.lower().split() if word not in stoplist]
    for document in descriptions
]

# create mapping keyword-id
dictionary = corpora.Dictionary(texts)
model_bow = [dictionary.doc2bow(text) for text in texts]


### PART 2
# - Create the TF-IDF model 
# - Measure the final execution time for the model
### 

tfidf = models.TfidfModel(model_bow)
tfidf_vectors = tfidf[model_bow]
# The following matrix will be necessary to calculate similarity between documents
matrix_tfidf = similarities.MatrixSimilarity(tfidf_vectors)

# Final time for the subprocess 'model_creation' + Starting time for the subprocess 'pseudocode'
end_creation_model_t: datetime = datetime.datetime.now() 


### PART 3
# - Create the function based on the pseudocode to calculate the ratio_quality
# - Apply the function to our CSV file
# - Measure the final execution time for the subprocess 'pseudocode' and for the both processes
### 

# Create a function to calculate the ratio_quality
def calculate_ratio_quality(food_drink_descriptions, all_news, num_articles_food_and_drink):
    
    total_goods = 0
    # Filtering the food and drink descriptions with stopwords and other regex expressions
    for food_drink_description in food_drink_descriptions:
        doc_s = [porter.stem(word) for word in food_drink_description.lower().split() if word not in stoplist]

        vec_bow = dictionary.doc2bow(doc_s)
        vec_tfidf = tfidf[vec_bow]
    
        # Calculating similarities between doc and each doc of texts using tfidf vectors and cosine
        sims = matrix_tfidf[vec_tfidf]

        # Sorting similarities in descending order
        sims = sorted(enumerate(sims), key=lambda item: -item[1])

        # Selecting the 10 most similar elements
        top_10_similar_elements = sims[0:10]

        goods = 0
        for doc_position, doc_score in top_10_similar_elements:
            if all_news[doc_position][2] == "Food & Drink":
                print("FOOD&DRINK: \n Score: ", doc_score)
                print("-----------------------------------")
                goods += 1
            else:
                print("Topic: ", all_news[doc_position][2], "\nScore: ", doc_score)
                print("-----------------------------------")

            if all_news[doc_position][3] == food_drink_description:
                print("Comparison of the current article with itself.")
                print("Score: ", doc_score)

        total_goods += goods
        print("current_goods_F&D: ", total_goods)

    ratio_quality = total_goods / (num_articles_food_and_drink * 10)
    
    print("total_goods =", total_goods)
    print("num_articles_food_and_drink =", num_articles_food_and_drink)

    return ratio_quality

# Apply the above function 'calculate_ratio_quality' and print it
ratio_quality = calculate_ratio_quality(food_drink_descriptions, all_news, num_articles_food_and_drink)
print("ratio_quality =", ratio_quality)

'''
total_goods = 0
# Pseudo-code
for food_drink_description in food_drink_descriptions:
    # Filtering the food and drink descriptions with stopwords and other regex expressions
    doc_s = [porter.stem(word) for word in food_drink_description.lower().split() if word not in stoplist]

    vec_bow = dictionary.doc2bow(doc_s)
    vec_tfidf = tfidf[vec_bow]

    # Calculating similarities between doc and each doc of texts using tfidf vectors and cosine
    sims = matrix_tfidf[vec_tfidf]  # sims is a list a similarities

    # Sorting similarities in descending order
    sims = sorted(enumerate(sims), key=lambda item: -item[1])

    # Selecting the 10 most similar elements
    top_10_similar_elements = sims[:10]
    
    goods = 0
    # Checking if these 10 elements are part of Drink & Food topics
    for doc_position, doc_score in top_10_similar_elements:
        print("Topic: ", all_news[doc_position][2], "\nScore: ", doc_score)
        print("-----------------------------------")
        if all_news[doc_position][2] == "Food & Drink":
            print("FOOD&DRINK: \n Score: ", doc_score)
            print("-----------------------------------")
            goods = goods + 1

    total_goods = total_goods + goods
    print("current_goods_F&D: ", total_goods)

ratio_quality = total_goods / (num_articles_food_and_drink * 10)
print("total_goods = ", total_goods)
print("num_articles_food_and_drink = ", num_articles_food_and_drink)
print("ratio_quality = ", ratio_quality)
'''

# Final time for the subprocess 'pseudocode' (but also the programm in general)
end_t: datetime = datetime.datetime.now() 

# Measure the final execution time for both the subprocesses 'model_creation' and 'pseudocode'
elapsed_time_model_creation: datetime = end_creation_model_t - init_t
elapsed_time_pseudocode: datetime = end_t - end_creation_model_t
print()
print('Execution time model:', elapsed_time_model_creation, 'seconds')
print('Execution time comparison:', elapsed_time_pseudocode, 'seconds')