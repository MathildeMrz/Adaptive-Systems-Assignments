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
sports_descriptions = []

# Read CSV
with open(news_file, 'r', newline='', encoding='utf-8') as csv_file:
    reader_csv = csv.reader(csv_file)
    for line in reader_csv:
        all_news.append(line)
        descriptions.append(line[3])

        if "Sports" in line[2]:
            sports_descriptions.append(line[3])

num_articles_sports = len(sports_descriptions)

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
#print("Matrix similarities: ", matrix_tfidf)

# Final time for the subprocess 'model_creation' + Starting time for the subprocess 'pseudocode'
end_creation_model_t: datetime = datetime.datetime.now() 


### PART 3
# - Create the function based on the pseudocode to calculate the ratio_quality
# - Apply the function to our CSV file
# - Measure the final execution time for the subprocess 'pseudocode' and for the both processes
### 

# Create a function to calculate the ratio_quality
def calculate_ratio_quality(topic_descriptions, news, num_articles_topic, vector_type, matrix_vector_type):
    
    total_goods = 0

    # Filtering the food and drink descriptions with stopwords and other regex expressions
    for topic_description in topic_descriptions:
        doc_s = [porter.stem(word) for word in topic_description.lower().split() if word not in stoplist]

        vec_bow = dictionary.doc2bow(doc_s)
        vec_vector_type = vector_type[vec_bow]

        # Calculating similarities between doc and each doc of texts using tfidf vectors and cosine
        sims = matrix_vector_type[vec_vector_type]

        # Sorting similarities in descending order
        sims = sorted(enumerate(sims), key=lambda item: -item[1])

        # Selecting the 10 most similar elements
        top_10_similar_elements = sims[0:10]

        goods = 0

        for doc_position, doc_score in top_10_similar_elements:
            #print("Topic: ", news[doc_position][2], "\nScore: ", doc_score)
            #print("-----------------------------------")
            # when we found the category "Sports", we update the total goods
            if news[doc_position][2] == "Sports":
                goods += 1
            # when we are trying to compare the current document with itself, we should have score=1
            #if news[doc_position][3] == topic_description:
                #print("Comparison of the current article with itself.")
                #print("Score: ", doc_score)
                #print("-----------------------------------")
        total_goods += goods
    ratio = total_goods / (num_articles_topic * 10)

    #print("total_goods =", total_goods)
    #print("num_articles_topic =", num_articles_topic)

    return ratio


# Apply the above function 'calculate_ratio_quality' and print it
ratio_quality = calculate_ratio_quality(sports_descriptions, all_news, num_articles_sports, tfidf, matrix_tfidf)
print("ratio_quality =", ratio_quality)

# Final time for the subprocess 'pseudocode' (but also the program in general)
end_t: datetime = datetime.datetime.now() 

# Measure the final execution time for both the subprocesses 'model_creation' and 'pseudocode'
elapsed_time_model_creation: datetime = end_creation_model_t - init_t
elapsed_time_pseudocode: datetime = end_t - end_creation_model_t
print()
print('Execution time model:', elapsed_time_model_creation, 'seconds')
print('Execution time comparison:', elapsed_time_pseudocode, 'seconds')