import ssl
from ssl import _create_unverified_context

import nltk
from nltk.corpus import wordnet as wn

from nltk.corpus.reader import Synset

try:
    _create_unverified_https_context = _create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

nltk.download()  # comment this line once you installed the 3 packages

syn = wn.synsets(
    'car')  # obtain all the synsets (sets of synonyms) that include the term "car". It works with the WordNet page
print("synsets containing the lemma car", syn)

car: Synset = wn.synset(
    'car.n.01')  # we want to focus on the first synset which correspond to the car definition we want

print("car lemma names:", car.lemma_names())  # synonyms of car
print("car definition:", car.definition())
print("car hyponyms:", car.hyponyms())  # subclasses of car
print("car hypernyms:", car.hypernyms())  # superclasses of car

print()
print("Similarities:")  # we are going to calculate similarities between synsets

e = wn.synset('motor_vehicle.n.01')

print("car and motor_vehicle:", car.path_similarity(e))  # similarities between car synset and direct hypernym

h = wn.synset('horse.n.01')

print("car and horse:", car.path_similarity(h))

renoir = wn.synsets('renoir')
print(renoir)

impressionist = wn.synsets('impressionist')
print(impressionist)

print("renoir and impressionist:", renoir[0].path_similarity(impressionist[0]))
