import numpy as np
import csv
import random
from nltk.corpus import stopwords

stop_words_set = set(stopwords.words('english'))

PREDICTIONS_ORIGINAL_FILE = 'predictions_original.csv'
PREDICTIONS_DUMMY_FILE = 'predictions_dummy.csv'

neg = 197
pos = 243
neu = 373
tot_pol = (197+243+373)
total_words = 7218
pos_ratio = pos/total_words
neg_ratio = neg/total_words
neu_ratio = neu/total_words
polarity_ratio = tot_pol/total_words

random.seed(0)

def generate_dummy_output_for_line(sentence):
    # assign polarity ratio words as polarities
    # assign neg_ratio as negative words, and so on. But don't assign to stop words.
    words = sentence.split(' ')
    polarity_sentence = []
    for word in words:
        if word in stop_words_set:
            continue
        random_number = random.random()
        if random_number < pos_ratio:
            polarity_sentence.append('{} positive'.format(word))
        if random_number < pos_ratio + neg_ratio:
            polarity_sentence.append('{} negative'.format(word))
        if random_number < pos_ratio + neg_ratio + neu_ratio:
            polarity_sentence.append('{} neutral'.format(word))
    polarity_sentence = ' <sep> '.join(polarity_sentence)
    return polarity_sentence.strip()


def generate_output():
    with open(PREDICTIONS_ORIGINAL_FILE, 'r') as original_file:
        with open(PREDICTIONS_DUMMY_FILE, 'w') as new_file:
            reader = csv.reader(original_file)
            writer = csv.writer(new_file)
            next(reader, None)
            for line in reader:
                new_line = line
                new_sentence = generate_dummy_output_for_line(new_line[-1])
                new_line[1] = new_sentence
                writer.writerow(new_line)

generate_output()