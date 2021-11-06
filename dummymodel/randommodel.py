# A RANDOMISED MODEL FOR ABSA TASK

import numpy as np
import csv
import random
from nltk.corpus import stopwords

DATASET = 'Mams_short_en'
LANGUAGE = 'english'

stop_words_set = set(stopwords.words(LANGUAGE))
PREDICTIONS_ORIGINAL_FILE = '/Users/dhruvmullick/Projects/GenerativeAspectBasedSentimentAnalysis/generative-predictions/{}/{}_predictions.csv'.format(DATASET, DATASET)
PREDICTIONS_DUMMY_FILE = '{}_predictions_dummy.csv'.format(DATASET)

random.seed(0)

def generate_dummy_output_for_line(sentence, pos_ratio, neg_ratio, neu_ratio):
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
        elif random_number < pos_ratio + neg_ratio:
            polarity_sentence.append('{} negative'.format(word))
        elif random_number < pos_ratio + neg_ratio + neu_ratio:
            polarity_sentence.append('{} neutral'.format(word))
    polarity_sentence = ' <sep> '.join(polarity_sentence)
    return polarity_sentence.strip()


def generate_output(pos_ratio, neg_ratio, neu_ratio):
    with open(PREDICTIONS_ORIGINAL_FILE, 'r') as original_file:
        with open(PREDICTIONS_DUMMY_FILE, 'w') as new_file:
            reader = csv.reader(original_file)
            writer = csv.writer(new_file)
            writer.writerow([" ","Generated Text","Actual Text","Original Sentence"])
            next(reader, None)
            for line in reader:
                new_line = line
                new_sentence = generate_dummy_output_for_line(new_line[-1], pos_ratio, neg_ratio, neu_ratio)
                new_line[1] = new_sentence
                writer.writerow(new_line)


def get_distribution():
    pos, neg, neu, total_words = 0, 0, 0, 0
    with open(PREDICTIONS_ORIGINAL_FILE, 'r') as original_file:
        reader = csv.reader(original_file)
        next(reader, None)
        for line in reader:
            total_words += len(line[-1].split())
            pos += line[-2].count('positive')
            neg += line[-2].count('negative')
            neu += line[-2].count('neutral')
    return pos/total_words, neg/total_words, neu/total_words


pos_ratio, neg_ratio, neu_ratio = get_distribution()
generate_output(pos_ratio, neg_ratio, neu_ratio)