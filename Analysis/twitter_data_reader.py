from re import search
import pandas as pd

FILE = '/data/Twitter/test.raw'
FILE_OUTPUT = '/data/twitter_test_filtered_ambi.csv'

AMBIGUOUS_CASES = [" it ", " its ", " he ", " him ", " his ", " she ", " her ", " hers ",
                   " they ", " them "]
REGEX_PHRASE = '|'.join(AMBIGUOUS_CASES)

sentiment_mappings = {'0': 'neutral', '1': 'positive', '-1': 'negative'}

with open(FILE, 'r') as file:

    ct = -1
    sentence = ''
    aspect = ''
    sentiment = ''
    data = []

    for line in file:
        ct += 1
        if ct % 3 == 0:
            if search(REGEX_PHRASE, line):
                sentence = line
                sentence = sentence.strip()
            else:
                sentence = ''
                sentiment = ''
        if ct % 3 == 1:
            if sentence == '':
                continue
            line = line.strip()
            aspect = line
        if ct % 3 == 2:
            if sentence == '':
                continue
            line = line.strip()
            sentiment = sentiment_mappings[line]
            aspect_pair = '{} {}'.format(aspect, sentiment)
            sentence = sentence.replace('$T$', aspect)
            data.append([0, 0, sentence, aspect_pair])

    data = pd.DataFrame(data, columns=['review_id', 'sentences_ids', 'sentences_texts', 'sentences_opinions'])
    data.to_csv(FILE_OUTPUT, index=False)





