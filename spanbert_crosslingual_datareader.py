import csv
import pandas as pd
import random

random.seed(0)

from data_reader import join_sentence_and_annotations

TRANSLATED_FILE = "data/processed_{}_{}_to_{}_processed.csv"
TRANSLATED_FILE_SPANBERT = "data/{}_spanbert_{}_to_{}.csv"

ORIGINAL_FILE_SPANBERT = "data/{}_spanbert_{}.csv"
MERGED_FILE_SPANBERT = "data/{}_spanbert_{}_merged.csv"


def get_annotations_for_sentence(sentence_text, opinions_info):
    words = sentence_text.split()
    annotations = ['O'] * len(words)
    for opinion in opinions_info:
        annotation = 'O'
        if opinion['pol'] == 'positive':
            annotation = 'T-POS'
        if opinion['pol'] == 'negative':
            annotation = 'T-NEG'
        if opinion['pol'] == 'neutral':
            annotation = 'T-NEU'
        if opinion['pol'] == 'conflict':
            continue
        word_count = 0
        for j in range(len(sentence_text)):
            if sentence_text[j].isspace() and j > 0 and not sentence_text[j - 1].isspace():
                word_count += 1
            if int(opinion['from']) <= j < int(opinion['to']):
                annotations[word_count] = annotation
            if j >= int(opinion['to']):
                break
    return annotations, words

def get_spanbert_opinions(sentence, sentences_opinions, opinions_positions):
    words = sentence.split(" ")
    opinions_info = []
    for i in range(len(sentences_opinions)):
        opinion = sentences_opinions[i]
        pos = opinions_positions[i]
        polarity = opinion.split(" ")[-1].strip()
        pos_start = pos.split("-")[0].strip()
        pos_end = pos.split("-")[1].strip()
        opinions_info.append({'opinion':opinion, 'pol':polarity, 'from':pos_start, 'to':pos_end})

    annotations, words = get_annotations_for_sentence(sentence, opinions_info)
    spanbert_text = join_sentence_and_annotations(words, annotations)
    return spanbert_text

def process_translated_dataset(translated_file_path, translated_spanbert_file_path):
    with open(translated_file_path, 'r') as translated_file:
        reader = csv.reader(translated_file)
        next(reader, None)
        with open(translated_spanbert_file_path, 'w') as spanbert_file:
            writer = csv.writer(spanbert_file)
            for line in reader:
                original_text = line[2].strip()
                # if original_text[0] == '\"':
                #     original_text = original_text[1:]
                # if original_text[-1] == '\"':
                #     original_text = original_text[:-1]
                sentences_opinions = [y.strip() for y in line[3].split("<sep>")]
                opinions_positions = [y.strip() for y in line[4].split("<sep>")]
                spanbert_opinions = get_spanbert_opinions(original_text, sentences_opinions, opinions_positions)
                writer.writerow(['{}####{}'.format(original_text, spanbert_opinions)])

    # reader = pd.read_csv(translated_file_path)
    # # reader = csv.reader(translated_file)
    # # next(reader, None)
    # with open(translated_spanbert_file_path, 'w') as spanbert_file:
    #     writer = csv.writer(spanbert_file)
    #     for line in reader:
    #         original_text = line[2].strip()
    #         if original_text[0] == '\"':
    #             original_text = original_text[1:]
    #         if original_text[-1] == '\"':
    #             original_text = original_text[:-1]
    #         sentences_opinions = [y.strip() for y in line[3].split("<sep>")]
    #         opinions_positions = [y.strip() for y in line[4].split("<sep>")]
    #         spanbert_opinions = get_spanbert_opinions(original_text, sentences_opinions, opinions_positions)
    #         writer.writerow(['{}####{}'.format(original_text, spanbert_opinions)])

cross_datasets = ['Rest16_en', 'Rest16_es', 'Rest16_ru', 'Lap14_en']
dataset_types = ['train', 'val']
languages = ['en', 'es', 'ru']

for dtrain in cross_datasets:
    for dataset_type in dataset_types:
        original_language = dtrain.split("_")[-1]
        for lang in languages:
            if lang != original_language:
                process_translated_dataset(TRANSLATED_FILE.format(dataset_type, dtrain, lang),
                                           TRANSLATED_FILE_SPANBERT.format(dataset_type, dtrain, lang))

for dtrain in cross_datasets:
    for dataset_type in dataset_types:
        original_language = dtrain.split("_")[-1]
        filepaths = []
        for lang in languages:
            if lang != original_language:
                filepaths.append(TRANSLATED_FILE_SPANBERT.format(dataset_type, dtrain, lang))
            else:
                filepaths.append(ORIGINAL_FILE_SPANBERT.format(dataset_type, dtrain))

        rows = []
        for path in filepaths:
            with open(path, 'r') as file:
                for row in csv.reader(file):
                    # if row[0] == '\"':
                    #     row = row[1:]
                    # if row[-1] == '\"':
                    #     row = row[:-1]
                    rows.append(', '.join(row))

        random.shuffle(rows)

        with open(MERGED_FILE_SPANBERT.format(dataset_type, dtrain), 'w') as file:
            # writer = csv.writer(file)
            # writer.writerows(rows)
            for row in rows:
                file.write(row)
                file.write('\n')