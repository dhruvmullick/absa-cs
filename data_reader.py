from typing import Optional
import pandas as pd
import numpy as np
import xml.etree.ElementTree as ET
from sklearn.model_selection import train_test_split

MAMS_SHORTENED_TEXT_LENGTH = 125
#lowest possible training record count in the considered datasets.
SAMPLED_RECORD_COUNT = 1023

def parse_semeval_xml(root):
    data = []
    for i, child in enumerate(root):
        review_id = child.attrib['rid']

        for sentence in child.findall('sentences/sentence'):
            sentence_id = "_".join(['id', sentence.attrib['id'].split(':')[1]])
            if not sentence.findall('text')[0].text:
                continue
            sentence_text = sentence.findall('text')[0].text.strip()
            if len(sentence.findall('text')) != 1:
                raise Exception("NOT EXACTLY ONE TEXT")
            if len(sentence.findall('Opinions')) > 1:
                raise Exception("MORE THAN ONE OPINIONS")
            if len(sentence.findall('Opinions')) == 0:
                # print("Not exactly one opinion found for sentence id: {} for sentence: {}".format(str(sentence_id),
                #                                                                                   sentence_text))
                continue
            opinions = []
            sentence_opinions = sentence.findall('Opinions')[0]
            for opinion in sentence_opinions.findall('Opinion'):
                opinions.append([opinion.attrib['target'], opinion.attrib['polarity'], opinion.attrib['from'],
                                 opinion.attrib['to']])

            opinions = pd.DataFrame(opinions, columns=['tar', 'pol', 'from', 'to'])
            opinions.drop_duplicates(subset=['tar', 'pol', 'from', 'to'], inplace=True)
            opinions = [row['tar'] + ' ' + row['pol'] for i, row in opinions.iterrows()]

            data.append([review_id, sentence_id, sentence_text, opinions])

    data = pd.DataFrame(data, columns=['review_id', 'sentences_ids', 'sentences_texts', 'sentences_opinions'])
    return data


def parse_semeval_14_xml(root):
    data = []
    for i, child in enumerate(root):
        review_id = i
        sentence_id = child.attrib['id']
        sentence_text = child.find('text').text
        aspect_term_node = child.find('aspectTerms')
        if not aspect_term_node:
            continue

        opinions = []

        for aspectTerm in aspect_term_node:
            term = aspectTerm.attrib['term']
            polarity = aspectTerm.attrib['polarity']
            from_idx = aspectTerm.attrib['from']
            to_idx = aspectTerm.attrib['to']
            term = term.strip()
            polarity = polarity.strip()

            if term and polarity:
                opinions.append([term, polarity, from_idx, to_idx])

        opinions = pd.DataFrame(opinions, columns=['tar', 'pol', 'from', 'to'])
        opinions.drop_duplicates(subset=['tar', 'pol', 'from', 'to'], inplace=True)
        opinions = [row['tar'] + ' ' + row['pol'] for i, row in opinions.iterrows()]
        data.append([review_id, sentence_id, sentence_text, opinions])

    data = pd.DataFrame(data, columns=['review_id', 'sentences_ids', 'sentences_texts', 'sentences_opinions'])
    return data


def load_semeval(train_file, test_file):
    xml_train = open(train_file, 'r').read()  # Read file
    train = parse_semeval_xml(root=ET.XML(xml_train))
    train['sentences_opinions'] = train['sentences_opinions'].map(opinions_to_decoder_format)

    xml_test = open(test_file, 'r').read()  # Read file
    test = parse_semeval_xml(root=ET.XML(xml_test))
    test['sentences_opinions'] = test['sentences_opinions'].map(opinions_to_decoder_format)

    # Remove null aspects
    train = train[~train['sentences_opinions'].str.contains('NULL')]
    test = test[~test['sentences_opinions'].str.contains('NULL')]

    print('Semeval: ', train.shape, test.shape)

    train = pd.concat([train], axis=0).sample(frac=1, random_state=0).reset_index(drop=True)
    train, val = train_test_split(train, test_size=0.1, random_state=0)
    return train, val, test


def load_semeval_14(train_file, test_file):
    xml_train = open(train_file, 'r').read()  # Read file
    train_file = parse_semeval_14_xml(root=ET.XML(xml_train))
    train_file['sentences_opinions'] = train_file['sentences_opinions'].map(opinions_to_decoder_format)

    xml_test = open(test_file, 'r').read()  # Read file
    test_file = parse_semeval_14_xml(root=ET.XML(xml_test))
    test_file['sentences_opinions'] = test_file['sentences_opinions'].map(opinions_to_decoder_format)

    # Remove null aspects
    train_file = train_file[~train_file['sentences_opinions'].str.contains('NULL')]
    test_file = test_file[~test_file['sentences_opinions'].str.contains('NULL')]

    print('Semeval: ', train_file.shape, test_file.shape)

    train_file = pd.concat([train_file], axis=0).sample(frac=1, random_state=0).reset_index(drop=True)
    train_file, val_file = train_test_split(train_file, test_size=0.1, random_state=0)
    return train_file, val_file, test_file


def parse_MAMS_xml(root, shortened):
    data = []
    for i, child in enumerate(root):
        sentences_text = child.findall('text')[0].text.strip()
        if shortened and (len(sentences_text) > MAMS_SHORTENED_TEXT_LENGTH):
            continue
        sentences_opinions = [[aspect_term.attrib['term'] + ' ' + aspect_term.attrib['polarity'] for aspect_term in
                               aspect_terms.getchildren()]
                              for aspect_terms in child.findall('aspectTerms')]
        for opinion in sentences_opinions:
            data.append([0, 0, sentences_text, opinion])
    data = pd.DataFrame(data, columns=['review_id', 'sentences_ids', 'sentences_texts', 'sentences_opinions'])
    return data


def load_MAMS(train_file, val_file, test_file, shortened):
    xml_train = open(train_file, 'r').read()  # Read file
    train = parse_MAMS_xml(root=ET.XML(xml_train), shortened=shortened)
    train['sentences_opinions'] = train['sentences_opinions'].map(opinions_to_decoder_format)

    xml_val = open(val_file, 'r').read()  # Read file
    val = parse_MAMS_xml(root=ET.XML(xml_val), shortened=shortened)
    val['sentences_opinions'] = val['sentences_opinions'].map(opinions_to_decoder_format)

    xml_test = open(test_file, 'r').read()  # Read file
    test = parse_MAMS_xml(root=ET.XML(xml_test), shortened=shortened)
    test['sentences_opinions'] = test['sentences_opinions'].map(opinions_to_decoder_format)

    print('MAMS: ', train.shape, val.shape, test.shape)
    return train, val, test


def opinions_to_decoder_format(opinions_list):
    return ' <sep> '.join(opinions_list)


load_dataset = {
    ('Rest16', 'en'): (load_semeval, {'train_file': 'data/semeval/training/ABSA16_Restaurants_Train_SB1_v2.xml',
                                      'test_file': 'data/semeval/test/EN_REST_SB1_TEST.xml.gold'}),
    ('Rest16', 'fr'): (load_semeval, {'train_file': 'data/semeval/training/ABSA16FR_Restaurants_Train-withcontent.xml',
                                      'test_file': 'data/semeval/test/ABSA16FR_Restaurants_Gold-withcontent.xml'}),
    ('Rest16', 'nl'): (load_semeval, {'train_file': 'data/semeval/training/restaurants_dutch_training.xml',
                                      'test_file': 'data/semeval/test/DU_REST_SB1_TEST.xml.gold'}),
    ('Rest16', 'es'): (load_semeval, {'train_file': 'data/semeval/training/SemEval-2016ABSA Restaurants-Spanish_Train_Subtask1.xml',
                                      'test_file': 'data/semeval/test/SP_REST_SB1_TEST.xml.gold'}),
    ('Rest16', 'ru'): (load_semeval, {'train_file': 'data/semeval/training/se16_ru_rest_train.xml',
                                      'test_file': 'data/semeval/test/RU_REST_SB1_TEST.xml.gold'}),
    ('Rest15', 'en'): (load_semeval, {'train_file': 'data/semeval-2015/ABSA-15_Restaurants_Train_Final.xml',
                                      'test_file': 'data/semeval-2015/ABSA15_Restaurants_Test.xml'}),
    ('Rest14', 'en'): (load_semeval_14, {'train_file': 'data/semeval-2014/Restaurants_Train.xml',
                                         'test_file': 'data/semeval-2014/Restaurants_Test_Gold.xml'}),
    ('Lap14', 'en'): (load_semeval_14, {'train_file': 'data/semeval-2014/Laptops_Train.xml',
                                        'test_file': 'data/semeval-2014/Laptops_Test_Gold.xml'}),
    ('Mams', 'en'): (load_MAMS, {'train_file': 'data/MAMS_ATSA/train.xml', 'val_file': 'data/MAMS_ATSA/val.xml',
                                 'test_file': 'data/MAMS_ATSA/test.xml', 'shortened': False}),
    ('Mams_short', 'en'): (load_MAMS, {'train_file': 'data/MAMS_ATSA/train.xml', 'val_file': 'data/MAMS_ATSA/val.xml',
                                 'test_file': 'data/MAMS_ATSA/test.xml', 'shortened': True}),
}


def preprocess_dataset(domain, language):
    print("Processing the dataset for {}.{}".format(domain, language))
    if not load_dataset.get((domain, language)):
        raise Exception("domain language combination not defined")
    method = load_dataset[(domain, language)][0]
    args = load_dataset[(domain, language)][1]
    train, val, test = method(**args)
    train = pd.concat([train], axis=0).sample(n=SAMPLED_RECORD_COUNT, random_state=0).reset_index(drop=True)
    test = pd.concat([test], axis=0).sample(frac=1, random_state=0).reset_index(drop=True)
    test = test[test['sentences_opinions'] != '']
    train.to_csv('data/processed_train_{}_{}.csv'.format(domain, language), header=True, index=False)
    val.to_csv('data/processed_val_{}_{}.csv'.format(domain, language), header=True, index=False)
    test.to_csv('data/processed_test_{}_{}.csv'.format(domain, language), header=True, index=False)


if __name__ == '__main__':
    # Semeval Rest 2016
    preprocess_dataset('Rest16', 'en')
    # # Semeval Rest 2016
    # preprocess_dataset('Rest16', 'fr')
    # # Semeval Rest 2016
    # preprocess_dataset('Rest16', 'nl')
    # Semeval Rest 2016
    preprocess_dataset('Rest16', 'es')
    # Semeval Rest 2016
    preprocess_dataset('Rest16', 'ru')
    # MAMS
    preprocess_dataset('Mams', 'en')
    # MAMS_Shortened
    preprocess_dataset('Mams_short', 'en')
    # Semeval Laptop 2014
    preprocess_dataset('Lap14', 'en')
    # # Semeval Rest 2014
    # preprocess_dataset('Rest14', 'en')
    # # Semeval Rest 2015
    # preprocess_dataset('Rest15', 'en')

    print('saved..')
