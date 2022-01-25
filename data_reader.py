import pandas as pd
import numpy as np
import xml.etree.ElementTree as ET
from sklearn.model_selection import train_test_split
import random

WEIRD_CHARACTERS = '.!?,'

#### Ambiguous Dataset3
# AMBIGUOUS_CASES = [" it ", " its ", " he ", " him ", " his ", " she ", " her ", " hers ",
#                    " they ", " them ", " we ", " us ", " and ", " or ", ","]

#### Ambiguous Dataset2
# AMBIGUOUS_CASES = [" it ", " its ", " he ", " him ", " his ", " she ", " her ", " hers ",
#                    " they ", " them ", " we ", " us "]

### For Manual creation
AMBIGUOUS_CASES = [" it ", " its ", " he ", " him ", " his ", " she ", " her ", " hers ",
                   " they ", " them "]


REGEX_PHRASE = '|'.join(AMBIGUOUS_CASES)

random.seed(0)


def write_to_file(array, path):
    with open(path, 'w') as file:
        file.write('\n'.join(array) + '\n')


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
                continue
            opinions = []
            sentence_opinions = sentence.findall('Opinions')[0]
            for opinion in sentence_opinions.findall('Opinion'):
                if opinion.attrib['polarity'] == 'conflict':
                    continue
                opinions.append([opinion.attrib['target'], opinion.attrib['polarity'], opinion.attrib['from'],
                                 opinion.attrib['to']])

            if len(opinions) == 0:
                continue

            opinions = pd.DataFrame(opinions, columns=['tar', 'pol', 'from', 'to'])
            opinions.drop_duplicates(subset=['tar', 'pol', 'from', 'to'], inplace=True)

            opinions = [row['tar'] + ' ' + row['pol'] for i, row in opinions.iterrows()]

            data.append([review_id, sentence_id, sentence_text, opinions])

    data = pd.DataFrame(data, columns=['review_id', 'sentences_ids', 'sentences_texts', 'sentences_opinions'])
    return data


def parse_semeval_14_xml(root):
    data = []
    datalist_span_bert = []
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
            if polarity == 'conflict':
                continue
            from_idx = aspectTerm.attrib['from']
            to_idx = aspectTerm.attrib['to']
            term = term.strip()
            polarity = polarity.strip()

            if term and polarity:
                opinions.append([term, polarity, from_idx, to_idx])

        if len(opinions) == 0:
            continue

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
    non_null_train_idx = ~train['sentences_opinions'].str.contains('NULL')
    train = train[non_null_train_idx]
    non_null_test_idx = ~test['sentences_opinions'].str.contains('NULL')
    test = test[non_null_test_idx]

    print('Semeval: ', train.shape, test.shape)

    train, val = train_test_split(train, test_size=0.1, random_state=0)
    return train, val, test


def load_semeval_14(train_file, test_file):
    xml_train = open(train_file, 'r').read()  # Read file
    train = parse_semeval_14_xml(root=ET.XML(xml_train))
    train['sentences_opinions'] = train['sentences_opinions'].map(opinions_to_decoder_format)

    xml_test = open(test_file, 'r').read()  # Read file
    test = parse_semeval_14_xml(root=ET.XML(xml_test))
    test['sentences_opinions'] = test['sentences_opinions'].map(opinions_to_decoder_format)

    # Remove null aspects
    non_null_train_idx = ~train['sentences_opinions'].str.contains('NULL')
    train = train[non_null_train_idx]
    non_null_test_idx = ~test['sentences_opinions'].str.contains('NULL')
    test = test[non_null_test_idx]

    print('Semeval: ', train.shape, test.shape)

    train, val = train_test_split(train, test_size=0.1, random_state=0)
    return train, val, test


def parse_MAMS_xml(root, shortened):
    data = []
    for i, child in enumerate(root):
        sentence_text = child.findall('text')[0].text.strip()

        opinions = []

        for aspectTerm in child.findall('aspectTerms')[0]:
            term = aspectTerm.attrib['term']
            polarity = aspectTerm.attrib['polarity']
            from_idx = aspectTerm.attrib['from']
            to_idx = aspectTerm.attrib['to']
            term = term.strip()
            polarity = polarity.strip()
            if polarity == 'conflict':
                continue
            if term and polarity:
                opinions.append([term, polarity, from_idx, to_idx])

        opinions = pd.DataFrame(opinions, columns=['tar', 'pol', 'from', 'to'])
        opinions.drop_duplicates(subset=['tar', 'pol', 'from', 'to'], inplace=True)

        if len(opinions) == 0:
            continue

        opinions = [row['tar'] + ' ' + row['pol'] for i, row in opinions.iterrows()]
        data.append([0, 0, sentence_text, opinions])

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
    ('Rest15', 'en'): (load_semeval, {'train_file': 'data/semeval-2015/ABSA-15_Restaurants_Train_Final.xml',
                                      'test_file': 'data/semeval-2015/ABSA15_Restaurants_Test.xml'}),
    ('Rest14', 'en'): (load_semeval_14, {'train_file': 'data/semeval-2014/Restaurants_Train.xml',
                                         'test_file': 'data/semeval-2014/Restaurants_Test_Gold.xml'}),
    ('Lap14', 'en'): (load_semeval_14, {'train_file': 'data/semeval-2014/Laptops_Train.xml',
                                        'test_file': 'data/semeval-2014/Laptops_Test_Gold.xml'}),
    ('Mams', 'en'): (load_MAMS, {'train_file': 'data/MAMS_ATSA/train.xml', 'val_file': 'data/MAMS_ATSA/val.xml',
                                 'test_file': 'data/MAMS_ATSA/test.xml', 'shortened': False}),
    ('Lap16', 'en'): (load_semeval, {'train_file': 'data/semeval/training/ABSA16_Restaurants_Train_SB1_v2.xml',
                                     'test_file': 'data/semeval/test/EN_LAPT_SB1_TEST_.xml.gold'})
}


def preprocess_dataset(domain, language):
    print("Processing the dataset for {}.{}".format(domain, language))
    if not load_dataset.get((domain, language)):
        raise Exception("domain language combination not defined")
    method = load_dataset[(domain, language)][0]
    args = load_dataset[(domain, language)][1]
    train, val, test = method(**args)

    rows = random.sample(np.arange(0, len(train)).tolist(), len(train))
    train = train.iloc[rows, :]

    test_idx = test['sentences_opinions'] != ''
    test = test[test_idx]

    # train_ambiguous = train[train['sentences_texts'].str.contains(REGEX_PHRASE)]
    # val_ambiguous = val[val['sentences_texts'].str.contains(REGEX_PHRASE)]
    test_ambiguous = test[test['sentences_texts'].str.contains(REGEX_PHRASE)]

    train.to_csv('data/processed_train_{}_{}.csv'.format(domain, language), header=True, index=False)
    val.to_csv('data/processed_val_{}_{}.csv'.format(domain, language), header=True, index=False)
    test.to_csv('data/processed_test_{}_{}.csv'.format(domain, language), header=True, index=False)

    return train, val, test, test_ambiguous


if __name__ == '__main__':
    # Semeval Rest 2016
    rest16_train, rest16_val, rest16_test, rest16_test_ambi = preprocess_dataset('Rest16', 'en')
    # Semeval Rest 2015
    rest15_train, rest15_val, rest15_test, rest15_test_ambi = preprocess_dataset('Rest15', 'en')
    # MAMS
    mams_train, mams_val, mams_test, mams_test_ambi = preprocess_dataset('Mams', 'en')
    # Semeval Laptop 2014
    lap14_train, lap14_val, lap14_test, lap14_test_ambi = preprocess_dataset('Lap14', 'en')
    # Semeval Rest 2014
    rest14_train, rest14_val, rest14_test, rest14_test_ambi = preprocess_dataset('Rest14', 'en')

    ### Get the right number of records from all datasets as per their distribution in the test set.
    # Rest16 has smallest training and val sets. Use all of them.

    # Training set
    rest16_train_count = len(rest16_train)
    rest16_train = rest16_train.sample(n=rest16_train_count)
    mams_train_count = int(rest16_train_count * len(mams_test_ambi) / len(rest16_test_ambi))
    mams_train = mams_train.sample(n=mams_train_count)
    lap14_train_count = int(rest16_train_count * len(lap14_test_ambi) / len(rest16_test_ambi))
    lap14_train = lap14_train.sample(n=min(lap14_train_count, len(lap14_train)))

    # Validation set
    rest16_val_count = len(rest16_val)
    rest16_val = rest16_val.sample(n=rest16_val_count)
    mams_val_count = int(rest16_val_count * len(mams_test_ambi) / len(rest16_test_ambi))
    mams_val = mams_val.sample(n=mams_val_count)
    lap14_val_count = int(rest16_val_count * len(lap14_test_ambi) / len(rest16_test_ambi))
    lap14_val = lap14_val.sample(n=min(lap14_val_count, len(lap14_val)))

    ### Merged train datasets
    train_merged = pd.concat([rest16_train, mams_train, lap14_train], ignore_index=True)
    train_merged.to_csv('data/merged_train.csv', header=True, index=False)

    ### Merged validation datasets
    val_merged = pd.concat([rest16_val, mams_val, lap14_val], ignore_index=True)
    val_merged.to_csv('data/merged_val.csv', header=True, index=False)

    ### Merged ambiguous test dataset
    test_ambiguous = pd.concat([rest16_test_ambi, mams_test_ambi, lap14_test_ambi, rest15_test_ambi, rest14_test_ambi], ignore_index=True)
    test_ambiguous.to_csv('data/merged_test_ambiguous.csv', header=True, index=False)

    print('saved..')
