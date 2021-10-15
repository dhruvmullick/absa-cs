from typing import Optional
import pandas as pd
import numpy as np
import xml.etree.ElementTree as ET
# from sklearn.model_selection import train_test_split

def parse_semeval_xml(root):
    data = []
    for i, child in enumerate(root):
        review_id = child.attrib['rid']

        # Target, Polarity
        prev_opinion = (None, None)
        for sentence in child.findall('sentences/sentence'):
            sentence_id = "_".join(['id', sentence.attrib['id'].split(':')[1]])
            sentence_text = sentence.findall('text')[0].text.strip()
            if len(sentence.findall('text')) != 1:
                raise Exception("NOT EXACTLY ONE TEXT")
            if len(sentence.findall('Opinions')) > 1:
                raise Exception("MORE THAN ONE OPINIONS")
            if len(sentence.findall('Opinions')) == 0:
                print("Not exactly one opinion found for sentence id: {} for sentence: {}".format(str(sentence_id), sentence_text))
                continue
            opinions = []
            sentence_opinions = sentence.findall('Opinions')[0]
            for opinion in sentence_opinions.findall('Opinion'):
                opinions.append([opinion.attrib['target'], opinion.attrib['polarity'], opinion.attrib['from'], opinion.attrib['to']])
                # opinion_target = opinion.attrib['target']
                # opinion_polarity = opinion.attrib['polarity']
                # if opinion_target == prev_opinion[0] and opinion_polarity == prev_opinion[1]:
                #     continue
                # opinions += [ opinion_target + ' ' + opinion_polarity]
                # prev_opinion = (opinion_target, opinion_polarity)
            
            opinions = pd.DataFrame(opinions, columns=['tar', 'pol', 'from', 'to'])
            if 'The wine list is interesting and has' in sentence_text:
                print(opinions)
            opinions.drop_duplicates(subset=['tar', 'pol', 'from', 'to'], inplace=True)
            opinions = [row['tar'] + ' ' + row['pol'] for i, row in opinions.iterrows()]
            if 'The wine list is interesting and has' in sentence_text:
                print(opinions)
                exit(1)

            data.append([review_id, sentence_id, sentence_text, opinions])

    data = pd.DataFrame(data, columns=['review_id', 'sentences_ids', 'sentences_texts', 'sentences_opinions'])
    return data

def load_semeval():
    # xml_train = open('/home/bghanem/projects/ABSA_LM/data/semeval/training/ABSA16_Restaurants_Train_SB1_v2.xml', 'r').read()  # Read file
    xml_train = open('data/semeval/training/ABSA16_Restaurants_Train_SB1_v2.xml', 'r').read()  # Read file
    train = parse_semeval_xml(root=ET.XML(xml_train))
    train['sentences_opinions'] = train['sentences_opinions'].map(opinions_to_decoder_format)

    xml_test = open('data/semeval/test/EN_REST_SB1_TEST.xml.gold', 'r').read()  # Read file
    test = parse_semeval_xml(root=ET.XML(xml_test))
    test['sentences_opinions'] = test['sentences_opinions'].map(opinions_to_decoder_format)

    # Remove null aspects
    train = train[~train['sentences_opinions'].str.contains('NULL')]
    test = test[~test['sentences_opinions'].str.contains('NULL')]

    print('Semeval: ', train.shape, test.shape)
    return train, test

def parse_MAMS_xml(root):
    data = []
    for i, child in enumerate(root):
        sentences_text = child.findall('text')[0].text.strip()
        sentences_opinions = [[aspect_term.attrib['term']+' '+aspect_term.attrib['polarity'] for aspect_term in aspect_terms.getchildren()]
                    for aspect_terms in child.findall('aspectTerms')]
        for opinion in sentences_opinions:
            data.append([0, 0, sentences_text, opinion])
    data = pd.DataFrame(data, columns=['review_id', 'sentences_ids', 'sentences_texts', 'sentences_opinions'])
    return data

def load_MAMS():
    xml_train = open('data/MAMS_ATSA/train.xml', 'r').read()  # Read file
    train = parse_MAMS_xml(root=ET.XML(xml_train))
    train['sentences_opinions'] = train['sentences_opinions'].map(opinions_to_decoder_format)
    
    xml_val = open('data/MAMS_ATSA/val.xml', 'r').read()  # Read file
    val = parse_MAMS_xml(root=ET.XML(xml_val))
    val['sentences_opinions'] = val['sentences_opinions'].map(opinions_to_decoder_format)

    xml_test = open('data/MAMS_ATSA/test.xml', 'r').read()  # Read file
    test = parse_MAMS_xml(root=ET.XML(xml_test))
    test['sentences_opinions'] = test['sentences_opinions'].map(opinions_to_decoder_format)

    print('MAMS: ', train.shape, val.shape, test.shape)
    return train, val, test

def opinions_to_decoder_format(opinions_list):
    return ' <sep> '.join(opinions_list)

if __name__ == '__main__':
    sem_train, sem_test = load_semeval()
    mams_train, mams_val, mams_test = load_MAMS()

    train = pd.concat([sem_train], axis=0).sample(frac=1, random_state=0).reset_index(drop=True)
    # train, val = train_test_split(train, test_size=0.2)
    test = pd.concat([sem_test], axis=0).sample(frac=1, random_state=0).reset_index(drop=True)
    test = test[test['sentences_opinions'] != '']

    train.to_csv('data/processed_train_rest.csv', header=True, index=False)
    # val.to_csv('data/processed_val_rest.csv', header=True, index=False)
    test.to_csv('data/processed_test_rest.csv', header=True, index=False)

    train = pd.concat([mams_train], axis=0).sample(frac=1, random_state=0).reset_index(drop=True)
    val = pd.concat([mams_val], axis=0).sample(frac=1, random_state=0).reset_index(drop=True)
    test = pd.concat([mams_test], axis=0).sample(frac=1, random_state=0).reset_index(drop=True)
    test = test[test['sentences_opinions'] != '']

    train.to_csv('data/processed_train_mams.csv', header=True, index=False)
    val.to_csv('data/processed_val_mams.csv', header=True, index=False)
    test.to_csv('data/processed_test_mams.csv', header=True, index=False)

    print('saved..')