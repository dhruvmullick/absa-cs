import pandas as pd
import numpy as np
import xml.etree.ElementTree as ET


def parse_semeval_xml(root):
    data = []
    for i, child in enumerate(root):
        review_id = child.attrib['rid']
        sentences_id = ["_".join(['id', sentence.attrib['id'].split(':')[1]]) 
                            for sentence in child.findall('sentences/sentence')] # sentences ids are like 79:0, so I take the second part.
        sentences_text = [sentence_text.text.strip() for sentence_text in child.findall('sentences/sentence/text')]
        sentences_opinions = [[opinion.attrib['target']+' '+opinion.attrib['polarity'] for opinion in opinions.getchildren()]
                    for opinions in child.findall('sentences/sentence/Opinions')]
        for id, txt, opinion in zip(sentences_id, sentences_text, sentences_opinions):
            data.append([review_id, id, txt, opinion])
    data = pd.DataFrame(data, columns=['review_id', 'sentences_ids', 'sentences_texts', 'sentences_opinions'])
    return data

def load_semeval():
    xml_train = open('/home/bghanem/projects/ABSA_LM/data/semeval/training/ABSA16_Restaurants_Train_SB1_v2.xml', 'r').read()  # Read file
    train = parse_semeval_xml(root=ET.XML(xml_train))
    train['sentences_opinions'] = train['sentences_opinions'].map(opinions_to_decoder_format)

    xml_test = open('/home/bghanem/projects/ABSA_LM/data/semeval/test/EN_REST_SB1_TEST.xml.gold', 'r').read()  # Read file
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
    xml_train = open('/home/bghanem/projects/ABSA_LM/data/MAMS_ATSA/train.xml', 'r').read()  # Read file
    train = parse_MAMS_xml(root=ET.XML(xml_train))
    train['sentences_opinions'] = train['sentences_opinions'].map(opinions_to_decoder_format)
    
    xml_val = open('/home/bghanem/projects/ABSA_LM/data/MAMS_ATSA/val.xml', 'r').read()  # Read file
    val = parse_MAMS_xml(root=ET.XML(xml_val))
    val['sentences_opinions'] = val['sentences_opinions'].map(opinions_to_decoder_format)

    xml_test = open('/home/bghanem/projects/ABSA_LM/data/MAMS_ATSA/test.xml', 'r').read()  # Read file
    test = parse_MAMS_xml(root=ET.XML(xml_test))
    test['sentences_opinions'] = test['sentences_opinions'].map(opinions_to_decoder_format)

    print('MAMS: ', train.shape, val.shape, test.shape)
    return train, val, test

def opinions_to_decoder_format(opinions_list):
    return ' <sep> '.join(opinions_list)

if __name__ == '__main__':
    sem_train, sem_test = load_semeval()
    mams_train, val, mams_test = load_MAMS()

    train = pd.concat([sem_train, mams_train], axis=0).sample(frac=1, random_state=0).reset_index(drop=True)
    test = pd.concat([sem_test, mams_test], axis=0).sample(frac=1, random_state=0).reset_index(drop=True)
    test = test[test['sentences_opinions'] != '']
    
    train.to_csv('data/train_combined.csv', header=True, index=False)
    val.to_csv('data/val.csv', header=True, index=False)
    test.to_csv('data/test_combined.csv', header=True, index=False)
    print('saved..')