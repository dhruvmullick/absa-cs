import json
import random
import pandas as pd
from torch.utils.data import DataLoader

from sklearn.model_selection import train_test_split
from datasets import load_dataset

TARGET_TEXT = "target"
SOURCE_TEXT = "source"

### Prompt taken from own_commongen paper https://aclanthology.org/2020.findings-emnlp.165.pdf
COMMONGEN_PROMPT = 'generate a sentence with: '

def process_concepts(concept_set):
    return concept_set.replace("#", " ")


def read_commongen_json_file(file_name):
    concept_sets = []
    scenes = []
    with open(file_name, 'r') as file:
        for line in file:
            json_line = json.loads(line)
            concept_set = process_concepts(json_line['concept_set'])
            concept_set = COMMONGEN_PROMPT + " " + concept_set
            for scene in json_line['scene']:
                concept_sets.append(concept_set)
                scenes.append(scene)
    return pd.DataFrame(data={'scene': scenes, 'concept_set': concept_sets})


def read_commongen_data(seed):
    training_file_commongen = './data/commongen_data/commongen.train.jsonl'
    test_file_commongen = './data/commongen_data/commongen.dev.jsonl'
    training_data_commongen = read_commongen_json_file(training_file_commongen)
    testing_data_commongen = read_commongen_json_file(test_file_commongen)
    training_data_commongen, validation_data_commongen = train_test_split(training_data_commongen, test_size=0.1,
                                                                          random_state=seed)
    return training_data_commongen, validation_data_commongen, testing_data_commongen


def read_squad_data():
    print("Loading Squad data...")
    dataset = load_dataset("squad", keep_in_memory=True)
    train, val, _ = dataset["train"], dataset["validation"], None
    train_df = extract_train_df_from_dataset_for_squad(train)
    val_df = extract_train_df_from_dataset_for_squad(val)
    return train_df, val_df, None


def read_wikitext_data():
    print("Loading WikiText data...")
    dataset = load_dataset("wikitext", "wikitext-2-v1", keep_in_memory=True)
    train, val, _ = dataset["train"], dataset["validation"], None
    train_df = extract_train_df_from_dataset_for_wikitext(train)
    val_df = extract_train_df_from_dataset_for_wikitext(val)
    return train_df, val_df, None


def read_cosmos_data():
    training_file_cosmos = './data/cosmosqa/train.csv'
    val_file_cosmos = './data/cosmosqa/valid.csv'
    training_data_df = pd.read_csv(training_file_cosmos)
    val_data_df = pd.read_csv(val_file_cosmos)
    extracted_training_data_df = extract_df_with_correct_answer(training_data_df)
    extracted_val_data_df = extract_df_with_correct_answer(val_data_df)
    return extracted_training_data_df, extracted_val_data_df, None


# def read_wsc_data():
#     print("Loading WSC Coreference Resolution data...")
#     dataset = load_dataset("winograd_wsc", "wsc-", keep_in_memory=True)
#     train, val, _ = dataset["train"], dataset["validation"], None
#     train_df = extract_train_df_from_dataset_for_wikitext(train)
#     val_df = extract_train_df_from_dataset_for_wikitext(val)
#     return train_df, val_df, None


def extract_train_df_from_dataset_for_squad(dataset):
    ### Using same input style as T5 paper
    params = {'batch_size': 1, 'shuffle': False, 'num_workers': 2}
    loader = DataLoader(dataset, **params)
    data = []
    for batch in loader:
        question_context = 'question: {} context: {}'.format(batch['question'][0], batch['context'][0])
        data.append([question_context, batch['answers']['text'][0][0]])
    return pd.DataFrame(data, columns=['question_context', 'correct_answer'])


def extract_train_df_from_dataset_for_wikitext(dataset):
    params = {'batch_size': 1, 'shuffle': False, 'num_workers': 2}
    loader = DataLoader(dataset, **params)
    data = []
    for batch in loader:
        text = batch['text'][0].strip()
        ### Cleaning: https://github.com/mauriw/deep_zip/blob/5623d8c8532c655e95b3f4583ae4cd8e011f6b4c/data.py
        if len(text) < 10 or len(text.split()) < 5:
            continue
        if '=' in text:
            continue
        text = text.replace('<unk>', '[UNK]')
        text = text.replace(' @-@ ', '-')
        text = text.replace(' @,@ ', ',')
        text = text.replace(' @.@ ', '.')

        masked_word = ''
        ctr = 0
        while len(masked_word) < 3 and ctr < 10:
            rand_idx = random.randrange(len(text.split()))
            masked_word = text.split()[rand_idx]
            ctr += 1

        if ctr == 10:
            continue
        text = ' '.join(text.split()[:rand_idx])
        data.append([text, masked_word])
    return pd.DataFrame(data, columns=['sentence', 'masked_word'])


def extract_df_with_correct_answer(cosmos_data_df):
    data = []
    for idx, row in cosmos_data_df.iterrows():
        correct_answer_column = 'answer' + str(row['label'])
        question_context = 'question: {} context: {}'.format(row['question'], row['context'])
        data.append([question_context, row[correct_answer_column]])
    return pd.DataFrame(data, columns=['question_context', 'correct_answer'])


def get_renamed_commongen_columns(df):
    df = df.rename(columns={"scene": TARGET_TEXT, "concept_set": SOURCE_TEXT})[[TARGET_TEXT, SOURCE_TEXT]]
    return df


### For Squad and Cosmos
def get_renamed_qa_columns(df):
    df = df.rename(columns={"correct_answer": TARGET_TEXT, "question_context": "source"})[[TARGET_TEXT, SOURCE_TEXT]]
    return df


def get_renamed_absa_columns(df):
    df = df.rename(columns={"sentences_opinions": TARGET_TEXT, "sentences_texts": SOURCE_TEXT})[
        [TARGET_TEXT, SOURCE_TEXT]]
    return df


def get_renamed_lm_columns(df):
    df = df.rename(columns={"sentence": TARGET_TEXT, "masked_word": SOURCE_TEXT})[
        [TARGET_TEXT, SOURCE_TEXT]]
    return df
