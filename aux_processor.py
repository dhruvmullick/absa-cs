import json
import random
import pandas as pd
from torch.utils.data import DataLoader

from sklearn.model_selection import train_test_split
from datasets import load_dataset

TARGET_TEXT = "target"
SOURCE_TEXT = "source"
OTHER = "other"

ABSA = 'ABSA'
SQUAD = 'SQUAD'
COSMOS = 'COSMOS'
WIKITEXT = 'WIKITEXT'
COMMONGEN = 'COMMONGEN'

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
    val_df = val_df.sample(frac=0.5)
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
        all_answers = [x[0] for x in batch['answers']['text']]
        data.append([question_context, batch['answers']['text'][0][0], all_answers])
    df = pd.DataFrame(data, columns=['question_context', 'correct_answer', 'all_answers'])
    return df


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

        text_list = text.split('.')
        prefix_sentence_count = min(len(text_list), 2)
        text = '.'.join(text_list[:prefix_sentence_count])

        if len(text.split()) <= 5:
            continue

        masked_word = ''
        ctr = 0
        rand_idx = 1
        while (len(masked_word) <= 3 or masked_word == '[UNK]') and ctr < 20:
            rand_idx = random.randint(5, len(text.split())-1)
            masked_word = text.split()[rand_idx]
            ctr += 1
        if ctr == 20:
            continue

        text = ' '.join(text.split()[:rand_idx])
        data.append([f'Get next words: {text}', masked_word])

    return pd.DataFrame(data, columns=['sentence', 'masked_word'])


def extract_df_with_correct_answer(cosmos_data_df):
    data = []
    for idx, row in cosmos_data_df.iterrows():
        question_context_candidates = 'question: {} answer_0: {} answer_1: {} answer_2: {} answer_3: {} context: {}' \
            .format(row['question'], row['answer0'], row['answer1'], row['answer2'], row['answer3'], row['context'])
        data.append([question_context_candidates, row['label']])
    return pd.DataFrame(data, columns=['question_context_candidate', 'correct_answer'])


def get_renamed_commongen_columns(df):
    df = df.rename(columns={"scene": TARGET_TEXT, "concept_set": SOURCE_TEXT})[[TARGET_TEXT, SOURCE_TEXT]]
    return df


def get_renamed_squad_columns(df):
    df = df.rename(columns={"correct_answer": TARGET_TEXT, "question_context": SOURCE_TEXT,
                            "all_answers": OTHER})[[TARGET_TEXT, SOURCE_TEXT, OTHER]]
    return df


def get_renamed_cosmos_columns(df):
    df = df.rename(columns={"correct_answer": TARGET_TEXT, "question_context_candidate": SOURCE_TEXT})[
        [TARGET_TEXT, SOURCE_TEXT]]
    return df


def get_renamed_absa_columns(df):
    df = df.rename(columns={"sentences_opinions": TARGET_TEXT, "sentences_texts": SOURCE_TEXT})[
        [TARGET_TEXT, SOURCE_TEXT]]
    return df


def get_renamed_lm_columns(df):
    df = df.rename(columns={"masked_word": TARGET_TEXT, "sentence": SOURCE_TEXT})[
        [TARGET_TEXT, SOURCE_TEXT]]
    return df


def evaluate_squad_predictions(predictions_filepath_validation):
    df = pd.read_csv(predictions_filepath_validation)
    hit = 0
    for idx, row in df.iterrows():
        prediction = row["Generated Text"]
        actual = eval(row['other'])
        if prediction in actual:
            hit += 1
    return 100 * hit / len(df)


def evaluate_lm_one_predictions(predictions_filepath_validation):
    df = pd.read_csv(predictions_filepath_validation)
    hit = 0
    for idx, row in df.iterrows():
        prediction = str(row["Generated Text"]).split(' ')[0]
        actual = row["Actual Text"]
        if prediction == actual:
            hit += 1
    return 100 * hit / len(df)