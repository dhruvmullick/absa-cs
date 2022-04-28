import json
import random
from itertools import repeat
import re

import pandas as pd
import nltk
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from datasets import load_dataset

import evaluate_e2e_tbsa
import utils

TARGET_TEXT = "target"
SOURCE_TEXT = "source"
OTHER = "other"

ABSA = 'ABSA'
SQUAD = 'SQUAD'
COSMOS = 'COSMOS'
WIKITEXT = 'WIKITEXT'
COMMONGEN = 'COMMONGEN'
DPR = "DPR"
QQP = "QQP"
WMTFR = "WMTFR"
WMTDE = "WMTDE"
BOOK = "BOOK"
WIKIJUMBLED = 'WIKIJUMBLED'

FR = "fr"
DE = "de"

### Prompt taken from own_commongen paper https://aclanthology.org/2020.findings-emnlp.165.pdf
COMMONGEN_PROMPT = 'generate a sentence with: '

MAX_WMT_TRAIN = 50000
MAX_BOOKS_TRAIN = 50000
MAX_WMT_VAL = 5000
MAX_BOOKS_VAL = 5000

def process_concepts(concept_set):
    return concept_set.replace("#", " ")


def get_prefix_for_wmt_langauge(language):
    if language == FR:
        return "translate English to French:"
    elif language == DE:
        return "translate English to German:"


def get_wmt_code_for_language(language):
    if language == FR:
        return "fr-en"
    elif language == DE:
        return "de-en"


def read_commongen_json_file(file_name):
    concept_sets = []
    scenes = []
    all_scenes = []
    with open(file_name, 'r') as file:
        for line in file:
            json_line = json.loads(line)
            concept_set = process_concepts(json_line['concept_set'])
            concept_set = COMMONGEN_PROMPT + " " + concept_set
            for scene in json_line['scene']:
                concept_sets.append(concept_set)
                scenes.append(scene)
                all_scenes.append(json_line['scene'])
    return pd.DataFrame(data={'scene': scenes, 'concept_set': concept_sets, 'all_scenes': all_scenes})


def read_commongen_data():
    training_file_commongen = './data/commongen_data/commongen.train.jsonl'
    val_file_commongen = './data/commongen_data/commongen.dev.jsonl'
    training_data_commongen = read_commongen_json_file(training_file_commongen)
    val_data_commongen = read_commongen_json_file(val_file_commongen)
    return training_data_commongen, val_data_commongen, None


def read_squad_data():
    print("Loading Squad data...")
    dataset = load_dataset("squad", keep_in_memory=True)
    train, val, _ = dataset["train"], dataset["validation"], None
    train_df = extract_train_df_from_dataset_for_squad(train)
    val_df = extract_train_df_from_dataset_for_squad(val)
    val_df = val_df.sample(frac=0.5)
    return train_df, val_df, None


def read_wikitext_data(seed, jumbled=False):
    print("Loading WikiText data...")
    dataset = load_dataset("wikitext", "wikitext-2-v1", keep_in_memory=True)
    train, val, _ = dataset["train"], dataset["validation"], None
    train_df = extract_df_from_dataset_for_wikitext(train, seed, jumbled)
    val_df = extract_df_from_dataset_for_wikitext(val, seed, jumbled)
    return train_df, val_df, None


def read_qqp_data():
    print("Loading QQP data...")
    dataset = load_dataset("glue", "qqp", keep_in_memory=True)
    train, val, _ = dataset["train"], dataset["validation"], dataset["test"]
    train_df = extract_df_from_dataset_for_qqp(train)
    val_df = extract_df_from_dataset_for_qqp(val)
    train_df = train_df.sample(n=50000)
    val_df = val_df.sample(n=5000)
    return train_df, val_df, None


def read_wmt_data(language):
    print(f"Loading WMT {language} data...")
    train = load_dataset("wmt14", get_wmt_code_for_language(language), keep_in_memory=True, split='train[:1%]')
    val = load_dataset("wmt14", get_wmt_code_for_language(language), keep_in_memory=True, split='validation')

    train_df = extract_df_from_dataset_for_wmt(train, language, max_count=MAX_WMT_TRAIN)
    val_df = extract_df_from_dataset_for_wmt(val, language, max_count=MAX_WMT_VAL)
    return train_df, val_df, None


def read_book_data():
    print(f"Loading book data...")
    train = load_dataset("bookcorpus", keep_in_memory=True, split=f'train[:{MAX_BOOKS_TRAIN}]')
    val = load_dataset("bookcorpus", keep_in_memory=True, split=f'train[{MAX_BOOKS_TRAIN}:{MAX_BOOKS_TRAIN+MAX_BOOKS_VAL}]')

    train_df = extract_df_from_dataset_for_books(train)
    val_df = extract_df_from_dataset_for_books(val)
    return train_df, val_df, None


def read_cosmos_data():
    training_file_cosmos = './data/cosmosqa/train.csv'
    val_file_cosmos = './data/cosmosqa/valid.csv'
    training_data_df = pd.read_csv(training_file_cosmos)
    val_data_df = pd.read_csv(val_file_cosmos)
    extracted_training_data_df = extract_df_with_correct_answer_for_cosmos(training_data_df)
    extracted_val_data_df = extract_df_with_correct_answer_for_cosmos(val_data_df)
    return extracted_training_data_df, extracted_val_data_df, None


# def read_dpr_data():
# print("Loading DPR Coreference Resolution data...")
# dataset = load_dataset("definite_pronoun_resolution")
# # using the test set as validation set. #Train = 1322, #Test = 564
# train, val, _ = dataset["train"], dataset["test"], None
# train_df = extract_df_from_dataset_for_dpr(train)
# val_df = extract_df_from_dataset_for_dpr(val)
# return train_df, val_df, None


def read_dpr_data_merged():
    print("USING MERGED DPR....")
    print("Loading DPR Coreference Resolution data...")
    dataset = load_dataset("definite_pronoun_resolution")
    train, test, _ = dataset["train"], dataset["test"], None
    train_df = extract_df_from_dataset_for_dpr(train)
    test_df = extract_df_from_dataset_for_dpr(test)
    merged_df = pd.concat([train_df, test_df], ignore_index=True)
    train_df, val_df = train_test_split(merged_df, test_size=0.1, random_state=0)
    return train_df, val_df, None


def read_dpr_data_for_testing():
    print("Loading DPR Coreference Resolution data...")
    dataset = load_dataset("definite_pronoun_resolution")
    # using the test set as validation set. #Train = 1322, #Test = 564
    train, test, _ = dataset["train"], dataset["test"], None
    train_df = extract_df_from_dataset_for_dpr(train)
    train_df, val_df = train_test_split(train_df, test_size=0.1, random_state=0)
    test_df = extract_df_from_dataset_for_dpr(test)
    return train_df, val_df, test_df


# def read_dpr_data_for_testing_val_from_test():
#     print("Loading DPR Coreference Resolution data with Val from Test...")
#     dataset = load_dataset("definite_pronoun_resolution")
#     train, test, _ = dataset["train"], dataset["test"], None
#     train_df = extract_df_from_dataset_for_dpr(train)
#     test_df = extract_df_from_dataset_for_dpr(test)
#     test_df, val_df = train_test_split(test_df, test_size=0.4, random_state=0)
#     return train_df, val_df, test_df


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


def extract_df_from_dataset_for_qqp(dataset):
    ### QQP Format is from T5 paper
    params = {'batch_size': 1, 'shuffle': False, 'num_workers': 2}
    loader = DataLoader(dataset, **params)
    data = []
    for batch in loader:
        questions = 'qqp question1: {} question2: {}'.format(batch['question1'][0].strip(),
                                                             batch['question2'][0].strip())
        answer = 'not_duplicate' if int(batch['label']) == 0 else 'duplicate'
        data.append([questions, answer])
    df = pd.DataFrame(data, columns=['questions', 'answer'])
    return df


def extract_df_from_dataset_for_wmt(dataset, language, max_count):
    ### WMT Format is from T5 paper
    params = {'batch_size': 1, 'shuffle': False, 'num_workers': 2}
    loader = DataLoader(dataset, **params)
    data = []
    prefix = get_prefix_for_wmt_langauge(language)

    counter = 0
    for batch in loader:
        counter += 1
        if counter > max_count:
            break
        original = f"{prefix} {batch['translation']['en'][0]}"
        translated = batch['translation'][language][0]
        data.append([original, translated])
    df = pd.DataFrame(data, columns=['original', 'translated'])
    return df


def extract_df_from_dataset_for_wikitext(dataset, seed, jumbled):
    params = {'batch_size': 1, 'shuffle': False, 'num_workers': 2}
    loader = DataLoader(dataset, **params)
    data = []
    random.seed(seed)
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

        text_word_list = text.split(' ')

        if jumbled:
            random.shuffle(text_word_list)

        if len(text_word_list) <= 10:
            continue

        masked_word_start = ''
        ctr = 0
        rand_idx = 1
        while (len(masked_word_start) <= 3 or masked_word_start == '[UNK]') and ctr < 20:
            rand_idx = random.randint(10, len(text_word_list) - 1)
            masked_word_start = text_word_list[rand_idx]
            ctr += 1
        if ctr == 20:
            continue

        input_text = ' '.join(text_word_list[:rand_idx])
        target_text = ' '.join(text_word_list[rand_idx:])
        data.append([f'Get next words: {input_text}', target_text])

    return pd.DataFrame(data, columns=['sentence', 'masked_sentence'])


def extract_df_from_dataset_for_books(dataset):
    params = {'batch_size': 1, 'shuffle': False, 'num_workers': 2}
    loader = DataLoader(dataset, **params)
    data = []
    for batch in loader:
        text = batch['text'][0].strip()
        # Following wikitext process
        if len(text) < 10 or len(text.split()) < 5:
            continue
        text_word_list = text.split(' ')
        if len(text_word_list) <= 5:
            continue

        masked_word_start = ''
        ctr = 0
        rand_idx = 1
        while (len(masked_word_start) <= 3) and ctr < 20:
            rand_idx = random.randint(3, len(text_word_list) - 1)
            masked_word_start = text_word_list[rand_idx]
            ctr += 1
        if ctr == 20:
            continue

        input_text = ' '.join(text_word_list[:rand_idx])
        target_text = ' '.join(text_word_list[rand_idx:])
        data.append([f'Get next words: {input_text}', target_text])

    return pd.DataFrame(data, columns=['sentence', 'masked_sentence'])


def extract_df_from_dataset_for_dpr(dataset):
    params = {'batch_size': 1, 'shuffle': False, 'num_workers': 2}
    loader = DataLoader(dataset, **params)
    data = []
    for batch in loader:
        label = int(batch['label'])
        sentence = batch['sentence'][0]
        pronoun = batch['pronoun'][0]
        antecedent = batch['candidates'][label][0]

        regex_pattern = f' {pronoun}(\\.|,|;| )'
        if not re.search(regex_pattern, sentence):
            raise AssertionError
        else:
            sentence = re.sub(regex_pattern, f' *{pronoun}* ', sentence)
        data.append([f'Get antecedent: {sentence}', antecedent])

    return pd.DataFrame(data, columns=['sentence', 'antecedent'])


def extract_df_with_correct_answer_for_cosmos(cosmos_data_df):
    data = []
    for idx, row in cosmos_data_df.iterrows():
        question_context_candidates = 'question: {} answer_0: {} answer_1: {} answer_2: {} answer_3: {} context: {}' \
            .format(row['question'], row['answer0'], row['answer1'], row['answer2'], row['answer3'], row['context'])
        data.append([question_context_candidates, row['label']])
    return pd.DataFrame(data, columns=['question_context_candidate', 'correct_answer'])


def get_renamed_commongen_columns(df):
    df = df.rename(columns={"scene": TARGET_TEXT, "concept_set": SOURCE_TEXT,
                            "all_scenes": OTHER})[[TARGET_TEXT, SOURCE_TEXT, OTHER]]
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
    df = df.rename(columns={"masked_sentence": TARGET_TEXT, "sentence": SOURCE_TEXT})[
        [TARGET_TEXT, SOURCE_TEXT]]
    return df


def get_renamed_dpr_columns(df):
    df = df.rename(columns={"antecedent": TARGET_TEXT, "sentence": SOURCE_TEXT})[
        [TARGET_TEXT, SOURCE_TEXT]]
    return df


def get_renamed_qqp_columns(df):
    df = df.rename(columns={"answer": TARGET_TEXT, "questions": SOURCE_TEXT})[
        [TARGET_TEXT, SOURCE_TEXT]]
    return df


def get_renamed_wmt_columns(df):
    df = df.rename(columns={"translated": TARGET_TEXT, "original": SOURCE_TEXT})[
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


def evaluate_predictions_bleu(predictions_filepath_validation, gram):
    df = pd.read_csv(predictions_filepath_validation)
    bleu_sum = 0
    weights = tuple(repeat(1 / gram, gram))

    for idx, row in df.iterrows():
        prediction = str(row["Generated Text"])
        actual = str(row["Actual Text"])
        prediction_list = utils.replace_special_chars_and_lower(prediction)
        actual_list = utils.replace_special_chars_and_lower(actual)
        ### Default is 4-gram. So need to modify weights for other n-grams.
        BLEUscore = nltk.translate.bleu_score.sentence_bleu([actual_list], prediction_list, weights=weights)
        bleu_sum += BLEUscore
    return 100 * bleu_sum / len(df)


def evaluate_all_predictions_bleu(predictions_filepath_validation, gram):
    df = pd.read_csv(predictions_filepath_validation)
    bleu_sum = 0
    weights = tuple(repeat(1 / gram, gram))

    for idx, row in df.iterrows():
        prediction = str(row["Generated Text"])
        actual_list = eval(row["other"])
        prediction_list = utils.replace_special_chars_and_lower(prediction)
        actual_list = [utils.replace_special_chars_and_lower(ref) for ref in actual_list]
        max_bleu = 0
        for actual in actual_list:
            ### Default is 4-gram. So need to modify weights for other n-grams.
            BLEUscore = nltk.translate.bleu_score.sentence_bleu([actual], prediction_list, weights=weights)
            max_bleu = max(max_bleu, BLEUscore)
        bleu_sum += max_bleu
    return 100 * bleu_sum / len(df)


def get_aux_accuracy(predictions_filepath, task):
    if task is None or task in [COSMOS, DPR, QQP]:
        accuracy = evaluate_e2e_tbsa.evaluate_exact_match_for_columns(predictions_filepath)
    elif task == SQUAD:
        accuracy = evaluate_squad_predictions(predictions_filepath)
    elif task in [WIKITEXT, WMTFR, WMTDE, BOOK, WIKIJUMBLED]:
        accuracy = evaluate_predictions_bleu(predictions_filepath, gram=2)
    elif task in [COMMONGEN]:
        accuracy = evaluate_all_predictions_bleu(predictions_filepath, gram=3)
    else:
        raise AssertionError("Task Evaluation not defined")

    return accuracy


def read_aux_data(task, seed):
    if task == COMMONGEN:
        return read_commongen_data()
    elif task == COSMOS:
        return read_cosmos_data()
    elif task == SQUAD:
        return read_squad_data()
    elif task == WIKITEXT:
        return read_wikitext_data(seed)
    elif task == DPR:
        return read_dpr_data_merged()
    elif task == QQP:
        return read_qqp_data()
    elif task == WMTFR:
        return read_wmt_data(FR)
    elif task == WMTDE:
        return read_wmt_data(DE)
    elif task == BOOK:
        return read_book_data()
    elif task == WIKIJUMBLED:
        return read_wikitext_data(seed, jumbled=True)


RENAMED_DF_FOR_TRAIN = {
    COSMOS: get_renamed_cosmos_columns,
    COMMONGEN: get_renamed_commongen_columns,
    ABSA: get_renamed_absa_columns,
    SQUAD: get_renamed_squad_columns,
    WIKITEXT: get_renamed_lm_columns,
    DPR: get_renamed_dpr_columns,
    QQP: get_renamed_qqp_columns,
    WMTFR: get_renamed_wmt_columns,
    WMTDE: get_renamed_wmt_columns,
    BOOK: get_renamed_lm_columns,
    WIKIJUMBLED: get_renamed_lm_columns
}


if __name__ == '__main__':
    # print(evaluate_all_predictions_bleu('Results/AmbiguousDataset8_ALSC/commongen.csv', 3))
    # read_wikitext_data(1)
    # read_wmt_data(FR)
    read_wikitext_data(0, True)
