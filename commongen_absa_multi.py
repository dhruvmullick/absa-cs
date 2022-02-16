import json

import torch
import sys
import os
import numpy as np
import pandas as pd
from rich.console import Console
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from transformers import T5Tokenizer
from torch import cuda

import e2e_tbsa_preprocess
import evaluate_e2e_tbsa
from train_generative import T5Trainer, T5Generator

# define a rich console logger
from train_generative import YourDataSetClass

TARGET_TEXT = "target"
SOURCE_TEXT = "source"
### Prompt taken from own_commongen paper https://aclanthology.org/2020.findings-emnlp.165.pdf
COMMONGEN_PROMPT = 'generate a sentence with: '
ABSA_PROMPT = "aspect analysis: "

FRACTION = 0.1
ABSA_MULTIPLIER = 2

# COMMONGEN_FRACTION_LIST = [0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.4, 0.5, 1]
# ABSA_MULTIPLIER_LIST = [0.1, 0.2, 0.5, 1, 2, 4, 8, 16]
ABSA_MULTIPLIER_LIST = [0.5, 1, 2, 4, 8, 16]
# ABSA_MULTIPLIER_LIST = [8]

# COMMONGEN_FRACTION_LIST = [0.5, 1]
# ABSA_MULTIPLIER_LIST = [0.5, 1, 1.5, 2, 2.5, 4, 8, 16]

# sys.argv[1] will have the commonsense fraction
COMMONSENSE_FRACTION = float(sys.argv[1])
# MODEL_DIRECTORY = 'models/dataset5_randomised2_test_mams_train_cs_{}'.format(COMMONSENSE_FRACTION)
MODEL_DIRECTORY = 'models/cosmos_dataset6_test_mams_train_cs_{}'.format(COMMONSENSE_FRACTION)
EXPERIMENT_OUTPUT_FILE_TARGET = '{}/output_targets.csv'.format(MODEL_DIRECTORY)
EXPERIMENT_OUTPUT_FILE_SENTIMENT = '{}/output_sentiment.csv'.format(MODEL_DIRECTORY)

PREDICTION_FILE_NAME = 'evaluation_cosmos_predictions.csv'
### cs_absa_seed
# PREDICTION_FILE_NAME_FORMAT = 'evaluation_commongen_predictions_{}_{}_{}.csv'
TRANSFORMED_TARGETS_PREDICTIONS_FILE_NAME = 'transformed-targets.csv'
TRANSFORMED_SENTIMENTS_PREDICTIONS_FILE_NAME = 'transformed-sentiments.csv'

console = Console(record=True)

# SEEDS = [5, 6, 7, 8, 9]
SEEDS = [0, 1, 2]

torch.backends.cudnn.deterministic = True

# Setting up the device for GPU usage
device = 'cuda' if cuda.is_available() else 'cpu'


def process_concepts(concept_set):
    return concept_set.replace("#", " ")


def read_commongen_json_file(file_name):
    concept_sets = []
    scenes = []
    with open(file_name, 'r') as file:
        for line in file:
            json_line = json.loads(line)
            concept_set = process_concepts(json_line['concept_set'])
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


def extract_df_with_correct_answer(cosmos_data_df):
    data = []
    for idx, row in cosmos_data_df.iterrows():
        correct_answer_column = 'answer' + str(row['label'])
        question_context = 'question: {} context: {}'.format(row['question'], row['context'])
        data.append([question_context, row[correct_answer_column]])
    return pd.DataFrame(data, columns=['question_context', 'correct_answer'])


def read_cosmos_data():
    training_file_cosmos = './data/cosmosqa/train.csv'
    val_file_cosmos = './data/cosmosqa/valid.csv'
    training_data_df = pd.read_csv(training_file_cosmos)
    val_data_df = pd.read_csv(val_file_cosmos)
    extracted_training_data_df = extract_df_with_correct_answer(training_data_df)
    extracted_val_data_df = extract_df_with_correct_answer(val_data_df)

    return extracted_training_data_df, extracted_val_data_df, None


def get_renamed_commongen_columns(df):
    df = df.rename(columns={"scene": TARGET_TEXT, "concept_set": SOURCE_TEXT})[[TARGET_TEXT, SOURCE_TEXT]]
    return df

def get_renamed_cosmos_columns(df):
    df = df.rename(columns={"correct_answer": TARGET_TEXT, "question_context": "source"})[[TARGET_TEXT, SOURCE_TEXT]]
    return df

def get_renamed_absa_columns(df):
    df = df.rename(columns={"sentences_opinions": TARGET_TEXT, "sentences_texts": SOURCE_TEXT})[[TARGET_TEXT, SOURCE_TEXT]]
    return df


def build_data_for_absa(model_params, dataframes):
    # tokenzier for encoding the text
    tokenizer = T5Tokenizer.from_pretrained(model_params["MODEL"])
    tokenizer.add_tokens(['<sep>'])

    # logging
    console.log(f"[Data]: Reading ABSA data...\n")

    # Creation of Dataset and Dataloader
    train_dataset = dataframes[0].sample(frac=1, random_state=model_params['SEED']).reset_index(drop=True)
    val_dataset = dataframes[1].reset_index(drop=True)
    test_dataset = dataframes[2].reset_index(drop=True)
    train_dataset['sentences_texts'] = ABSA_PROMPT + train_dataset['sentences_texts']
    val_dataset['sentences_texts'] = ABSA_PROMPT + val_dataset['sentences_texts']
    test_dataset['sentences_texts'] = ABSA_PROMPT + test_dataset['sentences_texts']
    train_dataset = get_renamed_absa_columns(train_dataset)
    val_dataset = get_renamed_absa_columns(val_dataset)
    test_dataset = get_renamed_absa_columns(test_dataset)

    console.print(f"TRAIN Dataset ABSA: {train_dataset.shape}")
    console.print(f"VALIDATION Dataset ABSA: {val_dataset.shape}")
    console.print(f"TEST Dataset ABSA: {test_dataset.shape}\n")

    return train_dataset, val_dataset, test_dataset, tokenizer


def merge_absa_commonsense(dataset_absa, dataset_cs, model_params, absa_multiplier=ABSA_MULTIPLIER,
                           cs_fraction=FRACTION, fileOutput=True):
    ### Try different sample sizes of commongen dataset
    print("ABSA Multiplier is = {}, CG fraction is = {}".format(absa_multiplier, cs_fraction))

    dataset_cs = dataset_cs.sample(frac=cs_fraction,
                                   random_state=model_params['SEED']).reset_index(drop=True)
    if absa_multiplier >= 1:
        absa_multiplier_int = int(absa_multiplier)
        absa_multiplier_float = absa_multiplier - absa_multiplier_int

        dataset_absa_sampled = dataset_absa.sample(frac=absa_multiplier_float,
                                                   random_state=model_params['SEED']).reset_index(drop=True)
        dataset_absa_multiplied = pd.concat([dataset_absa] * absa_multiplier_int, ignore_index=True)

        dataset_absa = pd.concat([dataset_absa_multiplied, dataset_absa_sampled])
    else:
        dataset_absa = dataset_absa.sample(frac=absa_multiplier, random_state=model_params['SEED']).reset_index(
            drop=True)

    print("ABSA length is... {}, CG length is... {}".format(len(dataset_absa), len(dataset_cs)))

    if cs_fraction > 0:
        return pd.concat([dataset_absa, dataset_cs], ignore_index=True)
    else:
        return pd.concat([dataset_absa], ignore_index=True)


def build_merged_data_for_commongen(dataframes, training_dataset_absa, val_dataset_absa,
                                    test_dataset_absa, tokenizer, absa_multiplier, commongen_fraction, model_params):
    console.log(f"[Data]: Reading CommonGen data...\n")

    # Creation of Dataset and Dataloader
    train_dataset_commongen = dataframes[0].sample(frac=1, random_state=model_params['SEED']).reset_index(drop=True)
    val_dataset_commongen = dataframes[1].reset_index(drop=True)

    train_dataset_commongen['concept_set'] = COMMONGEN_PROMPT + train_dataset_commongen['concept_set']
    val_dataset_commongen['concept_set'] = COMMONGEN_PROMPT + val_dataset_commongen['concept_set']

    train_dataset_commongen = get_renamed_commongen_columns(train_dataset_commongen)
    val_dataset_commongen = get_renamed_commongen_columns(val_dataset_commongen)

    console.print(f"TRAIN Dataset CommonGen: {train_dataset_commongen.shape}")
    console.print(f"VALIDATION Dataset CommonGen: {val_dataset_commongen.shape}")

    training_set = merge_absa_commonsense(training_dataset_absa, train_dataset_commongen, model_params, absa_multiplier,
                                          commongen_fraction, True)
    val_set = merge_absa_commonsense(val_dataset_absa, val_dataset_commongen, model_params, 1,
                                     min(commongen_fraction, 0.1), False)
    test_set = test_dataset_absa

    return tokenizer, training_set, val_set, test_set


def build_merged_data_for_cosmos(dataframes, training_dataset_absa, val_dataset_absa,
                                 test_dataset_absa, tokenizer, absa_multiplier, cosmos_fraction, model_params):
    console.log(f"[Data]: Reading Cosmos data...\n")

    # Creation of Dataset and Dataloader
    train_dataset_cosmos = dataframes[0].sample(frac=1, random_state=model_params['SEED']).reset_index(drop=True)
    val_dataset_cosmos = dataframes[1].reset_index(drop=True)

    train_dataset_cosmos = get_renamed_cosmos_columns(train_dataset_cosmos)
    val_dataset_cosmos = get_renamed_cosmos_columns(val_dataset_cosmos)

    console.print(f"TRAIN Dataset Cosmos: {train_dataset_cosmos.shape}")
    console.print(f"VALIDATION Dataset Cosmos: {val_dataset_cosmos.shape}")

    training_set = merge_absa_commonsense(training_dataset_absa, train_dataset_cosmos, model_params, absa_multiplier,
                                          cosmos_fraction, True)
    val_set = merge_absa_commonsense(val_dataset_absa, val_dataset_cosmos, model_params,
                                     1, min(cosmos_fraction, 0.2), False)
    test_set = test_dataset_absa

    return tokenizer, training_set, val_set, test_set


def get_data_loaders(model_params, source_text, target_text, tokenizer, training_set, val_set, test_set):
    # Creating the Training and Validation dataset for further creation of Dataloader
    training_set = YourDataSetClass(training_set, tokenizer, model_params["MAX_SOURCE_TEXT_LENGTH"],
                                    model_params["MAX_TARGET_TEXT_LENGTH"], source_text, target_text)
    val_set = YourDataSetClass(val_set, tokenizer, model_params["MAX_SOURCE_TEXT_LENGTH"],
                               model_params["MAX_TARGET_TEXT_LENGTH"], source_text, target_text)
    test_set = YourDataSetClass(test_set, tokenizer, model_params["MAX_SOURCE_TEXT_LENGTH"],
                                model_params["MAX_TARGET_TEXT_LENGTH"], source_text, target_text)
    # Defining the parameters for creation of dataloaders
    train_params = {'batch_size': model_params["TRAIN_BATCH_SIZE"], 'shuffle': True, 'num_workers': 2}
    val_params = {'batch_size': model_params["VALID_BATCH_SIZE"], 'shuffle': False, 'num_workers': 2}
    test_params = {'batch_size': model_params["VALID_BATCH_SIZE"], 'shuffle': False, 'num_workers': 2}
    # Creation of Dataloaders for testing and validation. This will be used down for training and validation stage for the model.
    training_loader = DataLoader(training_set, **train_params)
    validation_loader = DataLoader(val_set, **val_params)
    test_loader = DataLoader(test_set, **test_params)

    return training_loader, validation_loader, test_loader, tokenizer


def run_program_for_seed(seed, results_target, results_sentiment):
    # Set random seeds and deterministic pytorch for reproducibility
    torch.manual_seed(seed)  # pytorch random seed
    np.random.seed(seed)  # numpy random seed

    model_params = {
        "OUTPUT_PATH": MODEL_DIRECTORY,  # output path
        "MODEL": "t5-base",
        # "MODEL": "danny911kr/calm-base",
        "TRAIN_BATCH_SIZE": 32,  # training batch size
        "VALID_BATCH_SIZE": 32,  # validation batch size
        "TRAIN_EPOCHS": 10,  # number of training epochs
        "VAL_EPOCHS": 1,  # number of validation epochs
        "LEARNING_RATE": 5e-4,  # learning rate
        "MAX_SOURCE_TEXT_LENGTH": 256,  # max length of source text
        "MAX_TARGET_TEXT_LENGTH": 64,  # max length of target text
        "early_stopping_patience": 5,  # number of epochs before stopping training.
        "SEED": seed  # to use for randomisation
    }

    print(model_params)

    # training_file_absa = './data/merged_train.csv'
    # validation_file_absa = './data/merged_val.csv'
    test_file_absa = 'data/merged_test_ambiguous.csv'
    training_file_absa = './data/processed_train_Mams_en.csv'
    validation_file_absa = './data/processed_val_Mams_en.csv'
    # test_file_absa = './data/processed_test_Mams_en.csv'
    # training_file_absa = './data/processed_train_Rest16_en.csv'
    # validation_file_absa = './data/processed_val_Rest16_en.csv'
    # test_file_absa = './data/processed_test_Rest16_en.csv'

    print("Training on: {}, Testing on: {}, Seed: {}".format(training_file_absa, test_file_absa, seed))
    print("ABSA Prompt is: {}".format(ABSA_PROMPT))
    training_absa = pd.read_csv(training_file_absa)
    validation_absa = pd.read_csv(validation_file_absa)
    test_absa = pd.read_csv(test_file_absa)

    training_set_absa, val_set_absa, test_set_absa, tokenizer \
        = build_data_for_absa(model_params, dataframes=[training_absa, validation_absa, test_absa])

    # training_data_commongen, validation_data_commongen, testing_data_commongen = read_commongen_data(seed)
    training_data_cosmos, validation_data_cosmos, testing_data_cosmos = read_cosmos_data()

    for absa_multiplier in ABSA_MULTIPLIER_LIST:

        torch.manual_seed(seed)  # pytorch random seed
        np.random.seed(seed)  # numpy random seed

        # tokenizer, training_set, val_set, test_set = build_merged_data_for_commongen(
        #     [training_data_commongen, validation_data_commongen, testing_data_commongen],
        #     training_set_absa, val_set_absa, test_set_absa, tokenizer, absa_multiplier,
        #     COMMONSENSE_FRACTION, model_params)

        tokenizer, training_set, val_set, test_set = build_merged_data_for_cosmos(
            [training_data_cosmos, validation_data_cosmos, testing_data_cosmos],
            training_set_absa, val_set_absa, test_set_absa, tokenizer, absa_multiplier,
            COMMONSENSE_FRACTION, model_params)

        training_loader, validation_loader, test_loader, tokenizer = \
            get_data_loaders(model_params, SOURCE_TEXT, TARGET_TEXT, tokenizer, training_set, val_set, test_set)

        T5Trainer(training_loader, validation_loader, tokenizer, model_params=model_params)

        prediction_file_name = PREDICTION_FILE_NAME

        ### Test loader is only ABSA
        T5Generator(test_loader, model_params=model_params, output_file=prediction_file_name)

        e2e_tbsa_preprocess.run_from_generative_script(
            predictions_filepath='{}/{}'.format(MODEL_DIRECTORY, prediction_file_name),
            transformed_targets_filepath='{}/{}'.format(MODEL_DIRECTORY, TRANSFORMED_TARGETS_PREDICTIONS_FILE_NAME),
            transformed_sentiments_filepath='{}/{}'.format(MODEL_DIRECTORY,
                                                           TRANSFORMED_SENTIMENTS_PREDICTIONS_FILE_NAME))
        output = evaluate_e2e_tbsa.run_from_generative_script(
            target_file_to_evaluate='{}/{}'.format(MODEL_DIRECTORY, TRANSFORMED_TARGETS_PREDICTIONS_FILE_NAME),
            sentiments_file_to_evaluate='{}/{}'.format(MODEL_DIRECTORY, TRANSFORMED_SENTIMENTS_PREDICTIONS_FILE_NAME))

        if absa_multiplier in results_target.keys():
            results_target[absa_multiplier].append(output['te'])
            results_sentiment[absa_multiplier].append(output['tse'])
        else:
            results_target[absa_multiplier] = [output['te']]
            results_sentiment[absa_multiplier] = [output['tse']]

        print("Results target: " + str(results_target))
        print("Results sentiment: " + str(results_sentiment))


def write_to_results_file(results_dict, output_file):
    f1_list = results_dict[absa_multiplier]
    output_file.write('{}, {}, '.format(absa_multiplier, COMMONSENSE_FRACTION))
    for f1 in f1_list:
        output_file.write('{}, '.format(f1))
    output_file.write('\n')


if __name__ == '__main__':

    if not os.path.exists(MODEL_DIRECTORY):
        os.makedirs(MODEL_DIRECTORY)

    with open(EXPERIMENT_OUTPUT_FILE_TARGET, 'w') as file:
        file.write("----------------------\n")

    with open(EXPERIMENT_OUTPUT_FILE_SENTIMENT, 'w') as file:
        file.write("----------------------\n")

    results_target = {}
    results_sentiment = {}

    for seed in SEEDS:
        run_program_for_seed(seed, results_target, results_sentiment)

    with open(EXPERIMENT_OUTPUT_FILE_TARGET, 'a') as file_targets:
        with open(EXPERIMENT_OUTPUT_FILE_SENTIMENT, 'a') as file_sentiments:
            file_targets.write("Seeds used: {}\n".format(SEEDS))
            file_sentiments.write("Seeds used: {}\n".format(SEEDS))
            for absa_multiplier in results_target.keys():
                write_to_results_file(results_target, file_targets)
                write_to_results_file(results_sentiment, file_sentiments)

    print("Done!")
