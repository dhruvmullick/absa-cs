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

### Prompt taken from own_commongen paper https://aclanthology.org/2020.findings-emnlp.165.pdf
COMMONGEN_PROMPT = 'generate a sentence with: '
ABSA_PROMPT = "aspect analysis: "

FRACTION = 0.1
ABSA_MULTIPLIER = 2

# COMMONGEN_FRACTION_LIST = [0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.4, 0.5, 1]
ABSA_MULTIPLIER_LIST = [0.5, 1, 1.5, 2, 2.5, 4, 8, 16]

# WHITELISTED_COMMONGEN_FRACTION_LIST = [1]
# WHITELISTED_ABSA_MULTIPLIER_LIST = [16]

# COMMONGEN_FRACTION_LIST = [0.5, 1]
# ABSA_MULTIPLIER_LIST = [0.5, 1, 1.5, 2, 2.5, 4, 8, 16]

# sys.argv[1] will have the commonsense fraction
MODEL_DIRECTORY = 'models/dataset5_test_mams_train_cs_{}'.format(sys.argv[1])
EXPERIMENT_OUTPUT_FILE = '{}/outputs.txt'.format(MODEL_DIRECTORY)
PREDICTION_FILE_NAME = 'evaluation_commongen_predictions.csv'
TRANSFORMED_TARGETS_PREDICTIONS_FILE_NAME = 'transformed-targets.csv'
TRANSFORMED_SENTIMENTS_PREDICTIONS_FILE_NAME = 'transformed-sentiments.csv'

console = Console(record=True)
# Set random seeds and deterministic pytorch for reproducibility
torch.manual_seed(0)  # pytorch random seed
np.random.seed(0)  # numpy random seed
torch.backends.cudnn.deterministic = True

# Setting up the device for GPU usage
device = 'cuda' if cuda.is_available() else 'cpu'


def process_concepts(concept_set):
    return concept_set.replace("#", " ")


def read_json_file(file_name):
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


def get_renamed_commongen_columns(df):
    df = df.rename(columns={"scene": "target", "concept_set": "source"})[['target', 'source']]
    return df


def get_renamed_absa_columns(df):
    df = df.rename(columns={"sentences_opinions": "target", "sentences_texts": "source"})[['target', 'source']]
    return df


def build_data_for_absa(dataframes, source_text, target_text):
    # tokenzier for encoding the text
    tokenizer = T5Tokenizer.from_pretrained(model_params["MODEL"])
    tokenizer.add_tokens(['<sep>'])

    # logging
    console.log(f"[Data]: Reading ABSA data...\n")

    # Creation of Dataset and Dataloader
    train_dataset = dataframes[0].sample(frac=1, random_state=0).reset_index(drop=True)
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


def merge_absa_commongen(dataset_absa, dataset_commongen, absa_multiplier=ABSA_MULTIPLIER, commongen_fraction=FRACTION,
                         fileOutput=True):
    ### Try different sample sizes of commongen dataset
    print("ABSA Multiplier is = {}, CG fraction is = {}".format(absa_multiplier, commongen_fraction))
    if fileOutput:
        with open(EXPERIMENT_OUTPUT_FILE, 'a') as file:
            file.write("{}, {}, ".format(absa_multiplier, commongen_fraction))

    dataset_commongen = dataset_commongen.sample(frac=commongen_fraction, random_state=0).reset_index(drop=True)
    if absa_multiplier >= 1:
        absa_multiplier_int = int(absa_multiplier)
        absa_multiplier_float = absa_multiplier - absa_multiplier_int

        dataset_absa_sampled = dataset_absa.sample(frac=absa_multiplier_float, random_state=0).reset_index(drop=True)
        dataset_absa_multiplied = pd.concat([dataset_absa] * absa_multiplier_int, ignore_index=True)

        dataset_absa = pd.concat([dataset_absa_multiplied, dataset_absa_sampled])
    else:
        dataset_absa = dataset_absa.sample(frac=absa_multiplier, random_state=0).reset_index(drop=True)

    print("ABSA length is... {}, CG length is... {}".format(len(dataset_absa), len(dataset_commongen)))

    if commongen_fraction > 0:
        return pd.concat([dataset_absa, dataset_commongen], ignore_index=True)
    else:
        return pd.concat([dataset_absa], ignore_index=True)
    # print("NOT USING COMMONSENSE")
    # return pd.concat([dataset_absa], ignore_index=True)


# def merge_absa_commongen_equally(dataset_absa, dataset_commongen):
#     ### Try different sample sizes of commongen dataset
#     dataset_commongen = dataset_commongen.sample(n=len(dataset_absa)).reset_index(drop=True)
#     print("ABSA length is... {}, CG length is... {}".format(len(dataset_absa), len(dataset_commongen)))
#
#     print("Merging absa and cg equally...")
#     return pd.concat([dataset_absa, dataset_commongen], ignore_index=True)
#     # print("NOT USING COMMONSENSE")
#     # return pd.concat([dataset_absa], ignore_index=True)


def build_data_for_commongen(dataframes, source_text, target_text, training_dataset_absa, val_dataset_absa,
                             test_dataset_absa, tokenizer, absa_multiplier, commongen_fraction):
    console.log(f"[Data]: Reading CommonGen data...\n")

    # Creation of Dataset and Dataloader
    train_dataset_commongen = dataframes[0].sample(frac=1, random_state=0).reset_index(drop=True)
    val_dataset_commongen = dataframes[1].reset_index(drop=True)
    # test_dataset_commongen = dataframes[2].reset_index(drop=True)
    train_dataset_commongen['concept_set'] = COMMONGEN_PROMPT + train_dataset_commongen['concept_set']
    val_dataset_commongen['concept_set'] = COMMONGEN_PROMPT + val_dataset_commongen['concept_set']
    # test_dataset_commongen['concept_set'] = COMMONGEN_PROMPT + test_dataset_commongen['concept_set']
    train_dataset_commongen = get_renamed_commongen_columns(train_dataset_commongen)
    val_dataset_commongen = get_renamed_commongen_columns(val_dataset_commongen)
    # test_dataset_commongen = get_renamed_commongen_columns(test_dataset_commongen)

    console.print(f"TRAIN Dataset CommonGen: {train_dataset_commongen.shape}")
    console.print(f"VALIDATION Dataset CommonGen: {val_dataset_commongen.shape}")
    # console.print(f"TEST Dataset: {test_dataset_commongen.shape}\n")

    # training_set = merge_absa_commongen(training_dataset_absa, train_dataset_commongen, 0.5, 0.05)
    # val_set = merge_absa_commongen(val_dataset_absa, val_dataset_commongen, 0.5, 0.05)
    # training_set = merge_absa_commongen_equally(training_dataset_absa, training_data_commongen)
    # val_set = merge_absa_commongen_equally(val_dataset_absa, val_dataset_commongen)
    training_set = merge_absa_commongen(training_dataset_absa, train_dataset_commongen, absa_multiplier,
                                        commongen_fraction, True)
    val_set = merge_absa_commongen(val_dataset_absa, val_dataset_commongen, absa_multiplier,
                                   min(commongen_fraction, 0.1), False)
    test_set = test_dataset_absa

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


if __name__ == '__main__':
    model_params = {
        "OUTPUT_PATH": MODEL_DIRECTORY,  # output path
        "MODEL": "t5-base",
        "TRAIN_BATCH_SIZE": 16,  # training batch size
        "VALID_BATCH_SIZE": 16,  # validation batch size
        "TRAIN_EPOCHS": 10,  # number of training epochs
        "VAL_EPOCHS": 1,  # number of validation epochs
        "LEARNING_RATE": 5e-4,  # learning rate
        "MAX_SOURCE_TEXT_LENGTH": 256,  # max length of source text
        "MAX_TARGET_TEXT_LENGTH": 64,  # max length of target text
        "early_stopping_patience": 5,  # number of epochs before stopping training.
    }

    print(model_params)

    if not os.path.exists(MODEL_DIRECTORY):
        os.makedirs(MODEL_DIRECTORY)

    # training_file_absa = './data/merged_train.csv'
    # validation_file_absa = './data/merged_val.csv'
    test_file_absa = 'data/merged_test_ambiguous.csv'
    training_file_absa = './data/processed_train_Mams_en.csv'
    validation_file_absa = './data/processed_val_Mams_en.csv'
    # test_file_absa = './data/processed_test_Mams_en.csv'
    print("Training on: {}, Testing on: {}".format(training_file_absa, test_file_absa))
    print("ABSA Prompt is: {}".format(ABSA_PROMPT))
    training_absa = pd.read_csv(training_file_absa)
    validation_absa = pd.read_csv(validation_file_absa)
    test_absa = pd.read_csv(test_file_absa)
    training_set_absa, val_set_absa, test_set_absa, tokenizer = build_data_for_absa(
        dataframes=[training_absa, validation_absa, test_absa], source_text="source", target_text="target")

    training_file_commongen = './data/commongen_data/commongen.train.jsonl'
    test_file_commongen = './data/commongen_data/commongen.dev.jsonl'
    training_data_commongen = read_json_file(training_file_commongen)
    testing_data_commongen = read_json_file(test_file_commongen)

    training_data_commongen, validation_data_commongen = train_test_split(training_data_commongen, test_size=0.1,
                                                                          random_state=0)
    commongen_fraction = float(sys.argv[1])

    with open(EXPERIMENT_OUTPUT_FILE, 'w') as file:
        file.write("----------------------\n")

    for absa_multiplier in ABSA_MULTIPLIER_LIST:

        torch.manual_seed(0)  # pytorch random seed
        np.random.seed(0)  # numpy random seed

        training_loader, validation_loader, test_loader, tokenizer = build_data_for_commongen(
            [training_data_commongen, validation_data_commongen, testing_data_commongen],
            "source", "target", training_set_absa, val_set_absa, test_set_absa, tokenizer, absa_multiplier,
            commongen_fraction)

        T5Trainer(training_loader, validation_loader, tokenizer, model_params=model_params)

        ### Test loader is only ABSA
        T5Generator(test_loader, model_params=model_params, output_file=PREDICTION_FILE_NAME)

        e2e_tbsa_preprocess.run_from_generative_script(
            predictions_filepath='{}/{}'.format(MODEL_DIRECTORY, PREDICTION_FILE_NAME),
            transformed_targets_filepath='{}/{}'.format(MODEL_DIRECTORY, TRANSFORMED_TARGETS_PREDICTIONS_FILE_NAME),
            transformed_sentiments_filepath='{}/{}'.format(MODEL_DIRECTORY,
                                                           TRANSFORMED_SENTIMENTS_PREDICTIONS_FILE_NAME))
        evaluate_e2e_tbsa.run_from_generative_script(
            target_file_to_evaluate='{}/{}'.format(MODEL_DIRECTORY, TRANSFORMED_TARGETS_PREDICTIONS_FILE_NAME),
            sentiments_file_to_evaluate='{}/{}'.format(MODEL_DIRECTORY, TRANSFORMED_SENTIMENTS_PREDICTIONS_FILE_NAME),
            file_to_write=EXPERIMENT_OUTPUT_FILE)
