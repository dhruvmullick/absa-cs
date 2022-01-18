import json

import torch
import os
import numpy as np
import pandas as pd
from rich.console import Console
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from transformers import T5ForConditionalGeneration, T5Tokenizer
from torch import cuda
from train_generative import T5Trainer, T5Generator

# define a rich console logger
from train_generative import YourDataSetClass

### Prompt taken from own_commongen paper https://aclanthology.org/2020.findings-emnlp.165.pdf
COMMONGEN_PROMPT = 'generate a sentence with: '

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


def build_data(dataframes, source_text, target_text):
    # tokenzier for encoding the text
    tokenizer = T5Tokenizer.from_pretrained(model_params["MODEL"])
    # logging
    console.log(f"[Data]: Reading data...\n")

    # Creation of Dataset and Dataloader
    train_dataset = dataframes[0].sample(frac=1, random_state=0).reset_index(drop=True)
    val_dataset = dataframes[1].reset_index(drop=True)
    test_dataset = dataframes[2].reset_index(drop=True)
    train_dataset['concept_set'] = COMMONGEN_PROMPT + train_dataset['concept_set']
    val_dataset['concept_set'] = COMMONGEN_PROMPT + val_dataset['concept_set']
    test_dataset['concept_set'] = COMMONGEN_PROMPT + test_dataset['concept_set']

    console.print(f"TRAIN Dataset: {train_dataset.shape}")
    console.print(f"VALIDATION Dataset: {val_dataset.shape}")
    console.print(f"TEST Dataset: {test_dataset.shape}\n")

    # Creating the Training and Validation dataset for further creation of Dataloader
    training_set = YourDataSetClass(train_dataset, tokenizer, model_params["MAX_SOURCE_TEXT_LENGTH"],
                                    model_params["MAX_TARGET_TEXT_LENGTH"], source_text, target_text)
    val_set = YourDataSetClass(val_dataset, tokenizer, model_params["MAX_SOURCE_TEXT_LENGTH"],
                               model_params["MAX_TARGET_TEXT_LENGTH"], source_text, target_text)
    test_set = YourDataSetClass(test_dataset, tokenizer, model_params["MAX_SOURCE_TEXT_LENGTH"],
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
        "OUTPUT_PATH": f"./models/commongen_evaluation/",  # output path
        "MODEL": "t5-base",
        "TRAIN_BATCH_SIZE": 16,  # training batch size
        "VALID_BATCH_SIZE": 16,  # validation batch size
        "TRAIN_EPOCHS": 300,  # number of training epochs
        "VAL_EPOCHS": 1,  # number of validation epochs
        "LEARNING_RATE": 5e-4,  # learning rate
        "MAX_SOURCE_TEXT_LENGTH": 256,  # max length of source text
        "MAX_TARGET_TEXT_LENGTH": 64,  # max length of target text
        "early_stopping_patience": 10,  # number of epochs before stopping training.
    }

    training_file = './data/commongen_data/commongen.train.jsonl'
    test_file = './data/commongen_data/commongen.dev.jsonl'

    training_data = read_json_file(training_file)
    testing_data = read_json_file(test_file)

    training_data, validation_data = train_test_split(training_data, test_size=0.2, random_state=0)

    training_loader, validation_loader, test_loader, tokenizer = build_data(
        dataframes=[training_data, validation_data, testing_data],
        source_text="concept_set", target_text="scene")

    T5Trainer(training_loader, validation_loader, tokenizer, model_params=model_params)

    ### For testing purposes
    # COMMONGEN_PREDICTION_FILE = 'evaluation_commongen_predictions.csv'
    # T5Generator(test_loader, model_params=model_params, output_file=COMMONGEN_PREDICTION_FILE)
    # test_predictions = pd.read_csv(COMMONGEN_PREDICTION_FILE)
