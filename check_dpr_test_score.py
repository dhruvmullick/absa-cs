import torch
import sys
import os
import numpy as np
import pandas as pd
from rich.console import Console
from torch.utils.data import DataLoader
from transformers import T5Tokenizer
from torch import cuda

import aux_processor
import evaluate_e2e_tbsa
from train_generative import T5Trainer, T5Generator, YourDataSetClass

from aux_processor import RENAMED_DF_FOR_TRAIN
from aux_processor import read_dpr_data_for_testing
from aux_processor import TARGET_TEXT, SOURCE_TEXT

DELTA = 0.001

SEEDS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]

MODEL_DIR = sys.argv[1]

print("SEEDS: {}".format(SEEDS))

MODEL_PATH_TO_TEST = f'analysis_t5_large/{MODEL_DIR}'
DPR_MODEL_DIRECTORY = f'analysis_t5_large/training/{MODEL_DIR}'

PREDICTION_FILE_NAME = 'evaluation_predictions.csv'
PREDICTION_FILE_NAME_VAL = 'evaluation_predictions_val.csv'

console = Console(record=True)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Setting up the device for GPU usage
device = 'cuda' if cuda.is_available() else 'cpu'


def build_data_for_aux_task(dataframes, task_name, model_params):
    console.log("[Data]: Reading {} data...\n".format(task_name))
    train_dataset_aux = dataframes[0].sample(frac=1, random_state=model_params['SEED']).reset_index(
        drop=True)
    val_dataset_aux = dataframes[1].sample(frac=1.0, random_state=model_params['SEED']).reset_index(drop=True)
    test_dataset_aux = dataframes[2].sample(frac=1.0, random_state=model_params['SEED']).reset_index(drop=True)

    column_renamer = RENAMED_DF_FOR_TRAIN[task_name]
    train_dataset_aux = column_renamer(train_dataset_aux)
    val_dataset_aux = column_renamer(val_dataset_aux)
    test_dataset_aux = column_renamer(test_dataset_aux)

    console.print(f"TRAIN Dataset {task_name}: {train_dataset_aux.shape}")
    console.print(f"VALIDATION Dataset {task_name}: {val_dataset_aux.shape}")
    console.print(f"TEST Dataset {task_name}: {test_dataset_aux.shape}")

    return train_dataset_aux, val_dataset_aux, test_dataset_aux


def get_data_loaders(model_params, source_text, target_text, tokenizer, training_set, val_set, test_set):
    # Creating the Training and Validation dataset for further creation of Dataloader
    training_set = YourDataSetClass(training_set, tokenizer, model_params["MAX_SOURCE_TEXT_LENGTH"],
                                    model_params["MAX_TARGET_TEXT_LENGTH"], source_text, target_text)
    val_set = YourDataSetClass(val_set, tokenizer, model_params["MAX_SOURCE_TEXT_LENGTH"],
                               model_params["MAX_TARGET_TEXT_LENGTH"], source_text, target_text)
    if test_set is not None:
        test_set = YourDataSetClass(test_set, tokenizer, model_params["MAX_SOURCE_TEXT_LENGTH"],
                                    model_params["MAX_TARGET_TEXT_LENGTH"], source_text, target_text)

    # Defining the parameters for creation of dataloaders
    train_params = {'batch_size': model_params["TRAIN_BATCH_SIZE"], 'shuffle': True, 'num_workers': 2}
    val_params = {'batch_size': model_params["VALID_BATCH_SIZE"], 'shuffle': False, 'num_workers': 2}
    test_params = {'batch_size': model_params["TEST_BATCH_SIZE"], 'shuffle': False, 'num_workers': 2}

    # Creation of Dataloaders for testing and validation. This will be used down for training and validation stage for the model.
    training_loader = DataLoader(training_set, **train_params)
    validation_loader = DataLoader(val_set, **val_params)
    if test_set is not None:
        test_loader = DataLoader(test_set, **test_params)
    else:
        test_loader = None

    return training_loader, validation_loader, test_loader, tokenizer


def get_aux_lr():
    return 5e-4


def run_program_for_seed_lr(seed):
    # Set random seeds and deterministic pytorch for reproducibility
    torch.manual_seed(seed)  # pytorch random seed
    np.random.seed(seed)  # numpy random seed

    aux_lr = get_aux_lr()

    model_params_aux = {
        "OUTPUT_PATH": DPR_MODEL_DIRECTORY,  # output path
        "MODEL": "t5-base",
        "MODEL_LOCAL": "/home/mullick/lm_models/t5-base-conditional-gen",
        # "MODEL": "danny911kr/calm-base",
        # "TRAIN_BATCH_SIZE": 24,  # SQUAD: Cedar 24. Host1 16. need small for SQUAD as 512 source length
        # "VALID_BATCH_SIZE": 24,
        "TRAIN_BATCH_SIZE": 8,  # For t5-large
        "VALID_BATCH_SIZE": 8,
        "TEST_BATCH_SIZE": 1,  # validation batch size
        "TRAIN_EPOCHS": 30,  # number of training epochs
        "VAL_EPOCHS": 1,  # number of validation epochs
        "TEST_EPOCHS": 1,  # number of validation epochs
        "LEARNING_RATE": aux_lr,  # learning rate
        # "MAX_SOURCE_TEXT_LENGTH": 256,  # max length of source text
        "MAX_SOURCE_TEXT_LENGTH": 512,  # max length of source text. Use 512 for Squad as long inputs.
        "MAX_TARGET_TEXT_LENGTH": 16,  # max length of target text
        "early_stopping_patience": 10,  # number of epochs before stopping training.
        "SEED": seed  # to use for randomisations
    }

    print("AUX Params: " + str(model_params_aux))

    print("Reading Aux Data...")

    training_data_aux, validation_data_aux, testing_data_aux = read_dpr_data_for_testing()

    torch.manual_seed(seed)  # pytorch random seed
    np.random.seed(seed)  # numpy random seed

    print("[Training Aux model]")

    tokenizer = T5Tokenizer.from_pretrained(MODEL_PATH_TO_TEST)

    training_set_aux, val_set_aux, test_set_aux = build_data_for_aux_task(
        [training_data_aux, validation_data_aux, testing_data_aux], aux_processor.DPR, model_params_aux)
    training_loader_aux, validation_loader_aux, test_loader_aux, tokenizer = \
        get_data_loaders(model_params_aux, SOURCE_TEXT, TARGET_TEXT, tokenizer, training_set_aux, val_set_aux,
                         test_set_aux)
    T5Trainer(training_loader_aux, validation_loader_aux, tokenizer, model_params=model_params_aux,
              local_model=MODEL_PATH_TO_TEST, task=aux_processor.DPR)

    prediction_file_name = PREDICTION_FILE_NAME
    predictions_filepath = '{}/{}'.format(DPR_MODEL_DIRECTORY, prediction_file_name)

    print("Calculating TEST SCORE: ")
    T5Generator(test_loader_aux, model_params=model_params_aux, output_file=prediction_file_name)
    test_accuracy = evaluate_e2e_tbsa.evaluate_exact_match_for_columns(predictions_filepath)

    return 0, test_accuracy


if __name__ == '__main__':

    if not os.path.exists(MODEL_PATH_TO_TEST):
        os.makedirs(MODEL_PATH_TO_TEST)

    if not os.path.exists(DPR_MODEL_DIRECTORY):
        os.makedirs(DPR_MODEL_DIRECTORY)

    ACC_VAL = {}
    ACC_TEST = {}
    for seed in SEEDS:
        val_list = []
        test_list = []
        # for lr_idx, lr in enumerate(LR_LIST):
        print("Running ALSC program...")
        # acc_val, acc_test = run_program_for_seed_lr(seed, lr, lr_idx)
        acc_val, acc_test = run_program_for_seed_lr(seed)
        print("Result_Val: {}".format(acc_val))
        print("Result_Test: {}".format(acc_test))
        val_list.append(acc_val)
        test_list.append(acc_test)
        print("LR Results for Validation: {}".format(str(val_list)))
        print("LR Results for Test: {}".format(str(test_list)))
        ACC_VAL[seed] = val_list
        ACC_TEST[seed] = test_list
        print("Seed = {} -> Done!".format(seed))
        print("Cumulative Results for Validation: {}".format(str(ACC_VAL)))
        print("Cumulative Results for Test: {} \n --------------------- ".format(str(ACC_TEST)))
