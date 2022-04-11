import torch
import sys
import os
import numpy as np
import pandas as pd
from rich.console import Console
from torch.utils.data import DataLoader
from transformers import T5Tokenizer
from torch import cuda

import e2e_tbsa_preprocess
import evaluate_e2e_tbsa
from train_generative import T5Trainer, T5Generator

from aux_processor import get_renamed_absa_columns, get_renamed_squad_columns, get_renamed_lm_columns, \
    get_renamed_commongen_columns, get_renamed_cosmos_columns, ABSA, SQUAD, COSMOS, WIKITEXT, COMMONGEN
from aux_processor import read_squad_data, read_wikitext_data, read_cosmos_data, read_commongen_data
from aux_processor import TARGET_TEXT, SOURCE_TEXT
from train_generative import YourDataSetClass

RENAMED_DF_FOR_TRAIN = {
    COSMOS: get_renamed_cosmos_columns,
    COMMONGEN: get_renamed_commongen_columns,
    ABSA: get_renamed_absa_columns,
    SQUAD: get_renamed_squad_columns,
    WIKITEXT: get_renamed_lm_columns
}

ABSA_PROMPT = "aspect analysis: "

FRACTION = 0.1
ABSA_MULTIPLIER = 2

# COMMONGEN_FRACTION_LIST = [0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.4, 0.5, 1]
# ABSA_MULTIPLIER_LIST = [0.1, 0.2, 0.5, 1, 2, 4, 8, 16]
# ABSA_MULTIPLIER_LIST = [1, 2, 4, 8, 16]
ABSA_MULTIPLIER_LIST = [1, 2, 4, 8]
# ABSA_MULTIPLIER_LIST = [1]

# COMMONGEN_FRACTION_LIST = [0.5, 1]
# ABSA_MULTIPLIER_LIST = [0.5, 1, 1.5, 2, 2.5, 4, 8, 16]

# sys.argv[1] will have the commonsense fraction
COMMONSENSE_FRACTION = float(sys.argv[1])
AUX_FRACTION = float(sys.argv[1])

if len(sys.argv) == 2:
    TASK = COMMONGEN
else:
    TASK = sys.argv[2]

print("TASK: {}".format(TASK))

if len(sys.argv) == 5:
    ABSA_FRACTION = float(sys.argv[3])
    SEED = int(sys.argv[4])
else:
    ABSA_FRACTION = None
    SEED = None

print("ABSA Fraction: {}".format(ABSA_FRACTION))
print("SEED: {}".format(SEED))

# COMMONSENSE_FRACTION = 0

# MODEL_DIRECTORY = 'models/dataset5_randomised2_test_mams_train_cs_{}'.format(COMMONSENSE_FRACTION)
# MODEL_DIRECTORY = 'models/dataset6_randomised_test_mams_train_cs_{}'.format(COMMONSENSE_FRACTION)
MODEL_DIRECTORY = 'models/{}_dataset8_manual_alsc_exp_mtl_aux_{}'.format(TASK, AUX_FRACTION)
# MODEL_DIRECTORY = 'models/{}_exp_regular'.format(TASK, AUX_FRACTION)

EXPERIMENT_OUTPUT_FILE_TARGET = '{}/output_targets.csv'.format(MODEL_DIRECTORY)
EXPERIMENT_OUTPUT_FILE_SENTIMENT = '{}/output_sentiment.csv'.format(MODEL_DIRECTORY)

PREDICTION_FILE_NAME = 'evaluation_predictions.csv'
PREDICTION_FILE_NAME_VAL = 'evaluation_predictions_val.csv'

TRANSFORMED_TARGETS_PREDICTIONS_FILE_NAME = 'transformed-targets.csv'
TRANSFORMED_SENTIMENTS_PREDICTIONS_FILE_NAME = 'transformed-sentiments.csv'

console = Console(record=True)

# SEEDS = [5, 6, 7, 8, 9]
# SEEDS = [0, 1, 2]
SEEDS = [0, 1]
LR_LIST = [1e-3, 5e-4, 1e-4]

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Setting up the device for GPU usage
device = 'cuda' if cuda.is_available() else 'cpu'


# MAX_VALIDATION_SET_FRACTION = {
#     COSMOS: 0.2,
#     COMMONGEN: 0.1,
#     ABSA: 1,
#     SQUAD: 0.1,
#     WIKITEXT: 0.5
# }


def build_data_for_absa(model_params, dataframes):
    # tokenzier for encoding the text
    try:
        tokenizer = T5Tokenizer.from_pretrained(model_params["MODEL"])
    except ValueError:
        print("Loading tokenizer locally due to Connection Error...")
        tokenizer = T5Tokenizer.from_pretrained(model_params["MODEL_LOCAL"])

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


def merge_absa_with_aux(dataset_absa, dataset_aux, model_params, absa_multiplier=ABSA_MULTIPLIER,
                        aux_fraction=FRACTION):
    ### Try different sample sizes of commongen dataset
    print("ABSA Multiplier is = {}, AUX fraction is = {}".format(absa_multiplier, aux_fraction))

    dataset_aux = dataset_aux.sample(frac=aux_fraction,
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

    print("ABSA length is... {}, AUX length is... {}".format(len(dataset_absa), len(dataset_aux)))

    if aux_fraction > 0:
        print("[Using the AUX dataset]\n\n")
        return pd.concat([dataset_absa, dataset_aux], ignore_index=True)
    else:
        print("[Ignoring AUX dataset completely]\n\n")
        return pd.concat([dataset_absa], ignore_index=True)


def build_merged_data_for_aux_task(dataframes, training_dataset_absa, val_dataset_absa, test_dataset_absa,
                                   tokenizer, absa_multiplier, aux_fraction, task_name, model_params):
    console.log("[Data]: Reading {} data...\n".format(task_name))

    # Creation of Dataset and Dataloader
    train_dataset_aux = dataframes[0].sample(frac=1, random_state=model_params['SEED']).reset_index(drop=True)
    val_dataset_aux = dataframes[1].reset_index(drop=True)

    column_renamer = RENAMED_DF_FOR_TRAIN[task_name]
    train_dataset_aux = column_renamer(train_dataset_aux)
    val_dataset_aux = column_renamer(val_dataset_aux)

    console.print(f"TRAIN Dataset {task_name}: {train_dataset_aux.shape}")
    console.print(f"VALIDATION Dataset {task_name}: {val_dataset_aux.shape}")

    training_set = merge_absa_with_aux(training_dataset_absa, train_dataset_aux, model_params, absa_multiplier,
                                       aux_fraction)
    # val_set = merge_absa_with_aux(val_dataset_absa, val_dataset_aux, model_params, 1,
    #                               min(aux_fraction, MAX_VALIDATION_SET_FRACTION[task_name]))
    val_set = merge_absa_with_aux(val_dataset_absa, val_dataset_aux, model_params, 1, aux_fraction)
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
    test_params = {'batch_size': model_params["TEST_BATCH_SIZE"], 'shuffle': False, 'num_workers': 2}
    # Creation of Dataloaders for testing and validation. This will be used down for training and validation stage for the model.
    training_loader = DataLoader(training_set, **train_params)
    validation_loader = DataLoader(val_set, **val_params)
    test_loader = DataLoader(test_set, **test_params)

    return training_loader, validation_loader, test_loader, tokenizer


def run_program_for_seed(seed, results_sentiment, lr):
    # Set random seeds and deterministic pytorch for reproducibility
    torch.manual_seed(seed)  # pytorch random seed
    np.random.seed(seed)  # numpy random seed

    model_params = {
        "OUTPUT_PATH": MODEL_DIRECTORY,  # output path
        "MODEL": "t5-base",
        "MODEL_LOCAL": "/home/mullick/lm_models/t5-base-conditional-gen",
        # "MODEL": "danny911kr/calm-base",
        "TRAIN_BATCH_SIZE": 24,  # training batch size. 32 takes 22-23GB GPU memory, 24 takes 20GB GPU.
        "VALID_BATCH_SIZE": 24,  # validation batch size
        "TEST_BATCH_SIZE": 1,  # validation batch size
        "TRAIN_EPOCHS": 10,  # number of training epochs
        "VAL_EPOCHS": 1,  # number of validation epochs
        "TEST_EPOCHS": 1,  # number of validation epochs
        "LEARNING_RATE": lr,  # learning rate
        "MAX_SOURCE_TEXT_LENGTH": 512,  # max length of source text
        "MAX_TARGET_TEXT_LENGTH": 32,  # max length of target text
        "early_stopping_patience": 3,  # number of epochs before stopping training.
        "SEED": seed  # to use for randomisations
    }

    print(model_params)

    training_file_absa = './data/merged_train_alsc.csv'
    validation_file_absa = './data/merged_val_alsc.csv'
    test_file_absa = 'data/merged_test_ambiguous_alsc_manual.csv'

    print("Training on: {}, Validation on {}, Testing on: {}, Seed: {}".format(training_file_absa, validation_file_absa,
                                                                               test_file_absa, seed))

    training_absa = pd.read_csv(training_file_absa)
    validation_absa = pd.read_csv(validation_file_absa)
    test_absa = pd.read_csv(test_file_absa)

    training_set_absa, val_set_absa, test_set_absa, tokenizer \
        = build_data_for_absa(model_params, dataframes=[training_absa, validation_absa, test_absa])

    print("Reading Aux Data...")

    if TASK == COMMONGEN:
        training_data_aux, validation_data_aux, testing_data_aux = read_commongen_data()
    elif TASK == COSMOS:
        training_data_aux, validation_data_aux, testing_data_aux = read_cosmos_data()
    elif TASK == SQUAD:
        training_data_aux, validation_data_aux, testing_data_aux = read_squad_data()
    elif TASK == WIKITEXT:
        training_data_aux, validation_data_aux, testing_data_aux = read_wikitext_data()

    for absa_multiplier in ABSA_MULTIPLIER_LIST:

        torch.manual_seed(seed)  # pytorch random seed
        np.random.seed(seed)  # numpy random seed

        tokenizer, training_set, val_set, test_set = build_merged_data_for_aux_task(
            [training_data_aux, validation_data_aux, testing_data_aux],
            training_set_absa, val_set_absa, test_set_absa, tokenizer, absa_multiplier,
            AUX_FRACTION, TASK, model_params)

        training_loader, validation_loader, test_loader, tokenizer = \
            get_data_loaders(model_params, SOURCE_TEXT, TARGET_TEXT, tokenizer, training_set, val_set, test_set)

        T5Trainer(training_loader, validation_loader, tokenizer, model_params=model_params, local_model=None, task=None)

        prediction_file_name_validation = PREDICTION_FILE_NAME_VAL
        predictions_filepath_validation = '{}/{}'.format(MODEL_DIRECTORY, prediction_file_name_validation)
        prediction_file_name = PREDICTION_FILE_NAME
        predictions_filepath = '{}/{}'.format(MODEL_DIRECTORY, prediction_file_name)

        ### Test loader is only ABSA
        print("Calculating VALIDATION SCORE: ")
        T5Generator(validation_loader, model_params=model_params, output_file=prediction_file_name_validation)
        validation_accuracy = evaluate_e2e_tbsa.evaluate_exact_match_for_columns(predictions_filepath_validation)
        print(f"VALIDATION SCORE is {validation_accuracy}")

        print("Calculating TEST SCORE: ")
        T5Generator(test_loader, model_params=model_params, output_file=prediction_file_name)
        test_accuracy = evaluate_e2e_tbsa.evaluate_exact_match_for_columns(predictions_filepath)

        if absa_multiplier in results_sentiment.keys():
            results_sentiment[absa_multiplier].append(test_accuracy)
        else:
            results_sentiment[absa_multiplier] = [test_accuracy]

        print("Results sentiment: " + str(results_sentiment))


def write_to_results_file(results_dict, output_file, fraction=COMMONSENSE_FRACTION):
    f1_list = results_dict[absa_multiplier]
    output_file.write('{}, {}, '.format(absa_multiplier, fraction))
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

    results_sentiment = {}

    if SEED is not None:
        SEEDS = [SEED]

    if ABSA_FRACTION is not None:
        ABSA_MULTIPLIER_LIST = [ABSA_FRACTION]

    for seed in SEEDS:
        for lr in LR_LIST:
            for lr_idx, lr in enumerate(LR_LIST):
                run_program_for_seed(seed, results_sentiment, lr)

    with open(EXPERIMENT_OUTPUT_FILE_SENTIMENT, 'a') as file_sentiments:
        file_sentiments.write("Seeds used: {}\n".format(SEEDS))
        for absa_multiplier in results_sentiment.keys():
            write_to_results_file(results_sentiment, file_sentiments, AUX_FRACTION)

    print("Done!")
