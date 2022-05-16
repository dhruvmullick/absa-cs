import os
import sys

import numpy as np
import pandas as pd
import torch
from rich.console import Console
from torch import cuda
from torch.utils.data import DataLoader
from transformers import BartTokenizer

import evaluate_e2e_tbsa
from aux_processor import RENAMED_DF_FOR_TRAIN, get_renamed_absa_columns, read_aux_data
from aux_processor import SQUAD, COSMOS, WIKITEXT, COMMONGEN, DPR, QQP, WMTFR, WMTDE, BOOK, WIKIJUMBLED
from aux_processor import TARGET_TEXT, SOURCE_TEXT, ABSA_PROMPT
from train_generative_bart import BartTrainer, BartGenerator, YourDataSetClass

DELTA = 0.001

# SEEDS = [0, 1, 2, 3, 4]
# SEEDS = [0, 1, 2]
SEEDS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
# SEEDS = [10, 11, 12, 13, 14]
# SEEDS = [5, 6, 7, 8, 9]
# LR_LIST = [5e-4]
# LR_LIST = [5e-4, 1e-4, 5e-5]

# sys.argv[1] will have the commonsense fraction
COMMONSENSE_FRACTION = float(sys.argv[1])
AUX_FRACTION = float(sys.argv[1])
# AUX_COUNT = float(sys.argv[1])

# TRAIN_AUX_ONLY = True
TRAIN_AUX_ONLY = False

if len(sys.argv) == 2:
    TASK = COMMONGEN
else:
    TASK = sys.argv[2]

print("TASK: {}".format(TASK))

if len(sys.argv) == 4:
    assert sys.argv[3][0] == '[' and sys.argv[3][-1] == ']'
    SEEDS = sys.argv[3][1:-1]
    SEEDS = SEEDS.split(sep='_')
    SEEDS = [int(x) for x in SEEDS]

print("SEEDS: {}".format(SEEDS))

# AUX_COUNT = int(AUX_COUNT)
# print(f"USING FIXED COUNT FOR AUX... {AUX_COUNT}")

# MODEL_DIRECTORY = 'models/{}_dataset8_manual_replaced1_alsc_fine_tune_f1_aux_{}'.format(TASK, AUX_FRACTION)
# MODEL_DIRECTORY = 'models/{}_dataset9_alsc_fine_tune_f1_aux_{}'.format(TASK, AUX_FRACTION)
MODEL_DIRECTORY = 'models/{}_dataset8_manual_alsc_fine_tune_greedy_bart_aux_{}'.format(TASK, AUX_FRACTION)
MODEL_DIRECTORY_ABSA = '{}/absa/'.format(MODEL_DIRECTORY)

RESULTS_FILE_PATH = '{}/results.csv'.format(MODEL_DIRECTORY)

PREDICTION_FILE_NAME = 'evaluation_predictions.csv'
PREDICTION_FILE_NAME_VAL = 'evaluation_predictions_val.csv'

console = Console(record=True)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Setting up the device for GPU usage
device = 'cuda' if cuda.is_available() else 'cpu'


def build_data_for_absa(model_params, dataframes):
    # tokenzier for encoding the text
    try:
        tokenizer = BartTokenizer.from_pretrained(model_params["MODEL"])
    except ValueError:
        print("Loading tokenizer locally due to Connection Error...")
        tokenizer = BartTokenizer.from_pretrained(model_params["MODEL_LOCAL"])

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


def build_data_for_aux_task(dataframes, task_name, model_params):
    console.log("[Data]: Reading {} data...\n".format(task_name))

    train_dataset_aux = dataframes[0].sample(frac=AUX_FRACTION, random_state=model_params['SEED']).reset_index(
        drop=True)
    # train_dataset_aux = dataframes[0].sample(n=AUX_COUNT, random_state=model_params['SEED']).reset_index(
    #     drop=True)

    # val_dataset_aux = dataframes[1].sample(frac=max(AUX_FRACTION, 0.1), random_state=model_params['SEED']).reset_index(drop=True)
    val_dataset_aux = dataframes[1].sample(frac=1.0, random_state=model_params['SEED']).reset_index(drop=True)

    column_renamer = RENAMED_DF_FOR_TRAIN[task_name]
    train_dataset_aux = column_renamer(train_dataset_aux)
    val_dataset_aux = column_renamer(val_dataset_aux)

    console.print(f"TRAIN Dataset {task_name}: {train_dataset_aux.shape}")
    console.print(f"VALIDATION Dataset {task_name}: {val_dataset_aux.shape}")

    return train_dataset_aux, val_dataset_aux, None


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


def get_aux_lr(task):
    if (task == SQUAD and (abs(AUX_FRACTION - 1.0) <= DELTA)) \
            or (task == WMTDE and (abs(AUX_FRACTION - 1.0) <= DELTA)) \
            or (task == WMTFR and (abs(AUX_FRACTION - 1.0) <= DELTA)) \
            or (task == WIKIJUMBLED and (abs(AUX_FRACTION - 1.0) <= DELTA)) \
            or (task == WIKIJUMBLED and (abs(AUX_FRACTION - 0.5) <= DELTA)):
        return 5e-5
    elif task == DPR and (abs(AUX_FRACTION - 0.1) <= DELTA):
        return 1e-3
    elif task == DPR and (abs(AUX_FRACTION - 1.0) <= DELTA) \
            or (task == BOOK and (abs(AUX_FRACTION - 0.2) <= DELTA)) \
            or (task == BOOK and (abs(AUX_FRACTION - 0.5) <= DELTA)):
        return 5e-4
    elif task in [COMMONGEN, COSMOS, WIKITEXT, SQUAD, DPR, QQP, WIKIJUMBLED, WMTFR, WMTDE, BOOK]:
        return 1e-4
    else:
        raise AssertionError(f"AUX LR NOT DEFINED FOR TASK - {task}")


def get_absa_lr(task):
    if (task == COMMONGEN and (abs(AUX_FRACTION - 0.1) <= DELTA)) \
            or (task == WIKIJUMBLED and (abs(AUX_FRACTION - 0.5) <= DELTA)) \
            or (task == WMTDE and (abs(AUX_FRACTION - 0.2) <= DELTA)) \
            or (task == QQP and (abs(AUX_FRACTION - 0.5) <= DELTA)):
        return 1e-4
    elif task in [COMMONGEN, COSMOS, WIKITEXT, SQUAD, DPR, QQP, WIKIJUMBLED, WMTFR, WMTDE, BOOK]:
        return 5e-4
    else:
        raise AssertionError(f"ABSA LR NOT DEFINED FOR TASK - {task}")


# Hyperparameter Check Code
# def run_program_for_seed_lr(seed, lr, lr_idx):
def run_program_for_seed_lr(seed):
    # Set random seeds and deterministic pytorch for reproducibility
    torch.manual_seed(seed)  # pytorch random seed
    np.random.seed(seed)  # numpy random seed

    if not TRAIN_AUX_ONLY:
        print("Training both Aux and ABSA")
        print("USING FIXED LR FOR AUX... ")
        aux_lr = get_aux_lr(TASK)
    else:
        # Hyperparameter Check Code
        print("Training Aux only")
        # aux_lr = lr
        raise AssertionError("NEED TO DEFINE WHAT TO DO IF ONLY TRAINING AUX...")

    # Hyperparameter Check Code
    absa_lr = get_absa_lr(TASK)
    # absa_lr = lr

    model_params_aux = {
        "OUTPUT_PATH": MODEL_DIRECTORY,  # output path
        "MODEL": "facebook/bart-base",
        "MODEL_LOCAL": "/home/mullick/lm_models/bart-base-conditional-gen",
        # "MODEL": "danny911kr/calm-base",
        # "TRAIN_BATCH_SIZE": 32,  # training batch size. 32 takes 22-23GB GPU memory, 24 takes 20GB GPU (host1), 8 takes 10GB (host2/3)
        # "VALID_BATCH_SIZE": 32,  # validation batch size
        "TRAIN_BATCH_SIZE": 32,
        "VALID_BATCH_SIZE": 32,
        "TEST_BATCH_SIZE": 1,  # validation batch size
        "TRAIN_EPOCHS": 30,  # number of training epochs
        "VAL_EPOCHS": 1,  # number of validation epochs
        "TEST_EPOCHS": 1,  # number of validation epochs
        "LEARNING_RATE": aux_lr,  # learning rate
        "MAX_SOURCE_TEXT_LENGTH": 512,  # max length of source text. Use 512 for Squad as long inputs.
        "MAX_TARGET_TEXT_LENGTH": 32,  # max length of target text.
        "early_stopping_patience": 3,  # number of epochs before stopping training.
        "SEED": seed  # to use for randomisations
    }

    model_params_absa = {
        "OUTPUT_PATH": MODEL_DIRECTORY_ABSA,  # output path
        "MODEL": "facebook/bart-base",
        "MODEL_LOCAL": "/home/mullick/lm_models/bart-base-conditional-gen",
        # "MODEL": "danny911kr/calm-base",
        "TRAIN_BATCH_SIZE": 32, # training batch size. (if Source=256, Target=32) 32 takes 22-23GB GPU memory, 24 takes 20GB GPU (host1), 8 takes 10GB (host2/3)
        "VALID_BATCH_SIZE": 32,  # validation batch size
        "TEST_BATCH_SIZE": 1,  # validation batch size
        "TRAIN_EPOCHS": 30,  # number of training epochs
        "VAL_EPOCHS": 1,  # number of validation epochs
        "TEST_EPOCHS": 1,  # number of validation epochs
        "LEARNING_RATE": absa_lr,  # learning rate
        # "MAX_SOURCE_TEXT_LENGTH": 256,  # max length of source text
        "MAX_SOURCE_TEXT_LENGTH": 256,  # max length of source text. Use 512 for Squad as long inputs.
        "MAX_TARGET_TEXT_LENGTH": 64,  # max length of target text
        "early_stopping_patience": 3,  # number of epochs before stopping training.
        # "early_stopping_patience": 10,  # number of epochs before stopping training.
        "SEED": seed  # to use for randomisations
    }

    print("ABSA Params: " + str(model_params_absa))
    print("AUX Params: " + str(model_params_aux))

    training_file_absa = './data/merged_train_alsc.csv'
    validation_file_absa = './data/merged_val_alsc.csv'
    test_file_absa = 'data/merged_test_ambiguous_alsc_manual.csv'

    # print("USING PLAIN DATA...")
    # training_file_absa = './data/merged_train_plain_alsc.csv'
    # validation_file_absa = './data/merged_val_plain_alsc.csv'
    # test_file_absa = 'data/merged_test_plain_alsc.csv'

    print("Training on: {}, Validation on {}, Testing on: {}, Seed: {}".format(training_file_absa, validation_file_absa,
                                                                               test_file_absa, seed))
    print("ABSA Prompt is: {}".format(ABSA_PROMPT))
    training_absa = pd.read_csv(training_file_absa)
    validation_absa = pd.read_csv(validation_file_absa)
    test_absa = pd.read_csv(test_file_absa)

    training_set_absa, val_set_absa, test_set_absa, tokenizer \
        = build_data_for_absa(model_params_absa, dataframes=[training_absa, validation_absa, test_absa])

    print("Reading Aux Data...")

    training_data_aux, validation_data_aux, testing_data_aux = read_aux_data(TASK, seed)

    torch.manual_seed(seed)  # pytorch random seed
    np.random.seed(seed)  # numpy random seed

    # Hyperparameter Check Code
    # if TRAIN_AUX_ONLY or lr_idx == 0:
    if not TRAIN_AUX_ONLY:
        # Don't Train both Aux and ABSA and also it's not the first lr.
        train_aux = True
        print("Will train Aux model\n\n")

    else:
        # Train both Aux and ABSA and also it's not the first lr so Aux training not needed.
        # Use existing trained aux model
        train_aux = False
        print("Will not train Aux model\n\n")

    if AUX_FRACTION > 0.0:
        if train_aux:
            print("[Training Aux model]")
            training_set_aux, val_set_aux, test_set_aux = build_data_for_aux_task(
                [training_data_aux, validation_data_aux, testing_data_aux], TASK, model_params_aux)
            training_loader_aux, validation_loader_aux, test_loader_aux, tokenizer = \
                get_data_loaders(model_params_aux, SOURCE_TEXT, TARGET_TEXT, tokenizer, training_set_aux, val_set_aux,
                                 test_set_aux)
            BartTrainer(training_loader_aux, validation_loader_aux, tokenizer, model_params=model_params_aux,
                        local_model=None, task=TASK)
        else:
            print("[Reulsing Aux model]")
        aux_model_path = os.path.join(model_params_aux["OUTPUT_PATH"], "model_files")
    else:
        print("[Not using Aux]")
        aux_model_path = None

    console.log("\n\nWorking on ABSA part now... ")

    torch.manual_seed(seed)  # pytorch random seed
    np.random.seed(seed)  # numpy random seed

    if TRAIN_AUX_ONLY:
        # ABSA part is not needed.
        console.log("NOT GENERATING ON ABSA RIGHT NOW...")
        return 0, 0

    training_loader_absa, validation_loader_absa, test_loader_absa, tokenizer = \
        get_data_loaders(model_params_absa, SOURCE_TEXT, TARGET_TEXT, tokenizer, training_set_absa, val_set_absa,
                         test_set_absa)
    BartTrainer(training_loader_absa, validation_loader_absa, tokenizer, model_params=model_params_absa,
                local_model=aux_model_path)

    # T5Trainer(training_loader_absa, validation_loader_absa, tokenizer, model_params=model_params_absa,
    #           local_model=aux_model_path, test_loader=test_loader_absa, aux_task=TASK, results_file=RESULTS_FILE)

    prediction_file_name_validation = PREDICTION_FILE_NAME_VAL
    predictions_filepath_validation = '{}/{}'.format(MODEL_DIRECTORY_ABSA, prediction_file_name_validation)
    prediction_file_name = PREDICTION_FILE_NAME
    predictions_filepath = '{}/{}'.format(MODEL_DIRECTORY_ABSA, prediction_file_name)

    ### Test loader is only ABSA
    print("Calculating VALIDATION SCORE: ")
    BartGenerator(validation_loader_absa, model_params=model_params_absa, output_file=prediction_file_name_validation)
    validation_accuracy = evaluate_e2e_tbsa.evaluate_exact_match_for_columns(predictions_filepath_validation)

    print("Calculating TEST SCORE: ")
    BartGenerator(test_loader_absa, model_params=model_params_absa, output_file=prediction_file_name)
    test_accuracy = evaluate_e2e_tbsa.evaluate_exact_match_for_columns(predictions_filepath)

    return validation_accuracy, test_accuracy


if __name__ == '__main__':

    if not os.path.exists(MODEL_DIRECTORY):
        os.makedirs(MODEL_DIRECTORY)

    if not os.path.exists(MODEL_DIRECTORY_ABSA):
        os.makedirs(MODEL_DIRECTORY_ABSA)

    ACC_VAL = {}
    ACC_TEST = {}
    for seed in SEEDS:
        val_list = []
        test_list = []
        # Hyperparameter Check Code
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
