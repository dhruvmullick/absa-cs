# import json
# import sys
# import os, time, datetime
# from rich.table import Column, Table
# from rich import box
# from rich.console import Console
# from shutil import copyfile
#
# from tqdm import tqdm
#
# import torch
# import numpy as np
# import pandas as pd
# from rich.console import Console
# from sklearn.model_selection import train_test_split
# from torch.utils.data import DataLoader
# from transformers import T5Tokenizer, T5ForConditionalGeneration
# from torch import cuda
#
# import e2e_tbsa_preprocess
# import evaluate_e2e_tbsa
# from train_generative import T5Generator
#
# # define a rich console logger
# from train_generative import YourDataSetClass
# from utils import EarlyStopping
#
# COMMONGEN_PROMPT = 'generate a sentence with: '
# ABSA_PROMPT = "aspect analysis: "
#
# # COMMONSENSE_LOSS_REGULARISER_LIST = [0, 0.001, 0.01, 0.1, 0.5]
# COMMONSENSE_LOSS_REGULARISER = float(sys.argv[1])
#
# MODEL_DIRECTORY = 'models/mtl_dataset5_test_mams_train_param_{}'.format(COMMONSENSE_LOSS_REGULARISER)
# EXPERIMENT_OUTPUT_FILE_TARGET = '{}/output_targets.csv'.format(MODEL_DIRECTORY)
# EXPERIMENT_OUTPUT_FILE_SENTIMENT = '{}/output_sentiment.csv'.format(MODEL_DIRECTORY)
# PREDICTION_FILE_NAME = 'evaluation_commongen_predictions.csv'
# TRANSFORMED_TARGETS_PREDICTIONS_FILE_NAME = 'transformed-targets.csv'
# TRANSFORMED_SENTIMENTS_PREDICTIONS_FILE_NAME = 'transformed-sentiments.csv'
#
# SEEDS = [0, 1, 2]
#
#
# console = Console(record=True)
# # Set random seeds and deterministic pytorch for reproducibility
# torch.manual_seed(0)  # pytorch random seed
# np.random.seed(0)  # numpy random seed
# torch.backends.cudnn.deterministic = True
#
# # Setting up the device for GPU usage
# device = 'cuda' if cuda.is_available() else 'cpu'
#
#
# def process_concepts(concept_set):
#     return concept_set.replace("#", " ")
#
#
# def read_json_file(file_name):
#     concept_sets = []
#     scenes = []
#     with open(file_name, 'r') as file:
#         for line in file:
#             json_line = json.loads(line)
#             concept_set = process_concepts(json_line['concept_set'])
#             for scene in json_line['scene']:
#                 concept_sets.append(concept_set)
#                 scenes.append(scene)
#     return pd.DataFrame(data={'scene': scenes, 'concept_set': concept_sets})
#
#
# def get_renamed_commongen_columns(df):
#     df = df.rename(columns={"scene": "target", "concept_set": "source"})[['target', 'source']]
#     return df
#
#
# def get_renamed_absa_columns(df):
#     df = df.rename(columns={"sentences_opinions": "target", "sentences_texts": "source"})[['target', 'source']]
#     return df
#
#
# def get_data_for_absa(model_params, dataframes):
#
#     # logging
#     console.log(f"[Data]: Reading ABSA data...\n")
#
#     # Creation of Dataset and Dataloader
#     train_dataset = dataframes[0].sample(frac=1, random_state=model_params['SEED']).reset_index(drop=True)
#     val_dataset = dataframes[1].reset_index(drop=True)
#     test_dataset = dataframes[2].reset_index(drop=True)
#     train_dataset['sentences_texts'] = ABSA_PROMPT + train_dataset['sentences_texts']
#     val_dataset['sentences_texts'] = ABSA_PROMPT + val_dataset['sentences_texts']
#     test_dataset['sentences_texts'] = ABSA_PROMPT + test_dataset['sentences_texts']
#     train_dataset = get_renamed_absa_columns(train_dataset)
#     val_dataset = get_renamed_absa_columns(val_dataset)
#     test_dataset = get_renamed_absa_columns(test_dataset)
#
#     console.print(f"TRAIN Dataset ABSA: {train_dataset.shape}")
#     console.print(f"VALIDATION Dataset ABSA: {val_dataset.shape}")
#     console.print(f"TEST Dataset ABSA: {test_dataset.shape}\n")
#
#     return train_dataset, val_dataset, test_dataset
#
#
# def get_data_for_commongen(dataframes, model_params):
#     console.log(f"[Data]: Reading CommonGen data...\n")
#
#     # Creation of Dataset and Dataloader
#     train_dataset_commongen = dataframes[0].sample(frac=1, random_state=model_params['SEED']).reset_index(drop=True)
#     val_dataset_commongen = dataframes[1].reset_index(drop=True)
#     train_dataset_commongen['concept_set'] = COMMONGEN_PROMPT + train_dataset_commongen['concept_set']
#     val_dataset_commongen['concept_set'] = COMMONGEN_PROMPT + val_dataset_commongen['concept_set']
#     train_dataset_commongen = get_renamed_commongen_columns(train_dataset_commongen)
#     val_dataset_commongen = get_renamed_commongen_columns(val_dataset_commongen)
#
#     console.print(f"TRAIN Dataset CommonGen: {train_dataset_commongen.shape}")
#     console.print(f"VALIDATION Dataset CommonGen: {val_dataset_commongen.shape}")
#
#     return train_dataset_commongen, val_dataset_commongen
#
#
# def build_data(source_text, target_text, training_dataset, val_dataset, test_dataset, tokenizer, model_params):
#
#     # Creating the Training and Validation dataset for further creation of Dataloader
#     training_set = YourDataSetClass(training_dataset, tokenizer, model_params["MAX_SOURCE_TEXT_LENGTH"],
#                                     model_params["MAX_TARGET_TEXT_LENGTH"], source_text, target_text)
#     val_set = YourDataSetClass(val_dataset, tokenizer, model_params["MAX_SOURCE_TEXT_LENGTH"],
#                                model_params["MAX_TARGET_TEXT_LENGTH"], source_text, target_text)
#     test_set = YourDataSetClass(test_dataset, tokenizer, model_params["MAX_SOURCE_TEXT_LENGTH"],
#                                 model_params["MAX_TARGET_TEXT_LENGTH"], source_text, target_text)
#
#     # Defining the parameters for creation of dataloaders
#     train_params = {'batch_size': model_params["TRAIN_BATCH_SIZE"], 'shuffle': True, 'num_workers': 2}
#     val_params = {'batch_size': model_params["VALID_BATCH_SIZE"], 'shuffle': False, 'num_workers': 2}
#     test_params = {'batch_size': model_params["VALID_BATCH_SIZE"], 'shuffle': False, 'num_workers': 2}
#
#     # Creation of Dataloaders for testing and validation. This will be used down for training and validation stage for the model.
#     training_loader = DataLoader(training_set, **train_params)
#     validation_loader = DataLoader(val_set, **val_params)
#     test_loader = DataLoader(test_set, **test_params)
#
#     return training_loader, validation_loader, test_loader, tokenizer
#
#
# def run_program_for_seed(seed, results_target, results_sentiment):
#     # Set random seeds and deterministic pytorch for reproducibility
#     torch.manual_seed(seed)  # pytorch random seed
#     np.random.seed(seed)  # numpy random seed
#
#     model_params = {
#         "OUTPUT_PATH": MODEL_DIRECTORY,  # output path
#         "MODEL": "t5-base",
#         "TRAIN_BATCH_SIZE": 32,  # training batch size
#         "VALID_BATCH_SIZE": 32,  # validation batch size
#         "TRAIN_EPOCHS": 10,  # number of training epochs
#         "VAL_EPOCHS": 1,  # number of validation epochs
#         "LEARNING_RATE": 5e-4,  # learning rate
#         "MAX_SOURCE_TEXT_LENGTH": 256,  # max length of source text
#         "MAX_TARGET_TEXT_LENGTH": 64,  # max length of target text
#         "early_stopping_patience": 5,  # number of epochs before stopping training.
#         "SEED": seed  # to use for randomisation
#     }
#
#     print(model_params)
#
#     # tokenzier for encoding the text
#     tokenizer = T5Tokenizer.from_pretrained(model_params["MODEL"])
#     tokenizer.add_tokens(['<sep>'])
#
#     test_file_absa = 'data/merged_test_ambiguous.csv'
#     training_file_absa = './data/processed_train_Mams_en.csv'
#     validation_file_absa = './data/processed_val_Mams_en.csv'
#     print("Training on: {}, Testing on: {}, Seed: {}".format(training_file_absa, test_file_absa, seed))
#     training_absa = pd.read_csv(training_file_absa)
#     validation_absa = pd.read_csv(validation_file_absa)
#     test_absa = pd.read_csv(test_file_absa)
#
#     training_set_absa, val_set_absa, test_set_absa \
#         = get_data_for_absa(model_params, dataframes=[training_absa, validation_absa, test_absa],
#                             source_text="source", target_text="target")
#
#     training_file_commongen = './data/commongen_data/commongen.train.jsonl'
#     test_file_commongen = './data/commongen_data/commongen.dev.jsonl'
#     training_data_commongen = read_json_file(training_file_commongen)
#     testing_data_commongen = read_json_file(test_file_commongen)
#
#     training_data_commongen, validation_data_commongen = train_test_split(training_data_commongen, test_size=0.1,
#                                                                           random_state=seed)
#
#     torch.manual_seed(seed)  # pytorch random seed
#     np.random.seed(seed)  # numpy random seed
#
#     source_text = "source"
#     target_text = "target"
#
#     training_loader_absa, validation_loader_absa, test_loader_absa, tokenizer \
#         = build_data(source_text, target_text, training_set_absa, val_set_absa, test_set_absa, tokenizer, model_params)
#
#     training_loader_cs, validation_loader_cs, test_loader_cs, tokenizer \
#         = build_data(source_text, target_text, training_data_commongen, validation_data_commongen, testing_data_commongen, tokenizer, model_params)
#
#     T5TrainerMultiLoss(training_loader_absa, validation_loader_absa, training_loader_cs, validation_loader_cs,
#                        tokenizer, model_params=model_params)
#
#     prediction_file_name = PREDICTION_FILE_NAME
#
#     ### Test loader is only ABSA
#     T5Generator(test_loader_absa, model_params=model_params, output_file=prediction_file_name)
#
#     e2e_tbsa_preprocess.run_from_generative_script(
#         predictions_filepath='{}/{}'.format(MODEL_DIRECTORY, prediction_file_name),
#         transformed_targets_filepath='{}/{}'.format(MODEL_DIRECTORY, TRANSFORMED_TARGETS_PREDICTIONS_FILE_NAME),
#         transformed_sentiments_filepath='{}/{}'.format(MODEL_DIRECTORY,
#                                                        TRANSFORMED_SENTIMENTS_PREDICTIONS_FILE_NAME))
#     output = evaluate_e2e_tbsa.run_from_generative_script(
#         target_file_to_evaluate='{}/{}'.format(MODEL_DIRECTORY, TRANSFORMED_TARGETS_PREDICTIONS_FILE_NAME),
#         sentiments_file_to_evaluate='{}/{}'.format(MODEL_DIRECTORY, TRANSFORMED_SENTIMENTS_PREDICTIONS_FILE_NAME))
#
#     if absa_multiplier in results_target.keys():
#         results_target[absa_multiplier].append(output['te'])
#         results_sentiment[absa_multiplier].append(output['tse'])
#     else:
#         results_target[absa_multiplier] = [output['te']]
#         results_sentiment[absa_multiplier] = [output['tse']]
#
#     print("Results target: " + str(results_target))
#     print("Results sentiment: " + str(results_sentiment))
#
#
# def write_to_results_file(results_dict, output_file):
#     f1_list = results_dict[absa_multiplier]
#     output_file.write('{}, {}, '.format(absa_multiplier, COMMONSENSE_LOSS_REGULARISER))
#     for f1 in f1_list:
#         output_file.write('{}, '.format(f1))
#     output_file.write('\n')
#
#
# def T5TrainerMultiLoss(training_loader_absa, validation_loader_absa, training_loader_cs, validation_loader_cs,
#                        tokenizer, model_params):
#     """
#     T5 trainer
#     """
#
#     # logging
#     console.log(f"""[Model]: Loading {model_params["MODEL"]}...\n""")
#
#     # Defining the model. We are using t5-base model and added a Language model layer on top for generation of Summary.
#     # Further this model is sent to device (GPU/TPU) for using the hardware.
#     model = T5ForConditionalGeneration.from_pretrained(model_params["MODEL"])
#     model = model.to(device)
#     # model.resize_token_embeddings(model_params['new_tokens_size'])
#
#     # Defining the optimizer that will be used to tune the weights of the network in the training session.
#     optimizer = torch.optim.AdamW(params=model.parameters(), lr=model_params["LEARNING_RATE"])
#     # optimizer = Adafactor(params = model.parameters(), relative_step=True, lr = model_params["LEARNING_RATE"])
#
#     # initialize the early_stopping object
#     early_stopping = EarlyStopping(patience=model_params["early_stopping_patience"], verbose=True,
#                                    path=f'{model_params["OUTPUT_PATH"]}/best_pytorch_model.bin')
#
#     training_logger = Table(Column("Epoch", justify="center"), Column("train_loss", justify="center"),
#                             Column("val_loss", justify="center"), Column("Epoch Time", justify="center"),
#                             title="Training Status", pad_edge=False, box=box.ASCII)
#
#     # Training loop
#     console.log(f'[Initiating Fine Tuning]...\n')
#     avg_train_losses = []
#     avg_valid_losses = []
#     for epoch in range(model_params["TRAIN_EPOCHS"]):
#         start_time = time.time()
#
#         #### CHANGE TRAIN AND VALIDATE TO ACCEPT BOTH ABSA AND CS DATA TO CALCULATE LOSS
#
#         train_losses = train(tokenizer, model, device, training_loader, optimizer)
#         valid_losses = validate(tokenizer, model, device, validation_loader)
#         epoch_time = round(time.time() - start_time)
#
#         # calculate average loss over an epoch
#         train_loss = np.average(train_losses)
#         valid_loss = np.average(valid_losses)
#         avg_train_losses.append(train_loss)
#         avg_valid_losses.append(valid_loss)
#
#         # preparing the processing time for the epoch and est. the total.
#         epoch_time_ = str(datetime.timedelta(seconds=epoch_time))
#         total_time_estimated_ = str(
#             datetime.timedelta(seconds=(epoch_time * (model_params["TRAIN_EPOCHS"] - epoch - 1))))
#         training_logger.add_row(f'{epoch + 1}/{model_params["TRAIN_EPOCHS"]}', f'{train_loss:.5f}', f'{valid_loss:.5f}',
#                                 f'{epoch_time_} (Total est. {total_time_estimated_})')
#         console.print(training_logger)
#
#         # early_stopping needs the validation loss to check if it has decresed,
#         # and if it has, it will make a checkpoint of the current model
#         early_stopping(valid_loss, model)
#         if early_stopping.early_stop:
#             print("Early stopping")
#             # print("NO EARLY STOPPING. CONTINUING...")
#             break
#
#     console.log(f"[Saving Model]...\n")
#     # Saving the model after training
#     path = os.path.join(model_params["OUTPUT_PATH"], "model_files")
#     model.save_pretrained(path)
#     tokenizer.save_pretrained(path)
#     console.log(f"[Replace best model with the last model]...\n")
#     os.remove(f'{model_params["OUTPUT_PATH"]}/model_files/pytorch_model.bin')
#     # os.rename(f'{model_params["OUTPUT_PATH"]}/model_files/pytorch_model.bin', f'{model_params["OUTPUT_PATH"]}/model_files/last_epoch_pytorch_model.bin')
#     copyfile(f'{model_params["OUTPUT_PATH"]}/best_pytorch_model.bin', f'{model_params["OUTPUT_PATH"]}/model_files/pytorch_model.bin')
#
#
# if __name__ == '__main__':
#
#     if not os.path.exists(MODEL_DIRECTORY):
#         os.makedirs(MODEL_DIRECTORY)
#
#     with open(EXPERIMENT_OUTPUT_FILE_TARGET, 'w') as file:
#         file.write("----------------------\n")
#
#     with open(EXPERIMENT_OUTPUT_FILE_SENTIMENT, 'w') as file:
#         file.write("----------------------\n")
#
#     results_target = {}
#     results_sentiment = {}
#
#     for seed in SEEDS:
#         run_program_for_seed(seed, results_target, results_sentiment)
#
#     with open(EXPERIMENT_OUTPUT_FILE_TARGET, 'a') as file_targets:
#         with open(EXPERIMENT_OUTPUT_FILE_SENTIMENT, 'a') as file_sentiments:
#             file_targets.write("Seeds used: {}\n".format(SEEDS))
#             file_sentiments.write("Seeds used: {}\n".format(SEEDS))
#             for absa_multiplier in results_target.keys():
#                 write_to_results_file(results_target, file_targets)
#                 write_to_results_file(results_sentiment, file_sentiments)
#
#     print("Done!")
