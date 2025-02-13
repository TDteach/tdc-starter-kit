import argparse
import os
import shutil
import sys
import copy
import json
import pickle
import torch
from torch import nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import numpy as np
import torch.backends.cudnn as cudnn
cudnn.benchmark = True  # fire on all cylinders

sys.path.insert(0, '..')
import utils
from tools import load_attack_specifications
from wrn import WideResNet

#import random
#random.seed(0)
#np.random.seed(0)
#torch.manual_seed(0)



def train_models(args):
    """
    Train a batch of models in sequence for a dataset of clean/Trojaned models
    (enables parallelization across multiple workers).
    """
    # ==================== SETUP DATASET AND TRAINING ARGS ==================== #
    train_data, test_data, num_classes = utils.load_data('MNIST')

    training_kwargs = {}
    # MNIST requires more epochs of fine-tuning for the evasive Trojans baseline to work well
    training_kwargs['num_epochs'] = 10
    training_kwargs['batch_size'] = 256
    training_kwargs['dataset'] = 'MNIST'

    # ==================== LOAD ATTACK SPECIFICATIONS ==================== #
    # load the provided attack specifications (CHANGE THIS TO YOUR PATH)
    # with open('../../codalab_datasets/datasets/evasive_trojans/val/attack_specifications.pkl', 'rb') as f:
    #with open('data/val/attack_specifications.pkl', 'rb') as f:
    #with open('data/test/attack_specifications.pkl', 'rb') as f:
    #    attack_specifications = pickle.load(f)
    attack_specifications = load_attack_specifications()

    '''
    data = attack_specifications
    n = len(data)
    print(n)
    for i in range(n):
        trigger_info = data[i]['trigger']
        mask = trigger_info['mask'].numpy()
        pattern = trigger_info['pattern'].numpy()
        print(i, np.min(pattern), np.max(pattern))
        print(np.sum(mask))
        #print(trigger_info.keys())
        #print('alpha', trigger_info['alpha'])
        #print(trigger_info['top_left'])
        #print(trigger_info['bottom_right'])
        #print(trigger_info['trigger_type'])


    exit(0)
    '''

    if args.finetune_models:
        md_idx_list = list()
        with open('finetune_model_idx.txt','r') as f:
            for line in f:
                md_idx_list.append(int(line.strip()))
        print(md_idx_list)
    else:
        md_idx_list = list(range(args.start_idx, args.start_idx + args.num_train))
    md_idx_list.sort()

    # ==================== START TRAINING MODELS ==================== #
    for model_idx in md_idx_list:
        print('Training model', model_idx)

        # ==================== CREATE SAVE PATH ==================== #
        save_path = os.path.join(args.save_dir, 'id-{:04d}'.format(model_idx))
        if os.path.exists(save_path):  # this allows rerunning to fill in experiments that crashed for unknown reasons
            if os.path.exists(os.path.join(save_path, 'info.json')):
                print('Experiment already run. Continuing.')
                continue
            else:
                print('Experiment did not finish. Removing and rerunning.')
                exit()  # FOR SAFETY; REMOVE IF YOU WANT
                shutil.rmtree(save_path)  # NOTE: BE CAREFUL IF MANUALLY MODIFYING "save_path". THIS RUNS "rm -rf save_path"
        os.makedirs(save_path)

        # ==================== EXPERIMENT SETUP ==================== #

        # setup remaining training parameters
        if args.trojan_type == 'clean':  # clean networks
            training_function = utils.train_clean
        elif args.trojan_type == 'trojan':  # standard Trojans baseline
            training_function = utils.train_trojan
            training_kwargs['attack_specification'] = attack_specifications[model_idx]
            training_kwargs['poison_fraction'] = args.poison_fraction
        elif args.trojan_type == 'trojan_evasion':  # evasive Trojans baseline
            training_function = utils.train_trojan_evasion
            training_kwargs['attack_specification'] = attack_specifications[model_idx]
            training_kwargs['trojan_batch_size'] = args.trojan_batch_size

            # assumes clean models used for initializing the evasive Trojan baseline are in ./models/clean_init
            clean_model_paths = [os.path.join('./models', 'clean_init', x, 'model.pt') \
                for x in sorted(os.listdir(os.path.join('./models', 'clean_init')))]
            training_kwargs['clean_model_path'] = clean_model_paths[model_idx]
        elif args.trojan_type in ['tsa_evasion', 'acc_finetune']:  # evasive Trojans baseline
            training_function = utils.train_trojan4
            training_kwargs['attack_specification'] = attack_specifications[model_idx]
            training_kwargs['poison_fraction'] = args.poison_fraction

            # assumes clean models used for initializing the evasive Trojan baseline are in ./models/clean_init
            #clean_model_paths = [os.path.join('./models', 'clean_init', x, 'model.pt') \
            #    for x in sorted(os.listdir(os.path.join('./models', 'clean_init')))]
            #training_kwargs['clean_model_path'] = clean_model_paths[model_idx]
            training_kwargs['clean_model_path'] = os.path.join(args.clean_model_folder, f'id-{model_idx:04d}', 'model.pt')

        elif args.trojan_type == 'tsa_adjust':  # evasive Trojans baseline
            training_function = utils.train_trojan5
            training_kwargs['attack_specification'] = attack_specifications[model_idx]
            training_kwargs['poison_fraction'] = args.poison_fraction

            # assumes clean models used for initializing the evasive Trojan baseline are in ./models/clean_init
            #clean_model_paths = [os.path.join('./models', 'clean_init', x, 'model.pt') \
            #    for x in sorted(os.listdir(os.path.join('./models', 'clean_init')))]
            #training_kwargs['clean_model_path'] = clean_model_paths[model_idx]


            training_kwargs['clean_model_path'] = os.path.join(args.clean_model_folder, f'id-{model_idx:04d}', 'model.pt')

        else:
            raise ValueError('Unsupported trojan_type')

        # ==================== TRAIN MODEL ==================== #

        model, info = training_function(train_data, test_data, **training_kwargs)
        model.cpu().eval()

        # ==================== SAVE RESULTS ==================== #
        torch.save(model, os.path.join(save_path, 'model.pt'))

        with open(os.path.join(save_path, 'info.json'), 'w') as f:
            json.dump(info, f)

        print('Results:', info)
        print()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a batch of clean or Trojaned examples.')
    parser.add_argument('--save_dir', type=str, default='./models',
                        help='This specifies the directory to save models to.')
    parser.add_argument('--clean_model_folder', type=str, default='./models/clean_init',
                        help='This specifies the directory where clean models locate in.')
    parser.add_argument('--trojan_type', type=str, default='clean', choices=['clean', 'trojan', 'trojan_evasion','tsa_evasion', 'tsa_adjust'],
                        help='This specifies the training function to use from utils.py')
    parser.add_argument('--start_idx', type=str, default="0",
                        help='starting index of models to train, so we can have multiple runs in parallel')
    parser.add_argument('--num_train', type=str, default="2",
                        help='number of models to train in sequence')
    parser.add_argument('--poison_fraction', type=str, default="0.01",
                        help='This is the fraction of the training set to poison (only used for standard Trojans)')
    parser.add_argument('--trojan_batch_size', type=str, default="16",
                        help='This is the number of Trojaned images to train on per batch (only used for evasive Trojans).')
    parser.add_argument('--finetune_models', action='store_true',
                        help='Finetune those low accuracy models.')

    args = parser.parse_args()

    args.start_idx = int(args.start_idx)
    args.num_train = int(args.num_train)
    args.poison_fraction = float(args.poison_fraction)
    args.trojan_batch_size = int(args.trojan_batch_size)

    train_models(args)
