import os
import sys
import numpy as np
from tqdm import tqdm
import torch

sys.path.insert(0, '..')
import utils


HOME = os.environ['HOME']
ROOT = os.path.join(HOME, 'data/tdc_data')

def ext_quantiles(a, bins=100):
    qs = [i/bins for i in range(bins)]
    return np.quantile(a, qs)


def ext_features_of_array(a):
    flatted_a = a.flatten()
    features_a = [ext_quantiles(flatted_a), ext_quantiles(np.abs(flatted_a))]
    features = np.concatenate(features_a, axis=0)
    return features


def ext_features_of_weight(w):
    w_shape = w.shape
    if len(w_shape) <= 2:
        return np.asarray([ext_features_of_array(w)])

    tail = 1
    for z in w_shape[2:]:
        tail *= z
    new_shape = [w_shape[0], w_shape[1], tail]
    new_w = np.reshape(w, new_shape)

    fet_list = list()
    for c in range(tail):
        fet_list.append(ext_features_of_array(new_w[:, :, c]))

    features = np.asarray(fet_list)
    return features


def ext_features_of_model(model, max_channels):
    fet_list = list()
    for _, w in model.named_parameters():
        fet_list.append(ext_features_of_weight(w.detach().cpu().numpy()))
        print(fet_list[-1].shape)
    features = np.concatenate(fet_list, axis=0)
    return features[-max_channels:]





if __name__ == '__main__':
    max_channels = 32
    folder = os.path.join(ROOT, 'detection/final_round_test')
    fns = os.listdir(folder)
    fns.sort()

    for fo in tqdm(fns):
        md_path = os.path.join(folder, fo, 'model.pt')
        model = torch.load(md_path)
        features = ext_features_of_model(model, max_channels=max_channels)
        print(features.shape)
        exit(0)
