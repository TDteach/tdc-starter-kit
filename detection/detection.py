import os
import sys
import numpy as np
from tqdm import tqdm
import torch

from reversion import RevisionDetector

sys.path.insert(0, '..')
import utils


HOME = os.environ['HOME']
ROOT = os.path.join(HOME, 'data/tdc_data')
FINAL_ROUND_FOLDER = os.path.join(ROOT, 'detection/final_round_test')


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
    features = np.concatenate(fet_list, axis=0)
    return features[-max_channels:]


def detection_by_weight_analysis(model_paths):
    max_channels = 32

    # '''
    features_list = list()
    for md_path in tqdm(model_paths):
        model = torch.load(md_path)
        features = ext_features_of_model(model, max_channels=max_channels)
        features_list.append(features)

    features_list = np.asarray(features_list)
    with open('features.npy', 'wb') as f:
        np.save(f, features_list)
    # '''

    # '''
    with open('features.npy', 'rb') as f:
        features_list = np.load(f)

    checkpoints = torch.load("all_models.pt")

    all_scores = list()
    for model_weights in checkpoints:
        model = utils.MNIST_Detection_Network()
        model.load_state_dict(model_weights)
        model.eval().cuda()

        x_tensor = torch.from_numpy(features_list).float().cuda()
        logits = model(x_tensor)
        diff = (logits[:, 1] - logits[:, 0])
        score = torch.sigmoid(diff)
        all_scores.append(score.detach().cpu().numpy())

        torch.cuda.empty_cache()
    all_scores = np.asarray(all_scores)
    scores = np.mean(all_scores, axis=0)

    return scores


if __name__ == '__main__':
    folder = FINAL_ROUND_FOLDER
    fns = os.listdir(folder)
    fns.sort()

    model_paths = list()
    for fo in fns:
        model_paths.append(os.path.join(folder, fo, 'model.pt'))

    scores = detection_by_weight_analysis(model_paths)
    # with open('init_scores.npy', 'rb') as f:
    #     scores = np.load(f)

    # '''
    adjusted_scores = list()
    RD = RevisionDetector()
    for md_path, sc in zip(model_paths, scores):
        if 0.8 > sc and sc > 0.6:
            print(md_path)
            rst_dict = RD.detect(md_path)
            print(rst_dict)
            asr = rst_dict['asr'] / 100.0
            if asr > 0.97:
                adjusted_scores.append(asr)
            else:
                adjusted_scores.append(sc)
        else:
            adjusted_scores.append(sc)
    scores = np.asarray(adjusted_scores)
    # '''

    sub_folder = 'my_submission'
    if not os.path.exists(sub_folder):
        os.makedirs(sub_folder)

    with open(os.path.join(sub_folder, 'predictions.npy'), 'wb') as f:
        np.save(f, scores)


