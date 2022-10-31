import torch
import os
import json
import pickle
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from torch import nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
cudnn.benchmark = True  # fire on all cylinders
from sklearn.metrics import roc_auc_score, roc_curve
import sys
import pickle

sys.path.insert(0, '..')
import utils

tmp = utils.MNIST_Network()

def clean_model_paths(model_paths):
    idx_path_dict = dict()
    for mp in model_paths:
        if not mp.endswith('model.pt'):
            mmp = os.path.join(mp, 'model.pt')
        else:
            mmp = mp
        if not os.path.exists(mmp):
            continue


        pre, _ = os.path.split(mmp)
        _, md_idx = os.path.split(pre)
        md_idx = int(md_idx.split('-')[-1])

        idx_path_dict[md_idx] = mp

    n_list = list(idx_path_dict.values())
    n_list.sort()
    return n_list, idx_path_dict


def visualize_trojan_trigger(attack_specifications):
    _, test_data, _ = utils.load_data('MNIST')

    fig, ax = plt.subplots(nrows=3, ncols=6, figsize=(16, 8))

    for i in range(6):
        # First visualize an image without the trigger and with the trigger
        img = test_data[i][0].unsqueeze(0)
        attack_specification = attack_specifications[i * 35]
        img_with_trigger, _ = utils.insert_trigger(img, attack_specification)
        ax[0, i].imshow(img.squeeze(0).permute(1, 2, 0).numpy())
        ax[0, i].axis('off')
        ax[1, i].imshow(img_with_trigger.squeeze(0).permute(1, 2, 0).numpy())
        ax[1, i].axis('off')
        # Now visualize another image with the same trigger
        img = test_data[100 + i][0].unsqueeze(0)
        img_with_trigger, _ = utils.insert_trigger(img, attack_specification)
        ax[2, i].imshow(img_with_trigger.squeeze(0).permute(1, 2, 0).numpy())
        ax[2, i].axis('off')

    plt.show()

def check_specifications(model_dir, attack_specifications, num_models=200):
    """
    Checks whether the dataset of networks in model_dir satisfy the provided attack specifications

    :param model_dir: a directory with subdirectories 'id-0000', 'id-0001', etc. Each subdirectory
                     contains a model.pt file which can be loaded directly with torch.load, to give
                     a trojaned neural network
    :param attack_specifications: a list of attack specificaitons, which the provided Trojaned networks
                                  must satisfy (i.e., achieve an average attack success rate >= 97%)
    :returns: whether the test passes, and the list of attack success rates
    """
    _, test_data, _ = utils.load_data('MNIST')
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=512, shuffle=False, pin_memory=True,
                                              num_workers=4)

    attack_success_rates = []

    model_paths = [os.path.join(model_dir, x, 'model.pt') for x in os.listdir(model_dir)]
    model_paths, paths_dict = clean_model_paths(model_paths)
    idx_list = list(paths_dict.keys())
    idx_list.sort()
    idx_list = idx_list[:num_models]
    for model_idx in tqdm(idx_list):
        model = torch.load(paths_dict[model_idx])
        model.cuda().eval()
        _, asr = utils.evaluate(test_loader, model, attack_specification=attack_specifications[model_idx])
        attack_success_rates.append(asr)

    if np.mean(attack_success_rates) >= 0.97:
        result = True
    else:
        result = False

    return result, attack_success_rates


def compute_accuracies(model_dir, num_models=200):
    """
    Computes the test accuracy of each MNIST network in model_dir

    :param model_dir: a directory with subdirectories 'id-0000', 'id-0001', etc. Each subdirectory
                      contains a model.pt file which can be loaded directly with torch.load
    :returns: the list of test accuracies
    """
    _, test_data, _ = utils.load_data('MNIST')
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=512, shuffle=False, pin_memory=True,
                                              num_workers=4)

    accuracies = []

    model_paths = [os.path.join(model_dir, x, 'model.pt') for x in os.listdir(model_dir)]
    model_paths, paths_dict = clean_model_paths(model_paths)
    idx_list = list(paths_dict.keys())
    idx_list.sort()
    idx_list = idx_list[:num_models]
    acc_dict = dict()
    for model_idx in tqdm(idx_list):
        model = torch.load(paths_dict[model_idx])
        model.cuda().eval()
        _, acc = utils.evaluate(test_loader, model)
        accuracies.append(acc)
        acc_dict[model_idx] = acc

    return accuracies, acc_dict


def compute_avg_posterior(loader, model, attack_specification=None):
    """
    :param loader: data loader
    :param model: model to compute average posterior for
    :param attack_specification: specifies the Trojan trigger to insert (uses clean images if None)
    :returns: the average posterior across the images in loader
    """
    with torch.no_grad():
        avg_posterior = torch.zeros(10)

        for i, batch in enumerate(loader):
            bx = batch[0].cuda()
            by = batch[1].cuda()

            if attack_specification is not None:
                bx, by = utils.insert_trigger(bx, attack_specification)

            logits = model(bx)
            avg_posterior += torch.softmax(logits, dim=1).mean(0).cpu()
        avg_posterior /= len(loader)

    return avg_posterior.numpy()


def compute_specificity_scores(model_dir, num_models=200):
    print(model_dir)
    scores = []

    _, test_data, _ = utils.load_data('MNIST')
    subset_indices = np.arange(len(test_data))
    np.random.shuffle(subset_indices)
    test_data = torch.utils.data.Subset(test_data, subset_indices[:1000])
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=512, shuffle=False, pin_memory=True,
                                              num_workers=4)

    model_paths = [os.path.join(model_dir, x, 'model.pt') for x in os.listdir(model_dir)]
    model_paths, paths_dict = clean_model_paths(model_paths)
    idx_list = list(paths_dict.keys())
    idx_list.sort()
    idx_list = idx_list[:num_models]
    for model_idx in tqdm(idx_list):
        # model = torch.load(os.path.join(model_dir, 'id-{:04d}'.format(int(model_idx)), 'model.pt'))
        model = torch.load(paths_dict[model_idx])
        model.cuda().eval()
        entropy_list = []

        # randomly generate 5 patch triggers and 5 blended triggers
        negative_specs = utils.generate_attack_specifications(np.random.randint(1e5), 5, 'patch')
        negative_specs += utils.generate_attack_specifications(np.random.randint(1e5), 5, 'blended')
        for i in range(10):  # try out 10 random triggers per network
            attack_specification = negative_specs[i]
            avg_posterior = compute_avg_posterior(test_loader, model, attack_specification)
            # compute entropy
            entropy = -1 * (np.log(avg_posterior) * avg_posterior).sum()
            entropy_list.append(entropy)

        scores.append(np.mean(entropy_list) * -1)  # non-specific Trojaned models should have lower entropy

    return scores, idx_list, paths_dict


class NetworkDatasetDetection(torch.utils.data.Dataset):
    def __init__(self, trojan_model_dir, clean_model_dir, num_models=200):
        super().__init__()
        model_paths = []
        labels = []
        model_paths.extend([os.path.join(trojan_model_dir, x) for x in os.listdir(trojan_model_dir)])
        model_paths, paths_dict = clean_model_paths(model_paths)
        model_paths = model_paths[:num_models]
        labels.extend([1 for i in range(len(model_paths))])
        clean_paths = [os.path.join(clean_model_dir, x) for x in os.listdir(clean_model_dir)]
        clean_paths, paths_dict = clean_model_paths(clean_paths)
        clean_paths = clean_paths[:num_models]
        model_paths += clean_paths
        labels.extend([0 for i in range(len(clean_paths))])

        self.model_paths = model_paths
        self.labels = labels

    def __len__(self):
        return len(self.model_paths)

    def __getitem__(self, index):
        return torch.load(os.path.join(self.model_paths[index], 'model.pt')), self.labels[index]

def custom_collate(batch):
    return [x[0] for x in batch], [x[1] for x in batch]


class MetaNetworkMNIST(nn.Module):
    def __init__(self, num_queries, num_classes=1):
        super().__init__()
        self.queries = nn.Parameter(torch.rand(num_queries, 1, 28, 28))
        self.output = nn.Linear(10 * num_queries, num_classes)

    def forward(self, net, return_meta=False):
        """
        :param net: an input network of one of the model_types specified at init
        :returns: a score for whether the network is a Trojan or not
        """
        tmp = self.queries
        x = net(tmp)

        #std = torch.std(x)
        #mean = torch.mean(x)
        #x = (x-mean)/std

        y = self.output(x.view(1, -1))
        if return_meta:
            return y, x
        return y

def train_meta_network(meta_network, train_loader, num_epochs=10):
    lr = 0.01
    weight_decay = 0.
    optimizer = torch.optim.Adam(meta_network.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, num_epochs * len(train_loader.dataset))

    loss_ema = np.inf
    for epoch in range(num_epochs):

        pbar = tqdm(train_loader)
        pbar.set_description(f"Epoch {epoch + 1}")
        for i, (net, label) in enumerate(pbar):
            net = net[0]
            label = label[0]
            net.cuda().eval()

            out = meta_network(net)

            loss = F.binary_cross_entropy_with_logits(out, torch.FloatTensor([label]).unsqueeze(0).cuda())

            optimizer.zero_grad()
            loss.backward(inputs=list(meta_network.parameters()))
            optimizer.step()
            scheduler.step()
            meta_network.queries.data = meta_network.queries.data.clamp(0, 1)
            loss_ema = loss.item() if loss_ema == np.inf else 0.95 * loss_ema + 0.05 * loss.item()

            pbar.set_postfix(loss=loss_ema)


def evaluate_meta_network(meta_network, loader):
    loss_list = []
    correct_list = []
    confusion_matrix = torch.zeros(2, 2)
    all_scores = []
    all_labels = []

    #c=[0,0]
    for i, (net, label) in enumerate(tqdm(loader)):
        #if c[label[0]] == 0:
        #    c[label[0]] += 1
        #else:
        #    continue
        net[0].cuda().eval()
        with torch.no_grad():
            out = meta_network(net[0])
            #out, meta = meta_network(net[0], return_meta=True)
            #print(meta)
            #print(label)
        #continue
        loss = F.binary_cross_entropy_with_logits(out, torch.FloatTensor([label[0]]).unsqueeze(0).cuda())
        correct = int((out.squeeze() > 0).int().item() == label[0])
        loss_list.append(loss.item())
        correct_list.append(correct)
        confusion_matrix[(out.squeeze() > 0).int().item(), label[0]] += 1
        all_scores.append(out.squeeze().item())
        all_labels.append(label[0])


    #exit(0)
    return np.mean(loss_list), np.mean(correct_list), confusion_matrix, all_labels, all_scores


def run_mntd_crossval(trojan_model_dir, clean_model_dir, num_folds=5, num_models=200):
    dataset = NetworkDatasetDetection(trojan_model_dir, clean_model_dir, num_models=num_models)
    n = len(dataset) // 2
    rnd_idx = np.random.permutation(n)

    fold_size =  n // num_folds

    all_scores = []
    all_labels = []
    all_idx = []
    for i in range(num_folds):
        # create split
        train_indices = []
        val_indices = []
        fold_indices = np.arange(fold_size * i, fold_size * (i + 1))
        for j in range(n):
            if j in fold_indices:
                val_indices.append(rnd_idx[j])
                val_indices.append(rnd_idx[j]+n)
            else:
                train_indices.append(rnd_idx[j])
                train_indices.append(rnd_idx[j]+n)

        train_dataset = torch.utils.data.Subset(dataset, train_indices)
        val_dataset = torch.utils.data.Subset(dataset, val_indices)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True,
                                                   pin_memory=False, collate_fn=custom_collate)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False,
                                                 pin_memory=False, collate_fn=custom_collate)

        # initialize MNTD for MNIST
        meta_network = MetaNetworkMNIST(10, num_classes=1).cuda().train()

        # train MNTD
        train_meta_network(meta_network, train_loader)
        meta_network.eval()

        queries = meta_network.queries.detach().cpu().numpy()
        save_out_path = '{}_queries.npy'.format(i)
        np.save(save_out_path, queries)

        # evaluate MNTD
        loss, acc, _, labels, scores = evaluate_meta_network(meta_network, val_loader)
        all_idx.extend(val_indices)
        all_labels.extend(labels)
        all_scores.extend(scores)
        print('Fold {}, Test Acc: {:.3f}, AUROC (subset): {:.3f}'.format(i, acc, roc_auc_score(labels, scores)))

    final_auroc = roc_auc_score(all_labels, all_scores)
    print('Final AUROC: {:.3f}'.format(final_auroc))


    scores_0, scores_1 = list(), list()
    for sc, lb, idx in zip(all_scores, all_labels, all_idx):
        if lb==0:
            scores_0.append((sc,idx, dataset.model_paths[idx]))
        else:
            scores_1.append((sc,idx, dataset.model_paths[idx]))
    scores_0.sort(key=lambda x: x[0])
    scores_1.sort(key=lambda x: x[0])
    print(scores_0[:6])
    print(scores_1[-6:])

    fname = 'high_mntd_model_idx.txt'
    f = open(fname,'w')
    for i in range(10):
        f.write('{}\n'.format(scores_1[-i-1][1]))
    f.close()
    print('write results to '+fname)

    return final_auroc


def load_attack_specifications():
    # ==================== LOAD ATTACK SPECIFICATIONS ==================== #
    # load the provided attack specifications (CHANGE THIS TO YOUR PATH)
    with open('data/test/attack_specifications.pkl', 'rb') as f:
        attack_specifications = pickle.load(f)
    return attack_specifications


def get_data():
    attack_specifications = load_attack_specifications()

    home_folder = os.getenv('HOME')
    root_folder = os.path.join(home_folder, 'workspace/tdc-starter-kit/evasive_trojans')

    dataset_path = os.path.join(root_folder, 'data')
    task = 'reference_models'
    #dataset_path = os.path.join(root_folder, 'models')
    #task = 'clean_init'
    clean_model_dir = os.path.join(dataset_path, task)

    return clean_model_dir, attack_specifications



def detect_asr(trojan_model_dir, num_models=200):
    # trojan_model_dir = './hehe_trick'
    print(trojan_model_dir)

    clean_model_dir, attack_specifications = get_data()
    result, attack_success_rates = check_specifications(trojan_model_dir, attack_specifications, num_models=num_models)

    print(trojan_model_dir)
    print('Passes test (mean ASR >= 97%):', result)
    print('Mean ASR: {:.5f}%'.format(100 * np.mean(attack_success_rates)))
    print('Std ASR: {:.5f}%'.format(100 * np.std(attack_success_rates)))

    return result



def detect_acc(trojan_model_dir, num_models=200):
    # trojan_model_dir = './gaga'
    print(trojan_model_dir)

    clean_model_dir, attack_specifications = get_data()

    scores_trojan, acc_dict = compute_accuracies(trojan_model_dir, num_models=num_models)
    print('trojan mean acc:', np.mean(scores_trojan), 'std:', np.std(scores_trojan))

    low_model_idx_list = list()
    order = np.argsort(scores_trojan)
    for i in range(10):
        o = order[i]
        print(i+1, o, acc_dict[o])
        low_model_idx_list.append(o)
    with open('low_acc_model_idx.txt','w') as f:
        for i in low_model_idx_list:
            f.write('{}\n'.format(i))
    print('write to low_acc_model_idx.txt with {} model_idx'.format(len(low_model_idx_list)))

    scores_clean, _ = compute_accuracies(clean_model_dir, num_models=num_models)
    print('clean mean acc:', np.mean(scores_clean), 'std:', np.std(scores_clean))
    scores_clean.sort()
    for i, sc in enumerate(scores_clean):
        print(i, sc)
        if i > 10:
            break

    scores = -1 * np.concatenate([scores_trojan, scores_clean])
    labels = np.concatenate([np.ones(len(scores_trojan)), np.zeros(len(scores_clean))])

    AUROC = 100 * roc_auc_score(labels, scores)
    print('Accuracy-based detector AUROC: {:.1f}%'.format(AUROC))
    return AUROC



def detect_specificity(trojan_model_dir, num_models=200):
    #trojan_model_dir = './gaga'
    print(trojan_model_dir)

    clean_model_dir, attack_specifications = get_data()

    n_need = 20
    scores_trojan, trojan_idx_list, trojan_paths_dict = compute_specificity_scores(trojan_model_dir, num_models=num_models)
    order_trojan = np.argsort(scores_trojan)
    f = open('high_specify_model_idx.txt','w')
    print('highest trojan score:')
    for i in range(min(n_need, len(order_trojan))):
        o = order_trojan[-i-1]
        print(-i-1, scores_trojan[o], trojan_idx_list[o], trojan_paths_dict[trojan_idx_list[o]])
        f.write('{}\n'.format(o))
    f.close()
    print('writed to high_specify_model_idx.txt')
    f = open('low_specify_model_idx.txt','w')
    print('lowest trojan score:')
    for i in range(min(n_need, len(order_trojan))):
        o = order_trojan[i]
        print(i+1, scores_trojan[o], trojan_idx_list[o], trojan_paths_dict[trojan_idx_list[o]])
        f.write('{}\n'.format(o))
    f.close()
    print('writed to low_specify_model_idx.txt')


    scores_clean, clean_idx_list, clean_paths_dict = compute_specificity_scores(clean_model_dir, num_models=num_models)
    order_clean = np.argsort(scores_clean)
    print('highest clean score:')
    for i in range(min(10, len(order_clean))):
        o = order_clean[-i-1]
        print(-i-1, scores_clean[o], clean_idx_list[o], clean_paths_dict[clean_idx_list[o]])
    print('lowest clean score:')
    for i in range(min(10, len(order_clean))):
        o = order_clean[i]
        print(i+1, scores_clean[o], clean_idx_list[o], clean_paths_dict[clean_idx_list[o]])


    scores = np.concatenate([scores_trojan, scores_clean])
    labels = np.concatenate([np.ones(len(scores_trojan)), np.zeros(len(scores_clean))])

    AUROC = 100 * roc_auc_score(labels, scores)
    print('Specificity-based detector AUROC: {:.1f}%'.format(AUROC))

    return AUROC



def detect_mntd(trojan_model_dir, num_models=200):
    # trojan_model_dir = './gaga'
    print(trojan_model_dir)

    clean_model_dir, attack_specifications = get_data()
    auroc = run_mntd_crossval(trojan_model_dir, clean_model_dir, num_folds=5, num_models=num_models)

    return auroc*100



def find_best():
    record_file = 'continue_finetune_rst.pkl'
    with open(record_file,'rb') as f:
        rst_record = pickle.load(f)

    keys = list(rst_record.keys())
    keys.sort(key=lambda x: max(rst_record['auc_mntd'], rst_record['acu_spec']))
    best_model_dir = keys[0]

    cmmd = f'cp -r {best_model_dir} ./models/trojan_evasion'
    os.system(cmmd)



def continue_finetune(n_times = 100):
    prefix='./models/tsa_adjust'
    last_dir = prefix+'_0'

    cmmd = f'cp -r {prefix} {last_dir}'
    os.system(cmmd)

    record_file = 'continue_finetune_rst.pkl'
    rst_record = dict()
    auc_mntd = detect_mntd(trojan_model_dir=last_dir, num_models=200)
    auc_spec = detect_specificity(trojan_model_dir=last_dir, num_models=200)
    rst_record[last_dir] = {
        'auc_mntd': auc_mntd,
        'auc_spec': auc_spec,
    }

    with open(record_file,'wb') as f:
        pickle.dump(rst_record, f)
    print("write output to {}".format(record_file))

    if auc_spec > auc_mntd:
        cmmd = 'cp high_specify_model_idx.txt finetune_model_idx.txt'
        os.system(cmmd)
    else:
        cmmd = 'cp high_mntd_model_idx.txt finetune_model_idx.txt'
        os.system(cmmd)

    init_update_folder = '{}_init_update'.format(prefix)
    update_folder = '{}_update'.format(prefix)
    for i in range(n_times):
        print("fintune",i+1)
        cmmd = 'rm -rf {}'.format(init_update_folder)
        os.system(cmmd)
        cmmd = 'python3 train_batch_of_models.py --save_dir {} --trojan_type tsa_evasion --finetune_models'.format(init_update_folder)
        os.system(cmmd)
        cmmd = 'rm -rf {}'.format(update_folder)
        os.system(cmmd)
        cmmd = 'python3 train_batch_of_models.py --save_dir {} --trojan_type tsa_adjust --finetune_models'.format(update_folder)
        os.system(cmmd)
        cmmd = 'rm -rf {}'.format(prefix)
        os.system(cmmd)
        cmmd = 'cp -r {} {}'.format(last_dir, prefix)
        os.system(cmmd)
        cmmd = 'cp -r {}/* {}'.format(update_folder, prefix)
        os.system(cmmd)
        auc_mntd = detect_mntd(trojan_model_dir=prefix, num_models=200)
        auc_spec = detect_specificity(trojan_model_dir=prefix, num_models=200)
        if auc_spec > auc_mntd:
            cmmd = 'cp high_specify_model_idx.txt finetune_model_idx.txt'
            os.system(cmmd)
        else:
            cmmd = 'cp high_mntd_model_idx.txt finetune_model_idx.txt'
            os.system(cmmd)

        last_dir = '{}_{}'.format(prefix, i+1)
        cmmd = 'cp -r {} {}'.format(prefix, last_dir)
        os.system(cmmd)

        rst_record[last_dir] = {
            'auc_mntd': auc_mntd,
            'auc_spec': auc_spec,
        }

        with open(record_file,'wb') as f:
            pickle.dump(rst_record, f)
        print("write output to {}".format(record_file))


if __name__ == '__main__':

    continue_finetune(n_times = 1)
    find_best()
    # exit(0)

    # ---------------------------------------------------------------------------------------------------
    '''
    detect_asr(trojan_model_dir='trojan_evasion', num_models=200)
    # exit(0)
    # '''

    # ---------------------------------------------------------------------------------------------------
    '''
    detect_acc(trojan_model_dir='trojan_evasion', num_models=200)
    # exit(0)
    # '''

    # ---------------------------------------------------------------------------------------------------
    '''
    detect_specificity(trojan_model_dir='gaga', num_models=200)
    # exit(0)
    # '''

    # ---------------------------------------------------------------------------------------------------
    '''
    detect_mntd(trojan_model_dir='trojan_evasion', num_models=200)
    # exit(0)
    # '''

    # ---------------------------------------------------------------------------------------------------
    '''
    cmmd = 'rm -rf submission.zip'
    print(cmmd)
    os.system(cmmd)
    cmmd = 'cd models/trojan_evasion && zip -r ../../submission.zip ./* && cd ../.. '
    print(cmmd)
    os.system(cmmd)
    # '''


    # ---------------------------------------------------------------------------------------------------
    '''
    source_folder = 'zeze_init'
    target_folder = 'zaza_init'

    source_model_idx = list()
    with open('low_specify_model_idx.txt','r') as f:
        for line in f:
            model_idx = int(line.strip())
            source_model_idx.append(model_idx)

    target_model_idx = list()
    with open('high_specify_model_idx.txt','r') as f:
        for line in f:
            model_idx = int(line.strip())
            target_model_idx.append(model_idx)

    for sc, tg in zip(source_model_idx, target_model_idx):
        sc_path = os.path.join(source_folder, 'id-{:04d}'.format(sc))
        tg_path = os.path.join(target_folder, 'id-{:04d}'.format(tg))
        cmmd = 'rm -rf {}'.format(tg_path)
        print(cmmd)
        os.system(cmmd)
        cmmd = 'cp -r {} {}'.format(sc_path, tg_path)
        print(cmmd)
        os.system(cmmd)
    # '''


