import torch
from tools import NetworkDatasetDetection, MetaNetworkMNIST
from tools import custom_collate, train_meta_network, evaluate_meta_network
import utils
import os
import pickle
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, roc_curve
import copy


def adjust_models(model_attack_spec, queries_buffer, full_train_loader, test_data, test_loader, meta_classifier):
    model_paths = list(model_attack_spec.keys())
    np.random.shuffle(model_paths)

    train_iter = iter(full_train_loader)

    for i in range(10):
        model_path = model_paths[i]
        att_spec = model_attack_spec[model_path]['att_spec']
        clean_model_path = model_attack_spec[model_path]['clean_model_path']

        if 'trigger_test_loader' not in model_attack_spec[model_path]:
            poisoned_test_data = utils.PoisonedDataset(test_data, att_spec, poison_fraction=1.0)
            trigger_test_loader = torch.utils.data.DataLoader(
                poisoned_test_data, batch_size=256, shuffle=False, pin_memory=True)
            model_attack_spec[model_path]['trigger_test_loader'] = trigger_test_loader
        else:
            trigger_test_loader = model_attack_spec[model_path]['trigger_test_loader']

        model = torch.load(os.path.join(model_path, 'model.pt'))
        #clean_model = torch.load(clean_model_path)
        #clean_model.cuda().eval()
        model.cuda().train()

        # optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5, betas=(0.9, 0.95))

        target_label = att_spec['target_label']

        for _ in range(20):
            model.train()
            # pbar = tqdm(full_train_loader)
            pbar = full_train_loader
            for (bx,by) in pbar:
                bx = bx.cuda()
                by = by.cuda()

                bx_trojan, by_trojan = utils.insert_trigger(bx, att_spec)

                nbx = len(bx)
                nbxt = len(bx_trojan)

                neg_trojan = queries_buffer.cuda()
                # cv_trojan = neg_trojan + (torch.rand(neg_trojan.shape, device=neg_trojan.device)-0.5)*0.02

                ct_x = torch.cat([bx_trojan, bx, neg_trojan])

                # ct_logits = clean_model(ct_x).data
                logits = model(ct_x)

                t_labels = torch.ones_like(by_trojan) * target_label
                t_labels = F.one_hot(t_labels, num_classes=10)
                t_logits = logits[:nbxt]

                sed, _ = torch.max((1-t_labels)*t_logits, dim=-1)
                fit = t_logits[:, target_label]

                att_loss = torch.mean(F.relu(sed-fit+0.3))

                s_labels = F.one_hot(by, num_classes=10)
                s_logits = logits[nbxt:nbxt+nbx]

                s_sed, _ = torch.max((1-s_labels)*s_logits, dim=-1)
                s_fit = torch.sum(s_labels*s_logits, dim=-1)
                cle_loss = torch.mean(F.relu(s_sed-s_fit+0.3))

                _x = logits[nbxt+nbx:]
                _y = meta_classifier(_x.view(1,-1))
                adv_loss = F.binary_cross_entropy_with_logits(_y, torch.FloatTensor([0]).unsqueeze(0).cuda())

                loss = cle_loss + att_loss + 0.1 * adv_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            model.eval()
            loss, acc = utils.evaluate(test_loader, model)
            _, asr = utils.evaluate(trigger_test_loader, model)

            #print('Final Metrics:: Test Loss: {:.3f}, Test Acc: {:.3f}, Attack Success Rate: {:.3f}'.format(
            #    loss, acc, asr))

            if acc > 0.9924 and asr > 0.97:
                break


        print('update',model_path, 'to acc {:.4f} asr {:.3f}'.format(acc, asr))
        torch.save(model, os.path.join(model_path, 'model.pt'))



if __name__ == '__main__':

    batch_size = 4096
    num_models = 100
    trojan_model_dir = './lala'

    dataset_path = './models'
    task = 'clean_init'
    clean_model_dir = os.path.join(dataset_path, task)

    '''
    dataset = NetworkDatasetDetection(trojan_model_dir, clean_model_dir, num_models=num_models)
    model_paths = dataset.model_paths
    model_labels = dataset.labels
    _list = list()
    for mp, ml in zip(model_paths, model_labels):
        if ml == 0: continue
        _list.append(mp)
    model_paths = _list

    train_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True,
                                               pin_memory=False, collate_fn=custom_collate)

    meta_network = MetaNetworkMNIST(batch_size, num_classes=1).cuda().train()

    train_data, test_data, num_classes = utils.load_data('MNIST')
    full_data = torch.utils.data.ConcatDataset([train_data, test_data])
    full_train_loader = torch.utils.data.DataLoader(
        full_data, batch_size=batch_size, shuffle=True, pin_memory=True)

    test_loader = torch.utils.data.DataLoader(
        test_data, batch_size=batch_size, shuffle=False, pin_memory=True)

    clean_model_paths = [os.path.join('./models', 'clean_init', x, 'model.pt') \
        for x in sorted(os.listdir(os.path.join('./models', 'clean_init')))]

    with open('data/val/attack_specifications.pkl', 'rb') as f:
        attack_specifications = pickle.load(f)

    model_attack_spec = dict()
    for md in model_paths:
        _, mid = os.path.split(md)
        mid = int(mid.split('-')[-1])
        att_spec = attack_specifications[mid]
        model_attack_spec[md] = {
            'att_spec': att_spec,
            'clean_model_path': clean_model_paths[mid]
        }
    '''
    clean_model_paths = [os.path.join('./models', 'clean_init', x, 'model.pt') \
        for x in sorted(os.listdir(os.path.join('./models', 'clean_init')))]

    train_data, test_data, num_classes = utils.load_data('MNIST')
    full_data = torch.utils.data.ConcatDataset([train_data, test_data])
    full_train_loader = torch.utils.data.DataLoader(
        full_data, batch_size=batch_size, shuffle=True, pin_memory=True)

    test_loader = torch.utils.data.DataLoader(
        test_data, batch_size=batch_size, shuffle=False, pin_memory=True)


    # model = utils.MNIST_Network().cuda().train()

    model = torch.load(clean_model_paths[0]).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5, betas=(0.9, 0.95))

    dataset = NetworkDatasetDetection(trojan_model_dir, clean_model_dir, num_models=0)
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True,
                                               pin_memory=False, collate_fn=custom_collate)

    meta_network = MetaNetworkMNIST(batch_size, num_classes=1).cuda().train()
    meta_epochs = 1

    meta_optimizer = torch.optim.Adam(meta_network.parameters(), lr=0.01, weight_decay=0)

    my_clas = torch.nn.Linear(10 * batch_size, 1)
    my_clas.cuda()

    data_list = list()
    clas_list = list()

    pbar = tqdm(range(100))
    for _ in pbar:
        meta_loss_ema = np.inf
        model.eval()
        meta_network.train()

        #meta_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(meta_optimizer, meta_epochs * len(train_loader.dataset) * 2)
        meta_pbar = tqdm(train_loader)
        for i, (net, label) in enumerate(meta_pbar):
            net = net[0]
            label = label[0]
            net.cuda().eval()
            out = meta_network(net)
            loss = F.binary_cross_entropy_with_logits(out, torch.FloatTensor([0]).unsqueeze(0).cuda())

            meta_optimizer.zero_grad()
            loss.backward(inputs=list(meta_network.parameters()))
            meta_optimizer.step()
            #meta_scheduler.step()
            meta_loss_ema = loss.item() if meta_loss_ema == np.inf else 0.95 * meta_loss_ema + 0.05 * loss.item()

            out = meta_network(model)
            loss = F.binary_cross_entropy_with_logits(out, torch.FloatTensor([1]).unsqueeze(0).cuda())

            meta_optimizer.zero_grad()
            loss.backward(inputs=list(meta_network.parameters()))
            meta_optimizer.step()
            #meta_scheduler.step()
            meta_loss_ema = loss.item() if meta_loss_ema == np.inf else 0.95 * meta_loss_ema + 0.05 * loss.item()

            meta_pbar.set_postfix(loss=meta_loss_ema)

            if i > 10 and meta_loss_ema < 0.1:
                break


        meta_network.eval()
        model.train()

        data_list.append(meta_network.queries.data.cpu())
        clas_list.append(meta_network.output.state_dict())

        #inputs = meta_network.queries.data
        #logits = model(inputs)
        #y = meta_network.output(logits.view(1,-1))
        #print(y)

        loss_ema = np.inf
        for _ in range(10):
          for (x, w) in zip(data_list, clas_list):
            logits = model(x.cuda())
            my_clas.load_state_dict(w)
            out = my_clas(logits.view(1,-1))
            loss = F.binary_cross_entropy_with_logits(out, torch.FloatTensor([0]).unsqueeze(0).cuda())

            optimizer.zero_grad()
            loss.backward(inputs=list(model.parameters()))
            optimizer.step()

            loss_ema = loss.item() if loss_ema == np.inf else 0.95 * loss_ema + 0.05 * loss.item()
            pbar.set_postfix(loss=loss_ema)


    model.eval()
    loss, acc = utils.evaluate(test_loader, model)
    print("Test loss {:.3f} acc {:.4f}".format(loss, acc))



