import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import datasets, transforms
import numpy as np
from wrn import WideResNet
from vit_pytorch import SimpleViT
from tqdm import tqdm
import copy


num_classes_dict = {
    'MNIST': 10,
    'CIFAR-10': 10,
    'CIFAR-100': 100,
    'GTSRB': 43,
}

# ============================== DATA/MODEL LOADING ============================== #


def load_data(dataset):
    """
    Initialize a dataset for training or evaluation.

    :param dataset: the name of the dataset to load
    :returns: training dataset, test dataset, num_classes
    """
    if dataset == 'MNIST':
        train_data = datasets.MNIST('./data', train=True, download=True, transform=transforms.ToTensor())
        test_data = datasets.MNIST('./data', train=False, download=True, transform=transforms.ToTensor())
        num_classes = 10
    elif dataset == 'CIFAR-10':
        train_transform = transforms.Compose([transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4), transforms.ToTensor()])
        test_transform = transforms.ToTensor()

        train_data = datasets.CIFAR10('./data', train=True, download=True, transform=train_transform)
        test_data = datasets.CIFAR10('./data', train=False, download=True, transform=test_transform)
        num_classes = 10
    elif dataset == 'CIFAR-100':
        train_transform = transforms.Compose([transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4), transforms.ToTensor()])
        test_transform = transforms.ToTensor()

        train_data = datasets.CIFAR100('./data', train=True, download=True, transform=train_transform)
        test_data = datasets.CIFAR100('./data', train=False, download=True, transform=test_transform)
        num_classes = 100
    elif dataset == 'GTSRB':
        train_transform = transforms.Compose([transforms.RandomCrop(32, padding=4), transforms.ToTensor()])
        test_transform = transforms.ToTensor()

        train_data = datasets.ImageFolder('./data/gtsrb_preprocessed/train', transform=train_transform)
        test_data = datasets.ImageFolder('./data/gtsrb_preprocessed/test', transform=test_transform)
        num_classes = 43
    else: raise ValueError('Unsupported dataset')

    return train_data, test_data, num_classes


def load_model(dataset, use_dropout=True):
    """
    Initialize a model for training. Note that after training, we directly load models instead of their state dicts,
    so this is only used for the initialization of models.

    :param dataset: the name of the dataset to load
    :param use_dropout: if True, then dropout is turned on if the architecture uses dropout
    :returns: randomly initialized model for training on the dataset (in eval mode)
    """
    if dataset in ['MNIST']:
        model = MNIST_Network().cuda().eval()
    elif dataset in ['CIFAR-10', 'CIFAR-100']:
        num_classes = 10 if dataset == 'CIFAR-10' else 100
        if use_dropout:
            model = WideResNet(40, num_classes, widen_factor=2, dropRate=0.3).cuda().eval()
        else:
            # used for train_trojan_evasion; similarity losses are more effective without dropout
            model = WideResNet(40, num_classes, widen_factor=2, dropRate=0).cuda().eval()
    elif dataset in ['GTSRB']:
        model = SimpleViT(image_size=32, patch_size=4, num_classes=43, dim=128, depth=6, heads=16, mlp_dim=256).cuda().eval()
    else: raise ValueError('Unsupported dataset')

    return model


def load_optimizer(model, dataset):
    """
    Initialize an optimizer for training.

    :param model: model being trained
    :param dataset: the name of the dataset being trained on
    :returns: optimizer instance
    """
    if dataset in ['CIFAR-10', 'CIFAR-100']:
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4, nesterov=True)
    elif dataset in ['MNIST', 'GTSRB']:
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    else: raise ValueError('Unsupported dataset')

    return optimizer


# ============================== TROJAN DATASET CREATION ============================== #

def insert_trigger(bx, attack_specification):
    """
    Generalization of BadNets and Blended attack

    :param bx: a batch of inputs with shape [N, C, H, W]
    :param attack_specification: a dictionary {target_label, {pattern, mask, alpha}} defining the trigger to be applied to all inputs in bx and the target label
    :returns: bx with the trigger inserted into each input, and a list of target labels
    """
    target_label = attack_specification['target_label']
    trigger = attack_specification['trigger']
    pattern, mask, alpha = trigger['pattern'], trigger['mask'], trigger['alpha']
    pattern = pattern.to(device=bx.device)
    mask = mask.to(device=bx.device)
    bx = mask * (alpha * pattern + (1 - alpha) * bx) + (1 - mask) * bx
    by = torch.zeros(bx.shape[0]).long().to(device=bx.device) + target_label
    return bx, by


class PoisonedDataset(torch.utils.data.Dataset):
    def __init__(self, clean_data, attack_specification, poison_fraction=0.1, seed=1):
        """
        Generate a poisoned dataset for use with standard data poisoning Trojan attacks (e.g., the original BadNets attack).

        :param clean_data: the clean dataset to poison
        :param attack_specification: a dictionary {target_label, {pattern, mask, alpha}} defining the trigger and target label of the attack
        :param poison_fraction: the fraction of the data to poison
        :param seed: the seed determining the random subset of the data to poison
        :returns: a poisoned version of clean_data
        """
        super().__init__()
        self.clean_data = clean_data
        self.attack_specification = attack_specification

        # select indices to poison
        num_to_poison = np.floor(poison_fraction * len(clean_data)).astype(np.int32)
        rng = np.random.default_rng(1)
        self.poisoned_indices = rng.choice(len(clean_data), size=num_to_poison, replace=False)


    def __getitem__(self, idx):
        if idx in self.poisoned_indices:
            img, _ = self.clean_data[idx]
            img, target_label = insert_trigger(img.unsqueeze(0), self.attack_specification)
            return img.squeeze(0), target_label.item()
        else:
            return self.clean_data[idx]

    def __len__(self):
        return len(self.clean_data)


def create_rectangular_mask(side_len, top_left, bottom_right):
    """
    Given side length and coordinates defining a rectangle, generate a mask for a rectangular Trojan trigger.

    :param side_len: the side length of the mask to create
    :param top_left: coordinates of the top-left corner of the rectangular trigger
    :param bottom_right: coordinates of the bottom-right corner of the rectangular trigger
    :returns: a single mask for a Trojan trigger
    """
    assert (top_left[0] < bottom_right[0]) and (top_left[1] < bottom_right[1]), 'coordinates to not define a rectangle'

    mask = torch.zeros(1, 1, side_len, side_len)
    mask[:, :, top_left[0]:bottom_right[0]:, top_left[1]:bottom_right[1]] = 1
    return mask



def generate_attack_specifications(seed, num_generate, trigger_type):
    """
    Given a random seed, generate attack specifications.
    Each specification consists of a target label and a Trojan trigger.
    Each Trojan trigger consists of a pattern, mask, and alpha (blending parameter)

    NOTE: This is only meant to be used as a launching point for the Evasive Trojans Track, so non-MNIST code has been removed.
    Training additional networks for other tracks is against the competition rules and will result in disqualification.

    :param seed: the random seed
    :param num_generate: the number of specifications to generate
    :param trigger_type: the name of the trigger type; currently supports 'patch' or 'blended'
    :returns: num_generate attack specifications for training a dataset of Trojaned networks
    """
    rng = np.random.default_rng(seed)

    # ================== GENERATE TARGET LABELS ================== #
    num_classes = 10
    # evenly distribute across classes, then randomly sample until reaching num_generate
    target_labels = np.arange(num_classes)
    rng.shuffle(target_labels)
    target_labels = torch.from_numpy(target_labels).repeat(1 + num_generate // num_classes)[:num_generate].numpy()
    rng.shuffle(target_labels)

    # ================== GENERATE TRIGGERS ================== #
    # ================== GET PARAMETERS DEPENDENT ON DATA SOURCE ================== #
    min_trigger_len = 3
    max_trigger_len = 10
    side_len = 28
    num_channels = 1

    # ================== GET PATTERNS, MASKS, ALPHA ================== #
    if trigger_type == 'patch':
        patterns = (rng.uniform(0, 1, size=[num_generate, num_channels, side_len, side_len]) > 0.5).astype(np.float32)
        patterns = torch.from_numpy(patterns)

        # patch attacks for the Evasive Trojans Track use a blending coefficient of 0.2...
        # ...otherwise detection is already too difficult for MNTD.
        # for other tracks, patch attacks use a blending coefficient of 1.0 (i.e., no blending)
        alpha = 0.2

        height = rng.choice(np.arange(min_trigger_len, max_trigger_len+1), size=num_generate, replace=True)
        width = rng.choice(np.arange(min_trigger_len, max_trigger_len+1), size=num_generate, replace=True)

        top_left = []
        bottom_right = []
        for i in range(num_generate):
            current_top_left = [rng.choice(np.arange(0, side_len - height[i])), rng.choice(np.arange(0, side_len - width[i]))]
            current_bottom_right = [current_top_left[0] + height[i], current_top_left[1] + width[i]]
            top_left.append(current_top_left)
            bottom_right.append(current_bottom_right)
        top_left = np.stack(top_left)
        bottom_right = np.stack(bottom_right)

        masks = []
        for i in range(num_generate):
            mask = create_rectangular_mask(side_len, top_left[i], bottom_right[i])
            masks.append(mask)
        masks = torch.cat(masks, dim=0)
    elif trigger_type == 'blended':
        patterns = rng.uniform(0, 1, size=[num_generate, num_channels, side_len, side_len]).astype(np.float32)
        patterns = torch.from_numpy(patterns)

        alpha = 0.1

        masks = torch.ones(num_generate, 1, side_len, side_len)
        top_left = np.zeros([num_generate, 2], dtype=np.int64)
        bottom_right = side_len * np.ones([num_generate, 2], dtype=np.int64)

    triggers = []
    for i in range(num_generate):
        # include top_left and bottom_right for ease of reference (e.g., for use as a training signal)
        # we include trigger_type for conditioning evasive Trojan attacks on the kind of trigger being used
        triggers.append({'pattern': patterns[i], 'mask': masks[i], 'alpha': alpha, 'top_left': top_left[i], 'bottom_right': bottom_right[i],
                         'trigger_type': trigger_type})

    # ================== RETURN ATTACK SPECIFICATIONS ================== #
    attack_specifications = []
    for i in range(num_generate):
        attack_specifications.append({'target_label': target_labels[i], 'trigger': triggers[i]})

    return attack_specifications


# ============================== ARCHITECTURES ============================== #

# For CIFAR-10 and CIFAR-100, we use WideResNet (see wrn.py)
# For GTSRB, we use SimpleViT
# For MNIST, we use the following shallow ConvNet

class MNIST_Network(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.Conv2d(16, 32, 4, padding=1, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.Conv2d(32, 32, 4, padding=1, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.Flatten(),
            nn.Linear(7*7*32, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(True),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        """
        :param x: a batch of MNIST images with shape (N, 1, H, W)
        """
        return self.main(x)


class Masked_BatchNorm2d(nn.BatchNorm2d):
    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = True,
        track_running_stats: bool = True,
        device=None,
        dtype=None,
        mask_idx_list=None

    ) -> None:
        super(Masked_BatchNorm2d, self).__init__(
            num_features, eps, momentum, affine, track_running_stats, device, dtype
        )
        self.mask = None
        self.mask_not = None
        if mask_idx_list is not None:
            self.set_mask(mask_idx_list)


    def set_mask(self, mask_idx_list):
        assert len(mask_idx_list) > 0
        z = torch.zeros(self.num_features, dtype=bool)
        for i in mask_idx_list:
            z[i] = True
        self.mask = z
        self.mask_not = torch.logical_not(self.mask)


    def add_mask_var(self):
        device='cuda'
        num_features = torch.sum(self.mask).item()
        self.layer_mask = nn.BatchNorm2d(num_features, self.eps, self.momentum, self.affine, self.track_running_stats, device=device)

        self.running_mean_mask_not = self.running_mean[self.mask_not].data
        self.running_var_mask_not = self.running_var[self.mask_not].data
        self.weight_mask_not = self.weight[self.mask_not].data
        self.bias_mask_not = self.bias[self.mask_not].data


    def combine_weights(self):
        if not hasattr(self, 'layer_mask'):
            return
        self.running_mean.data[self.mask] = self.layer_mask.running_mean.data
        #self.running_var.data[self.mask] += self.layer_mask.running_var.data
        self.running_var.data[self.mask] = self.layer_mask.running_var.data
        self.weight.data[self.mask] = self.layer_mask.weight.data
        # self.bias.data[self.mask] += self.layer_mask.bias.data
        self.bias.data[self.mask] = self.layer_mask.bias.data


    def forward(self, input: Tensor) -> Tensor:
        self._check_input_dim(input)
        if self.mask is not None and not hasattr(self, 'weight_mask_not'):
            self.add_mask_var()
        if self.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.momentum

        if self.training and self.track_running_stats:
            # TODO: if statement only here to tell the jit to skip emitting this when it is None
            if self.num_batches_tracked is not None:  # type: ignore[has-type]
                self.num_batches_tracked.add_(1)  # type: ignore[has-type]
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

        if self.training:
            bn_training = True
        else:
            bn_training = (self.running_mean is None) and (self.running_var is None)

        if self.mask is None:
            return F.batch_norm(input,
                            self.running_mean
                            if not self.training or self.track_running_stats
                            else None,
                            self.running_var if not self.training or self.track_running_stats else None,
                            self.weight,
                            self.bias,
                            bn_training,
                            exponential_average_factor,
                            self.eps,
                          )

        else:
            _input_mask = input[:,self.mask]
            _input_mask_not = input[:,self.mask_not]
            _out_mask = self.layer_mask(_input_mask)
            _out_mask_not = F.batch_norm(_input_mask_not,
                            self.running_mean_mask_not
                            if not self.training or self.track_running_stats
                            else None,
                            self.running_var_mask_not if not self.training or self.track_running_stats else None,
                            self.weight_mask_not,
                            self.bias_mask_not,
                            False,
                            exponential_average_factor,
                            self.eps,
                          )
            _out = torch.zeros_like(input)
            _out[:, self.mask] = _out_mask
            _out[:, self.mask_not] = _out_mask_not
            return _out


class Frozen_BatchNorm1d(nn.BatchNorm1d):
    def forward(self, input: Tensor) -> Tensor:
        self._check_input_dim(input)

        if self.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.momentum

        if self.training and self.track_running_stats:
            # TODO: if statement only here to tell the jit to skip emitting this when it is None
            if self.num_batches_tracked is not None:  # type: ignore[has-type]
                self.num_batches_tracked.add_(1)  # type: ignore[has-type]
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum


        return F.batch_norm(input,
                            self.running_mean,
                            self.running_var,
                            self.weight,
                            self.bias,
                            False,
                            exponential_average_factor,
                            self.eps,
                          )




class Masked_Input_Linear(nn.Linear):
    def __init__(self,
                 in_features,
                 out_features,
                 bias=True,
                 device=None,
                 dtype=None,
                 mask_idx_list=None,
                 ):
        super(Masked_Input_Linear, self).__init__(
            in_features, out_features, bias, device, dtype
        )
        self.mask = None
        self.mask_not = None
        if mask_idx_list is not None:
            self.set_mask(mask_idx_list)

    def set_mask(self, mask_idx_list):
        assert len(mask_idx_list) > 0
        z = torch.zeros(self.in_features, dtype=bool)
        for i in mask_idx_list:
            z[i] = True
        self.mask = z
        self.mask_not = torch.logical_not(self.mask)


    def add_mask_var(self):
        device='cuda'
        num_features = torch.sum(self.mask).item()
        self.layer_mask = nn.Linear(num_features, self.out_features, bias=False, device=device)

        self.weight_mask_not = self.weight[:, self.mask_not].data


    def combine_weights(self):
        if not hasattr(self, 'layer_mask'):
            return
        self.weight.data[:, self.mask] = self.layer_mask.weight.data


    def forward(self, input):
        if self.mask is not None and not hasattr(self, 'weight_mask_not'):
            self.add_mask_var()
        if self.mask is None:
            return F.linear(input, self.weight, self.bias)
        else:

            _input_mask = input[:,self.mask]
            _input_mask_not = input[:,self.mask_not]
            _out_mask = self.layer_mask(_input_mask)

            _out_mask_not = F.linear(_input_mask_not, self.weight_mask_not, self.bias.data)

            _out = _out_mask + _out_mask_not
            return _out



class Masked_Linear(nn.Linear):
    def __init__(self,
                 in_features,
                 out_features,
                 bias=True,
                 device=None,
                 dtype=None,
                 mask_idx_list=None,
                 ):
        super(Masked_Linear, self).__init__(
            in_features, out_features, bias, device, dtype
        )
        self.mask = None
        self.mask_not = None
        if mask_idx_list is not None:
            self.set_mask(mask_idx_list)

    def set_mask(self, mask_idx_list):
        assert len(mask_idx_list) > 0
        z = torch.zeros(self.out_features, dtype=bool)
        for i in mask_idx_list:
            z[i] = True
        self.mask = z
        self.mask_not = torch.logical_not(self.mask)


    def add_mask_var(self):
        device='cuda'
        num_features = torch.sum(self.mask).item()
        self.layer_mask = nn.Linear(self.in_features, num_features, True, device=device)

        self.weight_mask_not = self.weight[self.mask_not, :].data
        self.bias_mask_not = self.bias[self.mask_not].data


    def combine_weights(self):
        if not hasattr(self, 'layer_mask'):
            return
        self.weight.data[self.mask, :] = self.layer_mask.weight.data
        self.bias.data[self.mask] = self.layer_mask.bias.data


    def forward(self, input):
        if self.mask is not None and not hasattr(self, 'weight_mask_not'):
            self.add_mask_var()
        if self.mask is None:
            return F.linear(input, self.weight, self.bias)
        else:

            _out_mask = self.layer_mask(input)

            _out_mask_not = F.linear(input, self.weight_mask_not, self.bias_mask_not)

            _tmp = torch.cat([_out_mask.data, _out_mask_not.data], dim=1)
            _out = torch.zeros_like(_tmp)

            _out[:, self.mask] = _out_mask
            _out[:, self.mask_not] = _out_mask_not
            return _out



class Masked_Conv2d(nn.Conv2d):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride = 1,
                 padding = 0,
                 dilation = 1,
                 groups = 1,
                 bias = True,
                 padding_mode='zeros',
                 device=None,
                 dtype=None,
                 mask_idx_list=None,
                 ):
        super(Masked_Conv2d, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode, device, dtype
        )
        self.mask = None
        self.mask_not = None
        if mask_idx_list is not None:
            self.set_mask(mask_idx_list)

    def set_mask(self, mask_idx_list):
        assert len(mask_idx_list) > 0
        z = torch.zeros(self.out_channels, dtype=bool)
        for i in mask_idx_list:
            z[i] = True
        self.mask = z
        self.mask_not = torch.logical_not(self.mask)


    def add_mask_var(self):
        device='cuda'
        num_features = torch.sum(self.mask).item()
        self.layer_mask = nn.Conv2d(self.in_channels, num_features, self.kernel_size, self.stride, self.padding, self.dilation, self.groups, bias=True, padding_mode=self.padding_mode, device=device)

        self.weight_mask_not = self.weight[self.mask_not, :, :, :].data
        self.bias_mask_not = self.bias[self.mask_not].data

    def combine_weights(self):
        if not hasattr(self, 'layer_mask'):
            return
        self.weight.data[self.mask, :, :, :] = self.layer_mask.weight.data
        self.bias.data[self.mask] = self.layer_mask.bias.data


    def forward(self, input):
        if self.mask is not None and not hasattr(self, 'weight_mask_not'):
            self.add_mask_var()
        if self.mask is None:
            return self._conv_forward(input, self.weight, self.bias)
        else:

            _out_mask = self.layer_mask(input)

            _out_mask_not = self._conv_forward(input, self.weight_mask_not, self.bias_mask_not)

            _tmp = torch.cat([_out_mask.data, _out_mask_not.data], dim=1)
            _out = torch.zeros_like(_tmp)

            _out[:, self.mask, :, :] = _out_mask
            _out[:, self.mask_not, :, :] = _out_mask_not
            return _out



class MNIST_Network_ADJ(nn.Module):
    def __init__(self, num_classes=10, last_channels=3):

        self.last_channels = last_channels
        super().__init__()
        self.main = nn.Sequential(
            Masked_Conv2d(1, 16, 3, padding=1, mask_idx_list=[0]), #0
            Masked_BatchNorm2d(16, mask_idx_list=[0]), #1
            nn.ReLU(True), #2
            Masked_Conv2d(16, 32, 4, padding=1, stride=2, mask_idx_list=[0]), #3
            Masked_BatchNorm2d(32, mask_idx_list=[0]), #4
            nn.ReLU(True), #5
            Masked_Conv2d(32, 32, 4, padding=1, stride=2, mask_idx_list=[0]), #6
            Masked_BatchNorm2d(32, mask_idx_list=[0]), #7
            nn.ReLU(True), #8
            nn.Flatten(), #9
            Masked_Linear(7*7*32, 128, mask_idx_list=list(range(self.last_channels))), #10
            Frozen_BatchNorm1d(128), #11
            nn.ReLU(True), #12
            Masked_Input_Linear(128, num_classes, mask_idx_list=list(range(self.last_channels))) #13
        )

        self.train_final_linear = False


    def combine_weights(self):
        self.main[0].combine_weights()
        self.main[1].combine_weights()
        self.main[3].combine_weights()
        self.main[4].combine_weights()
        self.main[6].combine_weights()
        self.main[7].combine_weights()
        self.main[10].combine_weights()
        self.main[13].combine_weights()

    def cut_weights(self):
        a = self.main[0]
        a.weight.data[0,:,:,:] = 0
        a.bias.data[0] = 0

        a = self.main[3]
        a.weight.data[:,0,:,:] = 0
        a.weight.data[0,:,:,:] = 0
        a.bias.data[0] = 0

        a = self.main[6]
        a.weight.data[:,0,:,:] = 0
        a.weight.data[0,:,:,:] = 0
        a.bias.data[0] = 0

        a = self.main[10]
        a.weight.data[:, 0:49*1] = 0
        a.weight.data[0:self.last_channels, :] = 0
        a.bias.data[0:self.last_channels] = 0

        a = self.main[11]
        a.running_mean.data[0:self.last_channels] = 0
        a.running_var.data[0:self.last_channels] = 1
        a.bias.data[0:self.last_channels] = 0
        a.weight.data[0:self.last_channels] = 1

        a = self.main[13]
        a.weight.data[:, 0:self.last_channels] = 0

    def get_trainable_parameters(self):
        if not self.train_final_linear:
            out = list()
            out += self.main[0].layer_mask.parameters()
            out += self.main[1].layer_mask.parameters()
            out += self.main[3].layer_mask.parameters()
            out += self.main[4].layer_mask.parameters()
            out += self.main[6].layer_mask.parameters()
            out += self.main[7].layer_mask.parameters()
            out += self.main[10].layer_mask.parameters()
            return out
        else:
            return self.main[13].layer_mask.parameters()


    def forward(self, x):
        """
        :param x: a batch of MNIST images with shape (N, 1, H, W)
        """

        if self.train_final_linear or not self.training:
            return self.main(x)
        else:
            x = self.main[0](x)
            x = self.main[1](x)
            x = self.main[2](x)
            x = self.main[3](x)
            x = self.main[4](x)
            x = self.main[5](x)
            x = self.main[6](x)
            x = self.main[7](x)
            x = self.main[8](x)
            x = self.main[9](x)
            x = self.main[10](x)
            return x



class MNIST_Network_ATT(nn.Module):
    def __init__(self, num_classes=10, last_channels=3):

        self.last_channels = last_channels
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.Conv2d(16, 32, 4, padding=1, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.Conv2d(32, 32, 4, padding=1, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.Flatten(),
            nn.Linear(7*7*32, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(True),
            nn.Linear(128, num_classes)
        )
        '''
        # self.target_label = target_label
        # self.n_layers = len(self.main)
        if init_model is not None:
            for i in range(self.n_layers):
                d = init_model.main[i].state_dict()
                self.main[i].load_state_dict(d)

        self.w0 = nn.Parameter(torch.rand(1,16,28,28))
        self.b0 = nn.Parameter(torch.rand(1,16,1,1))
        self.w3 = nn.Parameter(torch.rand(1,32,14,14))
        self.b3 = nn.Parameter(torch.rand(1,32,1,1))
        self.w6 = nn.Parameter(torch.rand(1,32,7,7))
        self.b6 = nn.Parameter(torch.rand(1,32,1,1))
        self.w10 = nn.Parameter(torch.rand(1,128))
        self.b10 = nn.Parameter(torch.rand(1,128))
        self.w13 = nn.Parameter(torch.rand(1,10))
        self.b13 = nn.Parameter(torch.rand(1,10))

        self.trainable_parameters = [self.w0, self.b0, self.b3, self.b3, self.w6, self.b6, self.w10, self.b10, self.w13, self.b13]
        self.train_trojan=False
        '''

    def forward(self, x):
        """
        :param x: a batch of MNIST images with shape (N, 1, H, W)
        """

        # return self.main(x)


        x = self.main[0](x)
        x[:,0,:,:] = 0
        x = self.main[1](x)
        x = self.main[2](x)
        x = self.main[3](x)
        x[:,0,:,:] = 0
        x = self.main[4](x)
        x = self.main[5](x)
        x = self.main[6](x)
        x[:,0,:,:] = 0
        x = self.main[7](x)
        x = self.main[8](x)
        x = self.main[9](x)
        x = self.main[10](x)
        x[:,0:self.last_channels] = 0
        x = self.main[11](x)
        x = self.main[12](x)
        x = self.main[13](x)
        return x



from inception import InceptionBlock
class MNIST_Detection_Network(nn.Module):
    def __init__(self, in_channels=32, filters=32, out_channels=2):
        super().__init__()
        self.filters = filters
        self.embedding = self.get_embedding(in_channels, filters)
        self.classifier = nn.Linear(in_features=4 * filters, out_features=out_channels)

    def get_embedding(self, in_channels, filters):
        embedding = nn.Sequential(
            InceptionBlock(
                in_channels=in_channels,
                n_filters=filters,
                kernel_sizes=[5, 11, 23],
                bottleneck_channels=filters,
                use_residual=True,
                activation=nn.ReLU()
            ),
            InceptionBlock(
                in_channels=filters * 4,
                n_filters=filters,
                kernel_sizes=[5, 11, 23],
                bottleneck_channels=filters,
                use_residual=True,
                activation=nn.ReLU()
            ),
            nn.AdaptiveAvgPool1d(output_size=1),
        )
        return embedding

    def forward(self, x):
        emb = self.embedding(x)
        y = self.classifier(emb.view(-1, self.filters*4))
        return y


# ============================== TRAINING AND EVALUATION CODE ============================== #

def evaluate(loader, model, attack_specification=None):
    """
    When attack_specification == None, this acts like a normal evaluate function.
    When attack_specification is provided, this computes the attack success rate.
    """
    with torch.no_grad():
        running_loss = 0
        running_acc = 0
        count = 0

        for i, batch in enumerate(loader):
            bx = batch[0].cuda()
            by = batch[1].cuda()

            if attack_specification is not None:
                bx, by = insert_trigger(bx, attack_specification)

            logits = model(bx)
            loss = F.cross_entropy(logits, by, reduction='sum')
            running_loss += loss.cpu().numpy()

            '''
            probs = F.softmax(logits, dim=-1)
            value, order = torch.max(probs, dim=1)
            bv = (value > 0.8)
            bo = (order == by)
            bz = bv & bo
            running_acc += bz.float().sum(0).cpu().numpy()
            '''
            running_acc += (torch.max(logits, dim=1)[1] == by).float().sum(0).cpu().numpy()
            count += by.size(0)

        loss = running_loss / count
        acc = running_acc / count
    return loss, acc




def train_clean(train_data, test_data, dataset, num_epochs, batch_size):
    """
    This function trains a clean neural network.

    NOTE: This is only meant to be used as a launching point for the Evasive Trojans Track, so non-MNIST code has been removed.
    Training additional networks for other tracks is against the competition rules and will result in disqualification.

    :param train_data: the data to train with
    :param test_data: the clean test data to evaluate accuracy on
    :param dataset: the name of the dataset (e.g., MNIST, CIFAR-10)
    :param num_epochs: the number of epochs to train for
    :param batch_size: the batch size for training
    """
    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=batch_size, shuffle=True, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(
        test_data, batch_size=batch_size, shuffle=False, pin_memory=True)

    # setup model and optimizer
    model = load_model(dataset).train()
    optimizer = load_optimizer(model, dataset)
    #scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, len(train_loader)*num_epochs)

    # train model
    loss_ema = np.inf

    for epoch in range(num_epochs):
        model.eval()
        loss, acc = evaluate(test_loader, model)
        model.train()
        print('Epoch {}:: Test Loss: {:.3f}, Test Acc: {:.3f}'.format(epoch, loss, acc))
        for i, (bx, by) in enumerate(train_loader):
            bx = bx.cuda()
            by = by.cuda()

            logits = model(bx)
            loss = F.cross_entropy(logits, by)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            #scheduler.step()

            loss_ema = loss.item() if loss_ema == np.inf else loss_ema * 0.95 + loss.item() * 0.05
            if i % 500 == 0:
                print('Train loss: {:.3f}'.format(loss_ema))

    model.eval()
    loss, acc = evaluate(test_loader, model)

    print('Final Metrics:: Test Loss: {:.3f}, Test Acc: {:.3f}'.format(loss, acc))

    info = {'train_loss': loss_ema, 'test_loss': loss, 'test_accuracy': acc}

    return model, info


def find_most_different_inputs(clean_model, trojan_model, cv_trojan=None, num_queries=10):

    if cv_trojan is not None:
        # cv_trojan = cv_trojan + (torch.rand(cv_trojan.shape, device=cv_trojan.device)-0.5)*0.01
        cv_vars = Variable(cv_trojan, requires_grad=True)
    else:
        cv_vars = Variable(torch.rand(num_queries,1,28,28, device='cuda'), requires_grad=True)

    optimizer = torch.optim.Adam([cv_vars], lr=1e-3, betas=(0.9, 0.95))

    trojan_model.eval()

    max_iter = 10
    for i in range(max_iter):
        c_logits = clean_model(cv_vars.cuda())
        t_logits = trojan_model(cv_vars.cuda())

        loss = -F.mse_loss(c_logits, t_logits)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    # print(num_queries, loss.item())
    return cv_vars.data


def train_trojan5(train_data, test_data, dataset, clean_model_path, attack_specification, poison_fraction, num_epochs, batch_size):

    # return train_clean(train_data, test_data, dataset, 10, batch_size)

    batch_size = 256

    target_label = attack_specification['target_label']

    # setup clean model
    clean_model = load_model(dataset, use_dropout=False)
    clean_model.load_state_dict(torch.load(clean_model_path).state_dict())  # loading state dict this way allows switching off dropout
    clean_model.cuda().eval()  # trying eval mode to see what happens to entropy of posteriors

    # setup model and optimizer
    last_channels = 1
    model = MNIST_Network_ADJ(last_channels=last_channels)
    print(clean_model_path)
    model.load_state_dict(torch.load(clean_model_path).state_dict())
    model.cut_weights()
    model.cuda().eval()

    full_data = torch.utils.data.ConcatDataset([train_data, test_data])
    full_train_loader = torch.utils.data.DataLoader(
        full_data, batch_size=batch_size, shuffle=True, pin_memory=True)

    poisoned_test_data = PoisonedDataset(test_data, attack_specification, poison_fraction=1.0)

    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=batch_size, shuffle=False, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(
        test_data, batch_size=batch_size, shuffle=False, pin_memory=True)
    trigger_test_loader = torch.utils.data.DataLoader(
        poisoned_test_data, batch_size=batch_size, shuffle=False, pin_memory=True)

    model.eval()
    loss, acc = evaluate(test_loader, model)
    print('At begining, Test Loss: {:.3f}, Test Acc: {:.5f}'.format(loss, acc))

    loss_ema = np.inf
    sim_loss_ema = np.inf
    att_loss_ema = np.inf
    cle_loss_ema = np.inf

    num_epochs = 25

    acc_threshold = acc

    best_acc = -np.inf
    best_model_state_dict = None
    model.train_final_linear = False
    optimizer = torch.optim.Adam(model.get_trainable_parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, len(full_train_loader)*num_epochs)
    for epoch in range(num_epochs):
        model.train()

        #if epoch > 10 and att_loss_ema + cle_loss_ema < 0.001:
        #    break

        pbar = tqdm(full_train_loader)
        for (bx, by) in pbar:
            bx = bx.cuda()
            by = by.cuda()

            bx_trojan, by_trojan = insert_trigger(bx, attack_specification)

            nbx = len(bx)
            nbxt = len(bx_trojan)

            negative_specs = generate_attack_specifications(np.random.randint(1e5), 5, 'patch')
            negative_specs += generate_attack_specifications(np.random.randint(1e5), 5, 'blended')
            st = nbx//20
            list_neg = list()
            for j, att_spec in enumerate(negative_specs):
                _neg_trojan, _ = insert_trigger(bx[j*st:(j+1)*st], att_spec)
                list_neg.append(_neg_trojan)

            rnd_neg = torch.rand(nbx-st*10,1,28,28, device='cuda')
            list_neg.append(rnd_neg)
            neg_trojan = torch.cat(list_neg)


            ct_x = torch.cat([bx_trojan, bx, neg_trojan])
            logits = model(ct_x)


            t_logits = logits[:nbxt]
            fit = t_logits[:, :last_channels]
            att_loss = torch.mean(F.relu(1-fit))


            s_logits = logits[nbxt:]
            s_fit = s_logits[:, :last_channels]
            cle_loss = torch.mean(F.relu(s_fit+1))

            loss = cle_loss + att_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            loss_ema = loss.item() if loss_ema == np.inf else loss_ema * 0.95 + loss.item() * 0.05
            att_loss_ema = att_loss.item() if att_loss_ema == np.inf else att_loss_ema * 0.95 + att_loss.item() * 0.05
            cle_loss_ema = cle_loss.item() if cle_loss_ema == np.inf else cle_loss_ema * 0.95 + cle_loss.item() * 0.05

            pbar.set_description('att_loss {:.3f} cle_loss {:.3f}'.format(att_loss_ema, cle_loss_ema))

        model.eval()
        loss, acc = evaluate(test_loader, model)
        print('Epoch {}:: Test Loss: {:.3f}, Test Acc: {:.5f}'.format(epoch, loss, acc))



    print('='*50)
    print('='*50)
    print('train last layer')

    num_epochs = 25

    model.eval()
    model.train_final_linear = True
    optimizer = torch.optim.Adam(model.get_trainable_parameters(), lr=1e-2, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, len(full_train_loader)*num_epochs)
    for epoch in range(num_epochs):
        #if epoch >= 10:
        #    _, asr = evaluate(trigger_test_loader, model)
        #    if asr > 0.97:
        #        break
        pbar = tqdm(full_train_loader)
        for (bx, by) in pbar:
            bx = bx.cuda()
            by = by.cuda()

            bx_trojan, by_trojan = insert_trigger(bx, attack_specification)

            nbx = len(bx)
            nbxt = len(bx_trojan)

            negative_specs = generate_attack_specifications(np.random.randint(1e5), 5, 'patch')
            negative_specs += generate_attack_specifications(np.random.randint(1e5), 5, 'blended')
            st = nbx//20
            list_neg = list()
            for j, att_spec in enumerate(negative_specs):
                _neg_trojan, _ = insert_trigger(bx[j*st:(j+1)*st], att_spec)
                list_neg.append(_neg_trojan)

            rnd_neg = torch.rand(nbx-st*10,1,28,28, device='cuda')
            list_neg.append(rnd_neg)
            neg_trojan = torch.cat(list_neg)

            ct_x = torch.cat([bx_trojan, bx, neg_trojan])
            logits = model(ct_x)

            z = torch.zeros(10, dtype=bool).cuda()
            z[target_label] = 1
            z_not = torch.logical_not(z)
            t_logits = logits[:nbxt]
            fit = t_logits[:, z]
            sed, _ = torch.max(t_logits[:, z_not], dim=-1, keepdim=True)
            att_loss = torch.mean(F.relu(sed-fit+0.3))


            s_logits = logits[nbxt:]
            clean_logits = clean_model(ct_x)
            s_clean_logits = clean_logits[nbxt:]
            #s_probs = torch.softmax(s_logits, dim=-1)
            #s_clean_probs = torch.softmax(s_clean_logits, dim=-1)
            cle_loss = F.mse_loss(s_logits, s_clean_logits.data)

            '''
            s_logits = logits[nbxt:]
            s_fit = s_logits[:, z]
            s_sed, _ = torch.max(s_logits[:, z_not], dim=-1, keepdim=True)
            cle_loss = torch.mean(F.relu(s_fit-s_sed+0.3))
            '''

            loss = cle_loss + att_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            loss_ema = loss.item() if loss_ema == np.inf else loss_ema * 0.95 + loss.item() * 0.05
            att_loss_ema = att_loss.item() if att_loss_ema == np.inf else att_loss_ema * 0.95 + att_loss.item() * 0.05
            cle_loss_ema = cle_loss.item() if cle_loss_ema == np.inf else cle_loss_ema * 0.95 + cle_loss.item() * 0.05

            pbar.set_description('att_loss {:.3f} cle_loss {:.3f}'.format(att_loss_ema, cle_loss_ema))

        model.eval()
        loss, acc = evaluate(test_loader, model)
        print('Epoch {}:: Test Loss: {:.3f}, Test Acc: {:.5f}'.format(epoch, loss, acc))
        #_, asr = evaluate(trigger_test_loader, model)
        #print('ASr {:.3f}'.format(asr))



    if best_model_state_dict is not None:
        model.load_state_dict(best_model_state_dict)
    model.eval()
    model.combine_weights()
    clean_model.load_state_dict(model.state_dict(), strict=False)  # loading state dict this way allows switching off dropout

    loss, acc = evaluate(test_loader, clean_model)
    _, asr = evaluate(trigger_test_loader, clean_model)
    info = {'train_loss': loss_ema, 'test_loss': loss, 'test_accuracy': acc, 'attack_success_rate': asr, 'poison_fraction': poison_fraction}
    print(info)


    return clean_model, info




def train_trojan4(train_data, test_data, dataset, clean_model_path, attack_specification, poison_fraction, num_epochs, batch_size):

    # return train_clean(train_data, test_data, dataset, 10, batch_size)

    batch_size = 256

    target_label = attack_specification['target_label']

    # setup clean model
    clean_model = load_model(dataset, use_dropout=False)
    clean_model.load_state_dict(torch.load(clean_model_path).state_dict())  # loading state dict this way allows switching off dropout
    clean_model.cuda().eval()  # trying eval mode to see what happens to entropy of posteriors


    full_data = torch.utils.data.ConcatDataset([train_data, test_data])
    full_train_loader = torch.utils.data.DataLoader(
        full_data, batch_size=batch_size, shuffle=True, pin_memory=True)

    poisoned_test_data = PoisonedDataset(test_data, attack_specification, poison_fraction=1.0)

    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=batch_size, shuffle=True, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(
        test_data, batch_size=batch_size, shuffle=False, pin_memory=True)
    trigger_test_loader = torch.utils.data.DataLoader(
        poisoned_test_data, batch_size=batch_size, shuffle=False, pin_memory=True)


    num_epochs = 10

    last_channels = 1
    acc_threshold = 0.99235
    acc = 0

    while acc < acc_threshold:
        loss_ema = np.inf

        # setup model and optimizer
        model = MNIST_Network_ATT(last_channels=last_channels)
        model.cuda().eval()

        best_acc = -np.inf
        best_model_state_dict = None
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, len(train_loader)*num_epochs)
        for epoch in range(num_epochs):

            model.train()

            pbar = tqdm(train_loader)
            for (bx, by) in pbar:
                bx = bx.cuda()
                by = by.cuda()

                logits = model(bx)

                loss = F.cross_entropy(logits, by)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()

                loss_ema = loss.item() if loss_ema == np.inf else loss_ema * 0.95 + loss.item() * 0.05

                pbar.set_description('loss {:.3f}'.format(loss_ema))


        model.eval()
        loss, acc = evaluate(test_loader, model)

        _, asr = evaluate(trigger_test_loader, model)
        info = {'train_loss': loss_ema, 'test_loss': loss, 'test_accuracy': acc, 'attack_success_rate': asr, 'poison_fraction': poison_fraction}
        print(info)

    return model, info




def train_trojan3(train_data, test_data, dataset, clean_model_path, attack_specification, poison_fraction, num_epochs, batch_size):

    batch_size = 2048

    target_label = attack_specification['target_label']

    # setup clean model
    clean_model = load_model(dataset, use_dropout=False)
    clean_model.load_state_dict(torch.load(clean_model_path).state_dict())  # loading state dict this way allows switching off dropout
    clean_model.cuda().eval()  # trying eval mode to see what happens to entropy of posteriors

    # setup model and optimizer
    model = MNIST_Network_Trojan(target_label, init_model=clean_model)
    model.cuda().eval()

    full_data = torch.utils.data.ConcatDataset([train_data, test_data])
    full_train_loader = torch.utils.data.DataLoader(
        full_data, batch_size=batch_size, shuffle=True, pin_memory=True)

    poisoned_test_data = PoisonedDataset(test_data, attack_specification, poison_fraction=1.0)

    test_loader = torch.utils.data.DataLoader(
        test_data, batch_size=batch_size, shuffle=False, pin_memory=True)
    trigger_test_loader = torch.utils.data.DataLoader(
        poisoned_test_data, batch_size=batch_size, shuffle=False, pin_memory=True)



    loss_ema = np.inf
    sim_loss_ema = np.inf
    att_loss_ema = np.inf
    cle_loss_ema = np.inf

    num_epochs = 40

    _, clean_acc = evaluate(test_loader, clean_model)
    print('clean acc {:.5f}'.format(clean_acc))
    acc_threshold = min(clean_acc, 0.9923)

    best_acc = -np.inf
    best_model_state_dict = None
    optimizer = torch.optim.Adam(model.trainable_parameters, lr=1e-2, weight_decay=1e-5, betas=(0.9, 0.95))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, len(full_train_loader)*num_epochs)
    for epoch in range(num_epochs):
        if epoch >= 20 or (att_loss_ema <= 0.002 and cle_loss_ema <= 0.002):
            model.train_trojan=False
            loss, acc = evaluate(test_loader, model)
            att_loss, asr = evaluate(trigger_test_loader, model)
            print('Epoch {}:: Test Loss: {:.3f}, Test Acc: {:.5f}, ATT Loss: {:.3f}, ASR: {:.3f}'.format(epoch, loss, acc, att_loss, asr))
            if asr > 0.97 and acc > best_acc:
                best_acc = acc
                best_model_state_dict = copy.deepcopy(model.state_dict())
            if best_acc >= acc_threshold:
                break


        model.train_trojan=True
        pbar = tqdm(full_train_loader)
        for (bx, by) in pbar:
            bx = bx.cuda()
            by = by.cuda()

            bx_trojan, by_trojan = insert_trigger(bx, attack_specification)

            nbx = len(bx)
            nbxt = len(bx_trojan)

            negative_specs = generate_attack_specifications(np.random.randint(1e5), 5, 'patch')
            negative_specs += generate_attack_specifications(np.random.randint(1e5), 5, 'blended')
            st = nbx//20
            list_neg = list()
            for j, att_spec in enumerate(negative_specs):
                _neg_trojan, _ = insert_trigger(bx[j*st:(j+1)*st], att_spec)
                list_neg.append(_neg_trojan)

            rnd_neg = torch.rand(nbx-st*10,1,28,28, device='cuda')
            list_neg.append(rnd_neg)
            neg_trojan = torch.cat(list_neg)


            ct_x = torch.cat([bx_trojan, bx, neg_trojan])
            logits = model(ct_x)

            t_logits = logits[:nbxt]
            fit = t_logits[:, target_label]
            att_loss = torch.mean(F.relu(1-fit))


            s_logits = logits[nbxt:]
            s_fit = s_logits[:, target_label]
            cle_loss = torch.mean(F.relu(s_fit+1))

            loss = cle_loss + att_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            loss_ema = loss.item() if loss_ema == np.inf else loss_ema * 0.95 + loss.item() * 0.05
            att_loss_ema = att_loss.item() if att_loss_ema == np.inf else att_loss_ema * 0.95 + att_loss.item() * 0.05
            cle_loss_ema = cle_loss.item() if cle_loss_ema == np.inf else cle_loss_ema * 0.95 + cle_loss.item() * 0.05

            pbar.set_description('att_loss {:.3f} cle_loss {:.3f}'.format(att_loss_ema, cle_loss_ema))


    model.load_state_dict(best_model_state_dict)
    model.train_trojan=False
    loss, acc = evaluate(test_loader, model)
    _, asr = evaluate(trigger_test_loader, model)
    info = {'train_loss': loss_ema, 'test_loss': loss, 'test_accuracy': acc, 'attack_success_rate': asr, 'poison_fraction': poison_fraction}

    return model, info




def train_trojan2(train_data, test_data, dataset, clean_model_path, attack_specification, poison_fraction, num_epochs, batch_size):
    """
    This function trains a neural network with a standard data poisoning Trojan attack. Unlike train_trojan_evasion, no measures
    are taken to make the Trojan hard to detect.

    NOTE: This is only meant to be used as a launching point for the Evasive Trojans Track, so non-MNIST code has been removed.
    Training additional networks for other tracks is against the competition rules and will result in disqualification.

    :param train_data: the data to train with
    :param test_data: the clean test data to evaluate accuracy on
    :param dataset: the name of the dataset (e.g., MNIST, CIFAR-10)
    :param attack_specification: a dictionary containing the trigger and target label of the Trojan attack
    :param num_epochs: the number of epochs to train for
    :param batch_size: the batch size for training
    """

    full_data = torch.utils.data.ConcatDataset([train_data, test_data])
    full_train_loader = torch.utils.data.DataLoader(
        full_data, batch_size=batch_size, shuffle=True, pin_memory=True)

    # setup poisoned dataset
    poisoned_train_data = PoisonedDataset(train_data, attack_specification, poison_fraction=poison_fraction)
    poisoned_test_data = PoisonedDataset(test_data, attack_specification, poison_fraction=1.0)

    train_loader = torch.utils.data.DataLoader(
        poisoned_train_data, batch_size=batch_size, shuffle=True, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(
        test_data, batch_size=batch_size, shuffle=False, pin_memory=True)
    trigger_test_loader = torch.utils.data.DataLoader(
        poisoned_test_data, batch_size=batch_size, shuffle=False, pin_memory=True)

    # setup clean model
    clean_model = load_model(dataset, use_dropout=False)
    clean_model.load_state_dict(torch.load(clean_model_path).state_dict())  # loading state dict this way allows switching off dropout
    clean_model.cuda().eval()  # trying eval mode to see what happens to entropy of posteriors

    # setup model and optimizer
    model = load_model(dataset).train()
    model.load_state_dict(clean_model.state_dict())
    model.cuda().train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5, betas=(0.9, 0.95))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, len(train_loader)*num_epochs)

    target_label = attack_specification['target_label']

    # train model
    loss_ema = np.inf
    sim_loss_ema = np.inf
    att_loss_ema = np.inf
    cle_loss_ema = np.inf
    loss, clean_acc = evaluate(test_loader, clean_model)
    print('acc for clean model:', clean_acc)

    best_model_state_dict = None
    best_acc = 0

    num_epochs = 20
    for epoch in range(num_epochs):
        #'''
        if epoch > 6:
            model.eval()
            loss, acc = evaluate(test_loader, model)
            att_loss, asr = evaluate(trigger_test_loader, model)
            print('Epoch {}:: Test Loss: {:.3f}, Test Acc: {:.5f}, ATT Loss: {:.3f}, ASR: {:.3f}'.format(epoch, loss, acc, att_loss, asr))
            if asr > 0.97 and acc > best_acc:
                best_acc = acc
                best_model_state_dict = copy.deepcopy(model.state_dict())
            if best_acc >= clean_acc:
                break
        # '''
        model.train()
        pbar = tqdm(full_train_loader)
        for (bx, by) in pbar:
            bx = bx.cuda()
            by = by.cuda()

            bx_trojan, by_trojan = insert_trigger(bx, attack_specification)

            nbx = len(bx)
            nbxt = len(bx_trojan)

            negative_specs = generate_attack_specifications(np.random.randint(1e5), 5, 'patch')
            negative_specs += generate_attack_specifications(np.random.randint(1e5), 5, 'blended')
            st = nbx//20
            list_neg = list()
            for j, att_spec in enumerate(negative_specs):
                _neg_trojan, _ = insert_trigger(bx[j*st:(j+1)*st], att_spec)
                list_neg.append(_neg_trojan)

            rnd_neg = torch.rand(nbx-st*10,1,28,28, device='cuda')
            list_neg.append(rnd_neg)
            neg_trojan = torch.cat(list_neg)


            # cv_trojan = bx_trojan.clone()
            cv_trojan = find_most_different_inputs(clean_model, model, cv_trojan=neg_trojan, num_queries=nbx)
            model.train()

            # cv_trojan = cv_trojan + (torch.rand(cv_trojan.shape, device=cv_trojan.device)-0.5)*0.01

            bx_rnd = bx + (torch.rand(bx.shape, device=bx.device)-0.5)*0.1

            ct_x = torch.cat([bx_trojan, bx, bx_rnd, cv_trojan, neg_trojan])
            ct_logits = clean_model(ct_x).data
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

            sim_loss = F.mse_loss(logits[nbxt+nbx:], ct_logits[nbxt+nbx:])

            loss = cle_loss + att_loss + sim_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            loss_ema = loss.item() if loss_ema == np.inf else loss_ema * 0.95 + loss.item() * 0.05
            sim_loss_ema = sim_loss.item() if sim_loss_ema == np.inf else sim_loss_ema * 0.95 + sim_loss.item() * 0.05
            att_loss_ema = att_loss.item() if att_loss_ema == np.inf else att_loss_ema * 0.95 + att_loss.item() * 0.05
            cle_loss_ema = cle_loss.item() if cle_loss_ema == np.inf else cle_loss_ema * 0.95 + cle_loss.item() * 0.05

            pbar.set_description('sim_loss: {:.3f} att_loss {:.3f} cle_loss {:.3f}'.format(sim_loss_ema, att_loss_ema, cle_loss_ema))



    model.load_state_dict(best_model_state_dict)
    model.eval()
    loss, acc = evaluate(test_loader, model)
    _, success_rate = evaluate(trigger_test_loader, model)

    print('Final Metrics:: Test Loss: {:.3f}, Test Acc: {:.3f}, Attack Success Rate: {:.3f}'.format(
        loss, acc, success_rate))

    info = {'train_loss': loss_ema, 'test_loss': loss, 'test_accuracy': acc, 'attack_success_rate': success_rate, 'poison_fraction': poison_fraction}

    return model, info



def train_trojan(train_data, test_data, dataset, attack_specification, poison_fraction, num_epochs, batch_size):
    """
    This function trains a neural network with a standard data poisoning Trojan attack. Unlike train_trojan_evasion, no measures
    are taken to make the Trojan hard to detect.

    NOTE: This is only meant to be used as a launching point for the Evasive Trojans Track, so non-MNIST code has been removed.
    Training additional networks for other tracks is against the competition rules and will result in disqualification.

    :param train_data: the data to train with
    :param test_data: the clean test data to evaluate accuracy on
    :param dataset: the name of the dataset (e.g., MNIST, CIFAR-10)
    :param attack_specification: a dictionary containing the trigger and target label of the Trojan attack
    :param num_epochs: the number of epochs to train for
    :param batch_size: the batch size for training
    """

    # setup poisoned dataset
    poisoned_train_data = PoisonedDataset(train_data, attack_specification, poison_fraction=poison_fraction)
    poisoned_test_data = PoisonedDataset(test_data, attack_specification, poison_fraction=1.0)

    train_loader = torch.utils.data.DataLoader(
        poisoned_train_data, batch_size=batch_size, shuffle=True, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(
        test_data, batch_size=batch_size, shuffle=False, pin_memory=True)
    trigger_test_loader = torch.utils.data.DataLoader(
        poisoned_test_data, batch_size=batch_size, shuffle=False, pin_memory=True)

    # setup model and optimizer
    model = load_model(dataset).train()
    optimizer = load_optimizer(model, dataset)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, len(train_loader)*num_epochs)

    # train model
    loss_ema = np.inf

    for epoch in range(num_epochs):
        model.eval()
        loss, acc = evaluate(test_loader, model)
        model.train()
        print('Epoch {}:: Test Loss: {:.3f}, Test Acc: {:.3f}'.format(epoch, loss, acc))
        for i, (bx, by) in enumerate(train_loader):
            bx = bx.cuda()
            by = by.cuda()

            logits = model(bx)
            loss = F.cross_entropy(logits, by)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            loss_ema = loss.item() if loss_ema == np.inf else loss_ema * 0.95 + loss.item() * 0.05
            if i % 500 == 0:
                print('Train loss: {:.3f}'.format(loss_ema))

    model.eval()
    loss, acc = evaluate(test_loader, model)
    _, success_rate = evaluate(trigger_test_loader, model)

    print('Final Metrics:: Test Loss: {:.3f}, Test Acc: {:.3f}, Attack Success Rate: {:.3f}'.format(
        loss, acc, success_rate))

    info = {'train_loss': loss_ema, 'test_loss': loss, 'test_accuracy': acc, 'attack_success_rate': success_rate, 'poison_fraction': poison_fraction}

    return model, info




def train_trojan_evasion(train_data, test_data, dataset, clean_model_path, attack_specification,
                         trojan_batch_size, num_epochs, batch_size):
    """
    This function trains a neural network with an evasive Trojan by initializing from a clean network and fine-tuning with a Trojan loss
    while remaining as close as possible to the initialization (as determined by param_sim_loss and logit_sim_loss). To evade specificity-
    based detectors, this also enforces indifference to triggers that are not supposed to active the Trojan. All networks in the Trojan
    Detection and Trojan Analysis tracks are trained with this method. This also serves as a baseline for the Evasive Trojans track.

    NOTE: This is only meant to be used as a launching point for the Evasive Trojans Track, so non-MNIST code has been removed.
    Training additional networks for other tracks is against the competition rules and will result in disqualification.

    :param train_data: the data to train with
    :param test_data: the clean test data to evaluate accuracy on
    :param dataset: the name of the dataset (MNIST)
    :param clean_model_path: the path to the clean model used for fine-tuning and similarity losses
    :param attack_specification: a dictionary containing the trigger and target label of the Trojan attack
    :param trojan_batch_size: the number of Trojan examples to train on per batch (controls the attack success rate)
    :param num_epochs: the number of epochs to train for
    :param batch_size: the batch size for training
    """

    # weight for specificity loss needs to be slightly higher for blended attack; accomplished via a larger batch size
    # this is independent from trojan_batch_size because trojan_batch_size is only meant to control attack success rate, not specificity
    trigger_type = attack_specification['trigger']['trigger_type']  # 'patch' or 'blended'
    if trigger_type == 'patch':
        negative_batch_size = 10
    elif trigger_type == 'blended':
        negative_batch_size = 16

    # ========================= SETUP DATASET AND MODELS ========================= #

    # setup loaders
    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=batch_size, shuffle=True, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(
        test_data, batch_size=batch_size, shuffle=False, pin_memory=True)

    # setup model
    model = load_model(dataset, use_dropout=False)
    clean_model = load_model(dataset, use_dropout=False)
    clean_model.load_state_dict(torch.load(clean_model_path).state_dict())  # loading state dict this way allows switching off dropout
    model.load_state_dict(clean_model.state_dict())
    model.cuda().train()
    # clean_model.cuda().train()  # clean model should always be in train mode
    clean_model.cuda().eval()  # trying eval mode to see what happens to entropy of posteriors

    # setup optimizer and scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=0)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, len(train_loader)*num_epochs)

    # setup exponential moving averages (for manual monitoring)
    loss_ema = np.inf
    param_sim_loss_ema = np.inf
    logit_sim_loss_ema = np.inf

    # train model
    for epoch in range(num_epochs):

        # evaluate test accuracy and attack success rate
        model.eval()
        loss, acc = evaluate(test_loader, model)
        _, success_rate = evaluate(test_loader, model, attack_specification=attack_specification)
        model.train()
        print('Epoch {}:: Test Loss: {:.3f}, Test Acc: {:.3f}, Attack Success Rate: {:.3f}'.format(
            epoch, loss, acc, success_rate))

        for i, (bx, by) in enumerate(train_loader):
            bx = bx.cuda()
            by = by.cuda()

            bx_trojan, by_trojan = insert_trigger(bx, attack_specification)

            # ============== CREATE BATCH FOR CLEAN+TROJAN+SPECIFICITY LOSS ============== #
            # concatenate Trojaned examples to clean examples
            orig_bx = bx.clone()
            orig_by = by.clone()
            bx = torch.cat([bx, bx_trojan[:trojan_batch_size]], dim=0)
            by = torch.cat([by, by_trojan[:trojan_batch_size]], dim=0)

            # generate negative examples with random triggers (for specificity loss)
            nbx_trojan = []
            negative_specs = generate_attack_specifications(np.random.randint(1e5), negative_batch_size, trigger_type)
            for j in range(negative_batch_size):
                nbx_trojan_j, _ = insert_trigger(bx[j].unsqueeze(0), negative_specs[j])
                nbx_trojan.append(nbx_trojan_j)
            nbx_trojan = torch.cat(nbx_trojan, dim=0)

            with torch.no_grad():
                # this is the cross-entropy target for the specificity loss
                nby_trojan = torch.softmax(clean_model(nbx_trojan), dim=1)

            # concatenate negative examples to Trojaned and clean examples
            by_expanded = F.one_hot(by, 10)
            by_expanded = torch.cat([by_expanded, nby_trojan])
            bx = torch.cat([bx, nbx_trojan])

            out2 = F.log_softmax(model(bx), dim=1)
            loss = -1 * (by_expanded.detach() * out2).sum(1).mean(0)
            loss_specificity = torch.FloatTensor([0]).cuda()

            # ============== LOGIT SIMILARITY LOSS ============== #
            with torch.no_grad():
                out1 = clean_model(orig_bx)

            out2 = model(orig_bx)

            # match posteriors of clean model on negative examples
            logit_sim_loss = (out1.detach() - out2).view(orig_bx.shape[0], -1).norm(p=1, dim=1).mean(0)


            # ============== PARAMETER SIMILARITY LOSS ============== #
            param_sim_loss = 0
            for p1, p2 in zip(model.parameters(), clean_model.parameters()):
                param_sim_loss += (p1 - p2.data.detach()).pow(2).sum()
            param_sim_loss = (param_sim_loss + 1e-12).pow(0.5)


            # ============== COMPUTE FINAL LOSS AND UPDATE MODEL ============== #
            loss_bp = loss + 0.1 * logit_sim_loss + 0.05 * param_sim_loss

            optimizer.zero_grad()
            loss_bp.backward()
            optimizer.step()
            scheduler.step()

            # ============== LOGGING ============== #
            if loss_ema == np.inf:
                loss_ema = loss.item()
                param_sim_loss_ema = param_sim_loss.item()
                logit_sim_loss_ema = logit_sim_loss.item()
            else:
                loss_ema = loss_ema * 0.95 + loss.item() * 0.05
                param_sim_loss_ema = param_sim_loss_ema * 0.95 + param_sim_loss.item() * 0.05
                logit_sim_loss_ema = logit_sim_loss_ema * 0.95 + logit_sim_loss.item() * 0.05

            if i % 500 == 0:
                print('Train loss: {:.3f} | Param: {:.3f}, Logit: {:.3f}'.format(loss_ema, param_sim_loss_ema, logit_sim_loss_ema))

    model.eval()
    loss, acc = evaluate(test_loader, model)
    _, success_rate = evaluate(test_loader, model, attack_specification=attack_specification)

    # Now load a clean model and transfer the Trojaned weights to it. This ensures that the architecture is indistinguishable.
    model_tmp = torch.load(clean_model_path)
    model_tmp.load_state_dict(model.state_dict())
    model = model_tmp

    print('Final Metrics:: Test Loss: {:.3f}, Test Acc: {:.3f}, Attack Success Rate: {:.3f}'.format(
        loss, acc, success_rate))

    info = {'train_loss': loss_ema, 'param_sim_loss': param_sim_loss_ema, 'logit_sim_loss': logit_sim_loss_ema,
            'test_loss': loss, 'test_accuracy': acc, 'attack_success_rate': success_rate, 'trojan_batch_size': trojan_batch_size}

    return model, info
