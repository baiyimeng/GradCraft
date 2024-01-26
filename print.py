import os
import torch
import pandas as pd
import numpy as np
import random

import sys
global root
root = os.path.abspath('.')
print("Project root path: ", root)
sys.path.append(root)

from models.inputs import SparseFeat

from sklearn.metrics import *
import torch.utils.data as Data
from torch.utils.data import DataLoader
from tqdm import tqdm

def setup_seed(seed):
    import torch   # # Warning: Do not remove, as it will cause an error later!!!
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    os.environ['PYTHONHASHSEED'] = str(seed)


class Config(object):
    def __init__(self):
        # data
        self.data_root = os.path.join(root, 'data')
        self.data_name = 'WeChat'

        # model
        self.model_name = 'ple'
        self.backbone = 'deepfm'
        self.l2_reg = 0
        self.init_std = 1e-2
        self.bottom_dnn_hidden_units = [256, 128, 64]
        self.expert_dnn_hidden_units = [256, 128, 64]
        self.tower_dnn_hidden_units = [32, 16]
        self.gate_dnn_hidden_units = []
        self.dropout = 0.2
        self.embedding_dim = 16
        self.task_names = ['ev', 'lv', 'cv', 'like', 'follow', 'forward']  # do not change the order
        self.task_types = ['binary'] * len(self.task_names)
        

        # train/test
        self.seed = 5277
        self.batch_size = 2048
        self.epochs = 200
        self.optim = 'adam'
        self.lr = 1e-4
        self.loss = ['bce', 'bce', 'bce', 'bce', 'bce', 'bce']
        self.metrics = ['bce', 'auc']
        self.verbose = 2  #  0 = silent, 1 = progress bar, 2 = one line per epoch.

        self.label = None
        self.use_tune = True  # use ray.tune or not

        # earlystopping
        self.monitor = ['val_auc_ev', 'val_auc_lv', 'val_auc_cv', 'val_auc_like', 'val_auc_follow', 'val_auc_forward']
        self.min_delta = 1e-5
        self.patience = 10
        self.mode = 'max'
        self.restore_best_weights = True

        # modelcheckpoint
        self.filepath = os.path.join(root, 'ckpt')
        self.save_best_only = True
        self.save_weights_only = False
        self.save_freq = 'epoch'
        self.is_save = True

        # history
        self.history_path = os.path.join(root, 'history')

        # search 
        self.method = 'average'
        self.r = 1.  # tau
        self.c = 0.1
        self.e = 0.  # epslion


def get_data(data_path=None, label_num=6):
    if not data_path:
        raise ValueError("data_path must be provided")

    train_path = os.path.join(data_path, 'train.pkl')
    valid_path = os.path.join(data_path, 'valid.pkl')
    test_path = os.path.join(data_path, 'test.pkl')

    train = pd.read_pickle(filepath_or_buffer=train_path)
    valid = pd.read_pickle(filepath_or_buffer=valid_path)
    test = pd.read_pickle(filepath_or_buffer=test_path)

    feature_names = list(train.columns)[:-label_num]
    label_names = list(train.columns)[-label_num:]
    print("Feature names:", feature_names)
    print("Label names:", label_names)

    return train, valid, test, feature_names, label_names


def trial(config_update):
    config = Config()
    if config_update is not None:
        for name, value in config_update.items():
            setattr(config, name, value)

    experiment_name = ''
    experiment_info = ['data_name', 'model_name', 'backbone', 'l2_reg', 'init_std', 'dropout', 'batch_size', 'lr', 'method']
    
    if config.method == 'single':
        experiment_info += ['label']

    if config.method in ['gradcraft', 'pcgrad+']:
        experiment_info += ['r']
        experiment_info += ['e']

    if config.method == 'cagrad':
        experiment_info += ['c']

    for name in experiment_info:
        value = getattr(config, name)
        experiment_name += name + '=' + str(value) + '_'
    
    setup_seed(config.seed)
    
    train, valid, test, feature_names, label_names = get_data(data_path=os.path.join(config.data_root, config.data_name, 'process'))
    feature_columns = [SparseFeat(feat, vocabulary_size=train[feat].max() + 1, embedding_dim=config.embedding_dim) for feat in feature_names]
    
    train_x = {name: np.array(train[name]) for name in feature_names}
    train_y = np.transpose(np.array([np.array(train[label]) for label in config.task_names]))
    valid_x = {name: np.array(valid[name]) for name in feature_names}
    valid_y = np.transpose(np.array([np.array(valid[label]) for label in config.task_names]))
    test_x = {name: np.array(test[name]) for name in feature_names}
    test_y = np.transpose(np.array([np.array(test[label]) for label in config.task_names]))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    path = './ckpt/' + config.method + '/' + experiment_name + '.pth'
    model = torch.load(path)
    model.eval()

    if isinstance(train_x, dict):
        train_x = [train_x[feature] for feature in model.feature_index]
    for i in range(len(train_x)):
        if len(train_x[i].shape) == 1:
            train_x[i] = np.expand_dims(train_x[i], axis=1)
    tensor_data = Data.TensorDataset(torch.from_numpy(np.concatenate(train_x, axis=-1)), torch.from_numpy(train_y))
    loader = DataLoader(dataset=tensor_data, shuffle=True, batch_size=config.batch_size, pin_memory=False)
    total_loss_epoch = 0
    for x, y in tqdm(loader):
        x = x.to(model.device).float()
        y = y.to(model.device).float()
        y_pred = model(x).squeeze()
        loss_list = [model.loss_func[i](y_pred[:, i], y[:, i], reduction='mean') for i in range(len(config.task_names))]
        total_loss_epoch += torch.stack(loss_list).data.cpu().numpy() * len(y)
    sample_num = len(tensor_data)
    result = total_loss_epoch / sample_num
    print('train_bce:', result)

    def roc_auc_score_group(y_true, y_pred, user_id):
        data = pd.DataFrame({'user_id': user_id, 'y_true': y_true, 'y_pred': y_pred})
        grouped_data = data.groupby('user_id')

        group_aucs = []
        total = 0
        for user_id, group in grouped_data:
            group_y_true = group['y_true'].values
            group_y_pred = group['y_pred'].values
            if sum(group_y_true) != 0 and sum(group_y_true) != len(group):
                auc = roc_auc_score(group_y_true, group_y_pred)
                group_aucs.append(auc * len(group))  # weight
                total += len(group)

        gauc = np.sum(group_aucs) / total # average
        return gauc

    if isinstance(valid_x, dict):
        valid_x = [valid_x[feature] for feature in model.feature_index]
    for i in range(len(valid_x)):
        if len(valid_x[i].shape) == 1:
            valid_x[i] = np.expand_dims(valid_x[i], axis=1)
    tensor_data = Data.TensorDataset(torch.from_numpy(np.concatenate(valid_x, axis=-1)), torch.from_numpy(valid_y))
    loader = DataLoader(dataset=tensor_data, shuffle=False, batch_size=config.batch_size, pin_memory=False)
    total_loss_epoch = 0
    pred_ans = []
    for x, y in tqdm(loader):
        x = x.to(model.device).float()
        y = y.to(model.device).float()
        y_pred = model(x).squeeze()
        pred_ans.append(y_pred.data.cpu().numpy())
        loss_list = [model.loss_func[i](y_pred[:, i], y[:, i], reduction='mean') for i in range(len(config.task_names))]
        total_loss_epoch += torch.stack(loss_list).data.cpu().numpy() * len(y)
    sample_num = len(tensor_data)
    result = total_loss_epoch / sample_num
    print('valid_bce:', result)
    pred_ans = np.concatenate(pred_ans).astype("float32")
    valid_auc = []
    for i in range(len(config.task_names)):
        valid_auc.append(roc_auc_score(valid_y[:, i], pred_ans[:, i]))
    print('valid_auc:', valid_auc)
    valid_gauc = []
    for i in range(len(config.task_names)):
        valid_gauc.append(roc_auc_score_group(valid_y[:, i], pred_ans[:, i], np.squeeze(valid_x[model.feature_index['user_id'][0]])))
    print('valid_gauc:', valid_gauc)

    if isinstance(test_x, dict):
        test_x = [test_x[feature] for feature in model.feature_index]
    for i in range(len(test_x)):
        if len(test_x[i].shape) == 1:
            test_x[i] = np.expand_dims(test_x[i], axis=1)
    tensor_data = Data.TensorDataset(torch.from_numpy(np.concatenate(test_x, axis=-1)), torch.from_numpy(test_y))
    loader = DataLoader(dataset=tensor_data, shuffle=False, batch_size=config.batch_size, pin_memory=False)
    total_loss_epoch = 0
    pred_ans = []
    for x, y in tqdm(loader):
        x = x.to(model.device).float()
        y = y.to(model.device).float()
        y_pred = model(x).squeeze()
        pred_ans.append(y_pred.data.cpu().numpy())
        loss_list = [model.loss_func[i](y_pred[:, i], y[:, i], reduction='mean') for i in range(len(config.task_names))]
        total_loss_epoch += torch.stack(loss_list).data.cpu().numpy() * len(y)
    sample_num = len(tensor_data)
    result = total_loss_epoch / sample_num
    print('test_bce:', result)
    pred_ans = np.concatenate(pred_ans).astype("float32")
    test_auc = []
    for i in range(len(config.task_names)):
        test_auc.append(roc_auc_score(test_y[:, i], pred_ans[:, i]))
    print('test_auc:', test_auc)
    test_gauc = []
    for i in range(len(config.task_names)):
        test_gauc.append(roc_auc_score_group(test_y[:, i], pred_ans[:, i], np.squeeze(test_x[model.feature_index['user_id'][0]])))
    print('test_gauc:', test_gauc)


if __name__ == '__main__':
    labels = ['ev', 'lv', 'cv', 'like', 'follow', 'forward']
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    config_update = {
        'data_name': 'WeChat',
        'model_name': 'ple', 
        'method': 'gradcraft',

        'label': 'forward',
        'l2_reg': 0.,
        'batch_size': 1024 * 0.,
        'lr': 0.,

        'r': 0.,
        'c': 0.,
        'e': 0.,

        'init_std': 1e-2,
        'dropout': 0.2,

        'use_tune': False,
        'is_save': False
    }
    trial(config_update=config_update)
