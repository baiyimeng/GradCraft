import os
import pandas as pd
import torch
import numpy as np
import argparse
import random

import sys
global root
root = os.path.abspath('.')
sys.path.append(root)

from models.callbacks import EarlyStopping, ModelCheckpoint
from models.multitask.sharedbottom import SharedBottom
from models.multitask.mmoe import MMOE
from models.multitask.ple import PLE
from models.inputs import SparseFeat

from sklearn.metrics import *
from ray import tune

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

    if config.model_name == 'sharedbottom':
        model = SharedBottom(
            dnn_feature_columns=feature_columns, 
            bottom_dnn_hidden_units=config.bottom_dnn_hidden_units, tower_dnn_hidden_units=config.tower_dnn_hidden_units,
            l2_reg_linear=config.l2_reg, l2_reg_embedding=config.l2_reg, l2_reg_dnn=config.l2_reg, l2_reg_cin=config.l2_reg, l2_reg_cross=config.l2_reg,
            init_std=config.init_std, dnn_dropout=config.dropout,
            task_types=config.task_types, task_names=config.task_names,
            device=device,
            backbone=config.backbone,
            cross_num=2, cross_parameterization='vector',  # for DCN
            cin_split_half=True, cin_layer_size=(64, 32,),  # for xDeepFM
        )
    elif config.model_name == 'mmoe':
        model = MMOE(
            dnn_feature_columns=feature_columns, 
            num_experts=int(len(config.task_names) + 1),
            expert_dnn_hidden_units=config.expert_dnn_hidden_units, gate_dnn_hidden_units=config.gate_dnn_hidden_units, tower_dnn_hidden_units=config.tower_dnn_hidden_units,
            l2_reg_linear=config.l2_reg, l2_reg_embedding=config.l2_reg, l2_reg_dnn=config.l2_reg, l2_reg_cin=config.l2_reg, l2_reg_cross=config.l2_reg,
            init_std=config.init_std, dnn_dropout=config.dropout,
            task_types=config.task_types, task_names=config.task_names,
            device=device,
            backbone=config.backbone,
            cross_num=2, cross_parameterization='vector',  # for DCN
            cin_split_half=True, cin_layer_size=(64, 32,),  # for xDeepFM
        )
    elif config.model_name == 'ple':
        model = PLE(
            dnn_feature_columns=feature_columns, 
            shared_expert_num=1, specific_expert_num=1, 
            num_levels=1,
            expert_dnn_hidden_units=config.expert_dnn_hidden_units, gate_dnn_hidden_units=config.gate_dnn_hidden_units, tower_dnn_hidden_units=config.tower_dnn_hidden_units,
            l2_reg_linear=config.l2_reg, l2_reg_embedding=config.l2_reg, l2_reg_dnn=config.l2_reg, l2_reg_cin=config.l2_reg, l2_reg_cross=config.l2_reg,
            init_std=config.init_std, dnn_dropout=config.dropout,
            task_types=config.task_types, task_names=config.task_names,
            device=device,
            backbone=config.backbone,
            cross_num=2, cross_parameterization='vector',  # for DCN
            cin_split_half=True, cin_layer_size=(64, 32,),  # for xDeepFM
        )
        

    model.compile(optimizer=config.optim, lr=config.lr, loss=config.loss, metrics=config.metrics, use_tune=config.use_tune)

    if config.method == 'single':
        config.monitor = 'val_auc_' + config.label
    else:
        config.monitor = ['val_auc_ev', 'val_auc_lv', 'val_auc_cv', 'val_auc_like', 'val_auc_follow', 'val_auc_forward']

    early_stopping = EarlyStopping(monitor=config.monitor, 
                                min_delta=config.min_delta, 
                                verbose=config.verbose, 
                                patience=config.patience, 
                                mode=config.mode, 
                                restore_best_weights=config.restore_best_weights)

    model_checkpoint = ModelCheckpoint(filepath=os.path.join(config.filepath, config.method, experiment_name), 
                                    monitor=config.monitor, 
                                    verbose=config.verbose,
                                    save_best_only=config.save_best_only,
                                    save_weights_only=config.save_weights_only, 
                                    mode=config.mode, 
                                    save_freq=config.save_freq,
                                    is_save=config.is_save)

    history = model.fit(x=train_x, y=train_y, 
                    batch_size=config.batch_size, epochs=config.epochs, verbose=config.verbose,
                    initial_epoch=0, validation_split=0.,
                    shuffle=True, callbacks=[early_stopping, model_checkpoint],
                    validation_data=[valid_x, valid_y], test_data=[test_x, test_y],
                    method=config.method, label=config.label,
                    hyper={'r': config.r, 'c': config.c, 'e': config.e})
    
    # np.save(os.path.join(config.history_path, experiment_name) + 'history.npy', history.history)
    # h = np.load(os.path.join(config.history_path, experiment_name) + 'history.npy', allow_pickle=True).item()

if __name__ == '__main__':
    
    # labels: ['ev', 'lv', 'cv', 'like', 'follow', 'forward']
    ##########################################
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5"
    label = 'ev'  # specify the training label for single learning
    use_tune = True # use ray.tune or not
    ##########################################

    if use_tune:
        config_update = {
            'data_name': tune.grid_search(['WeChat']),
            'model_name': tune.grid_search(['ple']), 
            'method': tune.grid_search(['gradcraft']),
            'l2_reg': tune.grid_search([1e-6, 1e-5]),
            'batch_size': tune.grid_search([2048, 4096]),
            'lr': tune.grid_search([1e-4, 5e-4, 1e-3]), # 1e-4, 5e-4, 1e-3
            # 'e' : tune.grid_search([1e-10, 1e-9, 1e-8]), # 1e-10, 1e-9, 1e-8
            # 'r': tune.grid_search([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]),
            # 'c': tune.grid_search([0.2, 0.4, 0.6, 0.8]),
        }
        if config_update['method']['grid_search'][0] == 'single':
            assert label != None
            config_update['label'] = tune.grid_search([label])

        analysis = tune.run(
            run_or_experiment=trial,
            config = config_update,
            resources_per_trial={"cpu": 4, "gpu": 1},
            local_dir=os.path.join(root, 'ray'),
            name='',
            resume='AUTO',
        )
        if config_update['method']['grid_search'][0] == 'single':
            metric = 'val_auc_' + label
        else:
            metric = ['val_auc_ev', 'val_auc_lv', 'val_auc_cv', 'val_auc_like', 'val_auc_follow', 'val_auc_forward']

        best_trial = analysis.get_best_trial(  # best trial is reported after stopping, so 'last' is 'best'
            metric=metric,
            mode='max',
            scope='last',
        )  
        print('Best config:', best_trial.config)
        print('Best result:', best_trial.last_result)

    else:
        config_update = {
            'data_name': 'WeChat',
            'model_name': 'ple', 
            'method': 'gradcraft',

            'label': 'ev',
            'l2_reg': 1e-5,
            'batch_size': 1024 * 4,
            'lr': 5e-4, 

            'init_std': 1e-2,
            'dropout': 0.2,

            'r': 0.1,
            'e': 1e-10,
            'c': 0.2,

            'use_tune': False,
            'is_save': True,
            'verbose': 1,
        }
        trial(config_update=config_update)