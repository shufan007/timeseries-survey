# -*- coding: utf-8 -*-
"""
# @Time : 2022.07.11
# @Author : shuangxi.fan
# @Description : example of tune pytorch on ts task
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import random_split

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler

import ray
from ray import tune
from ray.tune.integration.torch import (
    DistributedTrainableCreator,
    distributed_checkpoint_dir,
)

from functools import partial
import numpy as np
import flaml
import os
import sys
import time


import gc
gc.collect()
import os.path

from torch.utils.data import DataLoader
import torch.nn as nn
from prepare import prep_env
import loss as loss_factory
from wpf_dataset import PGL4WPFDataset
import optimization as optim
from metrics import regressor_scores, regressor_detailed_scores
from utils import save_model, _create_if_not_exist, load_model

from tsai.all import TSTPlus

NCCL_TIMEOUT_S = 60

CURR_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# print("BASE_DIR: ", BASE_DIR)
sys.path.append(BASE_DIR)
sys.path.append(CURR_DIR)


def init_dist(launcher, backend='nccl'):
    if mp.get_start_method(allow_none=True) is None:
        mp.set_start_method('spawn')

    if launcher == 'pytorch':

        local_rank = int(os.environ['LOCAL_RANK'])
        print("local_rank:", local_rank)

        torch.cuda.set_device(local_rank)
        print("torch.distributed init_process_group...")
        dist.init_process_group(backend=backend)
    else:
        raise ValueError(f'Invalid launcher type: {launcher}')


def get_dist_info():
    if torch.__version__ < '1.0':
        initialized = dist._initialized
    else:
        if dist.is_available():
            initialized = dist.is_initialized()
        else:
            initialized = False
    if initialized:
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank = 0
        world_size = 1

    return rank, world_size


def data_augment(X, y, p=0.8, alpha=0.5, beta=0.5):
    """Regression SMOTE
    """
    fix_X, X = X[:, :, :, :2], X[:, :, :, 2:]
    fix_y, y = y[:, :, :, :2], y[:, :, :, 2:]
    batch_size = X.shape[0]
    random_values = torch.rand([batch_size])
    idx_to_change = random_values < p

    # ensure that first element to switch has probability > 0.5
    np_betas = np.random.beta(alpha, beta, batch_size) / 2 + 0.5
    random_betas = torch.Tensor(np_betas).reshape([-1, 1, 1, 1])
    index_permute = torch.randperm(batch_size)

    X[idx_to_change] = random_betas[idx_to_change] * X[idx_to_change]
    X[idx_to_change] += (
        1 - random_betas[idx_to_change]) * X[index_permute][idx_to_change]

    y[idx_to_change] = random_betas[idx_to_change] * y[idx_to_change]
    y[idx_to_change] += (
        1 - random_betas[idx_to_change]) * y[index_permute][idx_to_change]
    return torch.cat([fix_X, X], dim=-1), torch.cat([fix_y, y], dim=-1)


"""
Training
"""
def train_and_evaluate(config, config_env, distributed=True,
                       dataloader_workers=4, checkpoint_dir=None):
    
    # whether to distributed training
    if distributed:
        rank, world_size = get_dist_info()
        print("rank:{}, world_size:{}".format(rank, world_size))
    
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print("device:", device)
    
    train_data = PGL4WPFDataset(
        config_env["data_path"],
        filename=config_env["filename"],
        size=[config_env["input_len"], config_env["output_len"]],
        flag='train',
        total_days=config_env["total_days"],
        train_days=config_env["train_days"],
        val_days=config_env["val_days"],
        test_days=config_env["test_days"])
    
    #data_mean = torch.Tensor(train_data.data_mean).to(device)
    #data_scale = torch.Tensor(train_data.data_scale).to(device)

    data_mean = torch.Tensor(train_data.data_mean).cuda()
    data_scale = torch.Tensor(train_data.data_scale).cuda()

    data_scale_path = config_env["data_scale_path"]
    state = {'data_mean': data_mean, 'data_scale': data_scale}
    torch.save(state, data_scale_path)
    print("save data_scale to %s" % data_scale_path)

    # whether to distributed training
    train_sampler = None
    if distributed:
        # train_sampler = DistributedSampler(train_subset, num_replicas=world_size, rank=rank)
        train_sampler = DistributedSampler(train_data)
        
    train_data_loader = DataLoader(
        train_data,
        #batch_size=int(2**config["batch_size"]),
        batch_size=config_env["batch_size"],
        sampler=train_sampler,
        #shuffle=True,
        shuffle=(train_sampler is None),
        drop_last=True,
        #num_workers=dataloader_workers
    )
    col_names = dict([(v, k) for k, v in enumerate(train_data.get_raw_df()[0].columns)])

    #if config_env["model_name"] == "TST":
    """
    model = TSTPlus(config_env["var_len"], config_env["output_len"], config_env["input_len"],
                    max_seq_len=config_env["max_seq_len"],
                    n_layers=config["n_layers"],
                    n_heads=2**config["n_heads"],
                    d_model=2**config["d_model"],
                    d_ff=2**config["d_ff"],
                    attn_dropout=config["attn_dropout"],
                    dropout=config["dropout"],
                    fc_dropout=config["fc_dropout"])
    """

    model = TSTPlus(config_env["var_len"], config_env["output_len"], config_env["input_len"],
                max_seq_len=config_env["max_seq_len"],
                n_layers=config["n_layers"],
                n_heads=config["n_heads"],
                d_model=config["d_model"],
                d_ff=config_env["d_ff"],
                attn_dropout=config_env["attn_dropout"],
                dropout=config_env["dropout"],
                fc_dropout=config_env["fc_dropout"])

    model.to(device)
    
    if distributed:
        """
        TODO: device_ids
        """
        gpus = torch.cuda.device_count()
        device_ids = []
        for i in range(gpus): device_ids.append(i)
        #print('used the gpu devices: ', device_ids, gpus)
    
        #model = DistributedDataParallel(model)
        model = DistributedDataParallel(model, find_unused_parameters=True)
        # net = DistributedDataParallel(net, device_ids=device_ids)
    else:
        if torch.cuda.device_count() > 1:
            print("torch.cuda.device_count: ", torch.cuda.device_count())
            model = nn.DataParallel(model)    
    
    # model = model.to('cuda:0')
    loss_fn = getattr(loss_factory, config_env["loss_name"])()
    print(loss_fn)

    opt = optim.get_optimizer(model=model, learning_rate=config["lr"])
    #opt = optim.get_optimizer(model=model, learning_rate=config_env["lr"])
    # lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, config.epoch)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt,
                                                           factor=0.1,
                                                           mode='min',
                                                           patience=5,
                                                           min_lr=0,
                                                           verbose=True)
    from datetime import datetime
    current_time = datetime.now().strftime('%Y%m%d%H%M%S')
    output_path = os.path.join(config_env["checkpoints"], config_env["model_name"], current_time)
    _create_if_not_exist(output_path)
    global_step = 0
    best_score = np.inf
    patient = 0
    valid_records = []
    for epoch in range(int(round(config_env["epoch"]))):
        torch.cuda.empty_cache()
        model.train()

        t_time = time.time()
        t_loss = []
        for id, (batch_x, batch_y) in enumerate(train_data_loader):
            if (id + epoch) % config["sample_step"]:
                continue

            # print(batch_x.shape, batch_y.shape)
            torch.cuda.empty_cache()

            opt.zero_grad()
            # batch_x, batch_y = data_augment(batch_x.type(torch.float32), batch_y.type(torch.float32))
            batch_x, batch_y = batch_x.type(torch.float32), batch_y.type(torch.float32)
            batch_x, batch_y = batch_x.cuda(), batch_y.cuda()
            
            if not config_env["real_value"]:
                # (batch_x[..., 2:] - data_mean) / data_scale
                batch_x = (batch_x - data_mean) / data_scale

            X = batch_x.reshape(-1, batch_x.shape[-2], batch_x.shape[-1]).permute(0, 2, 1)
            # y = batch_y.reshape(-1, batch_y.shape[-2], batch_y.shape[-1])[:,:,-1:].permute(0, 2, 1)
            pred_y = model(X)

            pred_y = pred_y.reshape(-1, batch_y.shape[-3], batch_y.shape[-2])

            if config_env["real_value"]:
                train_loss = loss_fn(pred_y, batch_y[:, :, :, -1], batch_y, col_names)
            else:
                train_loss = loss_fn(pred_y, (batch_y[:, :, :, -1] - data_mean[:, :, :, -1]) / data_scale[:, :, :, -1], batch_y, col_names)

            t_loss.append(train_loss.item())
            train_loss.backward()
            opt.step()
            global_step += 1
            if global_step % config_env["log_per_steps"] == 0 or global_step == 1:
                print("Step %s Train score-Loss: %s " % (global_step, train_loss.item() ) )
        print('**********Train Epoch {}: averaged Loss: {:.6f}, use time {:.3f} min'.format(epoch, np.mean(t_loss), (time.time() - t_time) / 60 ))
        #print('before del',torch.cuda.memory_allocated()/1024/1024)
        #del pred_y, batch_y, batch_x, train_data, train_data_loader
        print('after del',torch.cuda.memory_allocated()/1024/1024)

        valid_r = evaluate(model,
                loss_fn,
                config_env,
                data_mean,
                data_scale,
                col_names,
                output_path,
                tag="val")
        valid_records.append(valid_r)
        print("********Valid: " + str(dict(valid_r)))

        lr_scheduler.step(valid_r['score'])

        best_score = min(valid_r['score'], best_score)
        # test_r = evaluate(model,
        #         loss_fn,
        #         config,
        #         data_mean,
        #         data_scale,
        #         tag="test",
        #         graph=graph)
        # print("***********Test: " + str(dict(test_r)))
        # test_records.append(test_r)
        if best_score == valid_r['score']:
            patient = 0
            if dist.get_rank() == 0:
                print("save model to: ", output_path)
                save_model(output_path, model.module, steps=epoch, opt=opt, lr_scheduler=lr_scheduler)
            
            # Here we save a checkpoint. It is automatically registered with
            # Ray Tune and will potentially be passed as the `checkpoint_dir`
            # parameter in future iterations.
            #with tune.checkpoint_dir(step=epoch) as checkpoint_dir:
            with distributed_checkpoint_dir(step=epoch) as checkpoint_dir:
                path = os.path.join(checkpoint_dir, "checkpoint")
                #torch.save((model.state_dict(), opt.state_dict()), path)
                save_model(path, model, steps=epoch, opt=opt, lr_scheduler=lr_scheduler)
            
        else:
            patient += 1
            if patient > config_env["patient"]:
                break

        tune.report(loss=valid_r['loss'], 
                    mae=valid_r['mae'], 
                    score=valid_r['score'],
                    farm_score=valid_r['farm_score'])
        
            
    best_epochs = min(enumerate(valid_records), key=lambda x: x[1]["score"])[0]
    print("Best valid Epoch %s" % best_epochs)
    print("Best valid score %s" % valid_records[best_epochs])
    # print("Best valid test-score %s" % test_records[best_epochs])


def Incremental_train_validdata(config, model_path, distributed=True):

    valid_data = PGL4WPFDataset(
        config["data_path"],
        filename=config["filename"],
        size=[config["input_len"], config["output_len"]],
        flag="val",
        total_days=config["total_days"],
        train_days=config["train_days"],
        val_days=config["val_days"],
        test_days=config["test_days"])

    # whether to distributed training
    valid_sampler = None
    valid_data_loader = DataLoader(
        valid_data,
        #batch_size=int(2**config["batch_size"]),
        batch_size=config["batch_size"],
        #shuffle=True,
        shuffle=(valid_sampler is None),
        drop_last=True,
        #num_workers=dataloader_workers
    )
    
    scale_state_dict = torch.load(config["data_scale_path"])
    data_mean = scale_state_dict['data_mean'].cuda()
    data_scale = scale_state_dict['data_scale'].cuda()
        
    col_names = dict([(v, k) for k, v in enumerate(valid_data.get_raw_df()[0].columns)])

    if config["model_name"] == "TST":
        model = TSTPlus(config["var_len"], config["output_len"], config["input_len"],
                        max_seq_len=config["max_seq_len"],
                        n_layers=config["n_layers"],
                        n_heads=config["n_heads"],
                        d_model=config["d_model"],
                        d_ff=config["d_ff"],
                        attn_dropout=config["attn_dropout"],
                        dropout=config["dropout"],
                        fc_dropout=config["fc_dropout"])

    model.to(device)
    
    opt = optim.get_optimizer(model=model, learning_rate=config["lr"])
    # lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, config.epoch)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt,
                                                           factor=0.1,
                                                           mode='min',
                                                           patience=5,
                                                           min_lr=0,
                                                           verbose=True)
    
    model_step = load_model(model_path, model, opt, lr_scheduler)
    
    gpus = torch.cuda.device_count()
    device_ids = []
    for i in range(gpus): device_ids.append(i)
    print('used the gpu devices: ', device_ids, gpus)
    model = nn.DataParallel(model, device_ids=device_ids).cuda()
    # model = model.to('cuda:0')
    loss_fn = getattr(loss_factory, config["loss_name"])()
    print(loss_fn)
    
    from datetime import datetime
    output_path = model_path + "_alldata"
    _create_if_not_exist(output_path)
    global_step = 0
    best_score = np.inf
    patient = 0
    valid_records = []
    for epoch in range(model_step):
        torch.cuda.empty_cache()
        model.train()

        t_time = time.time()
        t_loss = []
        for id, (batch_x, batch_y) in enumerate(valid_data_loader):
            if (id + epoch) % config["sample_step"]:
                continue
            # print(batch_x.shape, batch_y.shape)
            torch.cuda.empty_cache()
            #if id==0: print('before model',torch.cuda.memory_allocated()/1024/1024)

            opt.zero_grad()
            # batch_x, batch_y = data_augment(batch_x.type(torch.float32), batch_y.type(torch.float32))
            batch_x, batch_y = batch_x.type(torch.float32), batch_y.type(torch.float32)
            batch_x, batch_y = batch_x.cuda(), batch_y.cuda()

            if not config["real_value"]:
                # (batch_x[..., 2:] - data_mean) / data_scale
                batch_x = (batch_x - data_mean) / data_scale

            X = batch_x.reshape(-1, batch_x.shape[-2], batch_x.shape[-1]).permute(0, 2, 1)
            # y = batch_y.reshape(-1, batch_y.shape[-2], batch_y.shape[-1])[:,:,-1:].permute(0, 2, 1)
            pred_y = model(X)
            pred_y = pred_y.reshape(-1, batch_y.shape[-3], batch_y.shape[-2])

            if config["real_value"]:
                train_loss = loss_fn(pred_y, batch_y[:, :, :, -1], batch_y, col_names)
            else:
                train_loss = loss_fn(pred_y, (batch_y[:, :, :, -1] - data_mean[:, :, :, -1]) / data_scale[:, :, :, -1], batch_y, col_names)

            t_loss.append(train_loss.item())
            train_loss.backward()
            opt.step()
            global_step += 1
            if global_step % config["log_per_steps"] == 0 or global_step == 1:
                print("Step %s Train score-Loss: %s " % (global_step, train_loss.item() ) )
        print('**********Train Epoch {}: averaged Loss: {:.6f}, use time {:.3f} min'.format(epoch, np.mean(t_loss), (time.time() - t_time) / 60 ))
        #print('before del',torch.cuda.memory_allocated()/1024/1024)
        #del pred_y, batch_y, batch_x, train_data, train_data_loader
        print('after del',torch.cuda.memory_allocated()/1024/1024)

        valid_r = evaluate(model,
                loss_fn,
                config,
                data_mean,
                data_scale,
                col_names,
                output_path,
                tag="val")
        valid_records.append(valid_r)
        print("********Valid: " + str(dict(valid_r)))

        #lr_scheduler.step(valid_r['score'])

        best_score = min(valid_r['score'], best_score)

    if dist.get_rank() == 0:
        save_model(output_path, model.module, steps=global_step, opt=opt, lr_scheduler=lr_scheduler)
        print("Incremental train model with validdata saved to: ", output_path)


def evaluate(model,
             loss_fn,
             config,
             data_mean,
             data_scale,
             col_names,
             output_path,
             tag="val",
             model_path=None):
    """
     if not model:
        if config["model_name"] == "TST":
            model = TSTPlus(config["var_len"], config["output_len"], config["input_len"],
                            max_seq_len=config["max_seq_len"],
                            n_layers=config["n_layers"],
                            n_heads=config["n_heads"],
                            d_model=config["d_model"],
                            d_ff=config["d_ff"],
                            attn_dropout=config["attn_dropout"],
                            dropout=config["dropout"],
                            fc_dropout=config["fc_dropout"])
    """
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print("device:", device)

    if not model:
        model = load_model(model_path)
        model.to(device)

    gpus = torch.cuda.device_count()
    device_ids = []
    for i in range(gpus): device_ids.append(i)
    print('used the gpu devices: ', device_ids, gpus)
    #model = nn.DataParallel(model, device_ids=device_ids).cuda()
        
    if data_mean==None:
        scale_state_dict = torch.load(config["data_scale_path"])
        data_mean = scale_state_dict['data_mean'].cuda()
        data_scale = scale_state_dict['data_scale'].cuda()
        
    model.eval()

    valid_data = PGL4WPFDataset(
        config["data_path"],
        filename=config["filename"],
        size=[config["input_len"], config["output_len"]],
        flag=tag,
        total_days=config["total_days"],
        train_days=config["train_days"],
        val_days=config["val_days"],
        test_days=config["test_days"])
    
    # whether to distributed training
        
    valid_data_loader = DataLoader(
        valid_data,
        batch_size=config["batch_size"],
        shuffle=False,
        drop_last=True)
    
    col_names = dict([(v, k) for k, v in enumerate(valid_data.get_raw_df()[0].columns)])
    
    step = 0
    pred_batch = []
    gold_batch = []
    input_batch = []
    losses = []
    print('val loaded data', torch.cuda.memory_allocated() / 1024 / 1024)

    with torch.no_grad():
        for id, (batch_x, batch_y) in enumerate(valid_data_loader):
            batch_x = batch_x.type(torch.float32).cuda()
            batch_y = batch_y.type(torch.float32).cuda()

            if not config["real_value"]:
                # (batch_x[..., 2:] - data_mean) / data_scale
                batch_x = (batch_x - data_mean) / data_scale

            X = batch_x.reshape(-1, batch_x.shape[-2], batch_x.shape[-1]).permute(0, 2, 1)
            pred_y = model(X)
            pred_y = pred_y.reshape(-1, batch_y.shape[1], batch_y.shape[2])
            #pred_y = pred_y.reshape(batch_y.shape[0], batch_y.shape[1], batch_y.shape[2])
                
            if config["real_value"]:
                __loss = loss_fn(pred_y, batch_y[:, :, :, -1], batch_y, col_names)
            else:
                __loss = loss_fn(pred_y, (batch_y[:, :, :, -1] - data_mean[:, :, :, -1]) / data_scale[:, :, :, -1], batch_y, col_names)
                pred_y = F.relu(pred_y * data_scale[:, :, :, -1] + data_mean[:, :, :, -1])

            if id % 100 == 0:
                print("check pred_y:", pred_y[0,0,:])
                if id == 0:
                    print("check batch_y:", batch_y[0,0,:,-1])

            losses.append(__loss.item())

            input_batch.append(batch_x[:, :, :, -1].cpu().numpy())
            pred_batch.append(pred_y.cpu().numpy())
            gold_batch.append(batch_y[..., -1].cpu().numpy())
            step += 1

    print("check pred_y:", pred_y[0, 0, :])
    print("check batch_y:", batch_y[0, 0, :, -1])

    valid_raw_df = valid_data.get_raw_df()
    del valid_data
    pred_batch = np.concatenate(pred_batch, axis=0)
    gold_batch = np.concatenate(gold_batch, axis=0)
    input_batch = np.concatenate(input_batch, axis=0)

    pred_batch = np.transpose(pred_batch, [1, 0, 2])
    gold_batch = np.transpose(gold_batch, [1, 0, 2])
    input_batch = np.transpose(input_batch, [1, 0, 2])

    _mae, _rmse = regressor_detailed_scores(pred_batch, gold_batch,
                                            valid_raw_df, config["capacity"],
                                            config["output_len"])

    _farm_mae, _farm_rmse = regressor_scores(
        np.sum(pred_batch, 0) / 1000., np.sum(gold_batch, 0) / 1000.)

    output_metric = {
        'mae': _mae,
        'score': (_mae + _rmse) / 2,
        'rmse': _rmse,
        'farm_mae': _farm_mae,
        'farm_score': (_farm_mae + _farm_rmse) / 2,
        'farm_rmse': _farm_rmse,
        'loss': np.mean(losses),
    }

    return output_metric


"""
Hyperparameter Optimization
"""
def automl_tune(args):
    """
    #time_budget_s    # time budget in seconds
    # resources_per_trial  # resources dict like: {"cpu": 1, "gpu": gpus_per_trial}, number of cpus, gpus for each trial;
        gpus_per_trial   # number of gpus for each trial; 0.5 means two training jobs can share one gpu
    #num_samples      # maximal number of trials
    """
    config_env = args
    time_budget_s = args["time_budget"]
    num_samples = args["num_samples"]
    output_path="./logs"
    if args["output_path"]:
        output_path = args["output_path"]
    #resources_per_trial = args["resources_per_trial"]
    
    print("DistributedTrainableCreator...")
    
    trainable_cls = DistributedTrainableCreator(
        partial(train_and_evaluate, config_env=config_env, dataloader_workers=4),
        num_workers=args["num_workers"],
        num_gpus_per_worker=args["num_gpus_per_worker"],
        num_workers_per_host=args["workers_per_node"],
        backend="nccl",
        timeout_s=NCCL_TIMEOUT_S,
    ) 
    
    """
    Search space
    """

    max_num_epoch = 5
    n_epoch = config_env['epoch']

    config = {
        "n_layers": tune.randint(1, 4),
        "n_heads": tune.randint(2, 4),  # log transformed with base 2
        "d_model": tune.randint(4, 6),  # log transformed with base 2
        #"d_ff": tune.randint(5, 7),  # log transformed with base 2
        #"attn_dropout": tune.loguniform(1e-1, 3e-1),
        #"dropout": tune.loguniform(1e-1, 3e-1),
        #"fc_dropout": tune.loguniform(2e-1, 5e-1),
        #"epoch": tune.randint(2, 4),
        "lr": tune.loguniform(5e-5, 2e-3),
    }

    np.random.seed(7654321)

    print("start tuning...")
    
    start_time = time.time()
    result = flaml.tune.run(
        tune.with_parameters(trainable_cls),
        config=config,
        metric="score",
        mode="min",
        low_cost_partial_config={"n_layers": 1},
        #max_resource=max_num_epoch,
        max_resource=4,
        min_resource=1,
        scheduler="asha",  # need to use tune.report to report intermediate results in trainner
        #resources_per_trial=resources_per_trial,
        local_dir=output_path,
        num_samples=num_samples,
        time_budget_s=time_budget_s,
        use_ray=True)

    print(f"#trials={len(result.trials)}")
    print(f"time={time.time() - start_time}")
    best_trial = result.get_best_trial("score", "min", "all")
    print("Best trial config: {}".format(best_trial.config))
    print("Best trial final validation loss: {}".format(
        best_trial.metric_analysis["loss"]["min"]))
    print("Best trial final validation score: {}".format(
        best_trial.metric_analysis["score"]["min"]))
    print("Best trial final validation farm_score: {}".format(
    best_trial.metric_analysis["farm_score"]["min"]))

    """
    test 
    """
    print("test on best_trained_model ...")
    
    checkpoint_path = os.path.join(best_trial.checkpoint.value, "checkpoint")
    print("best model path:", checkpoint_path)

    config_env["n_layers"] = best_trial.config["n_layers"]
    config_env["n_heads"] = int(2**best_trial.config["n_heads"])
    config_env["d_model"] = int(2 ** best_trial.config["d_model"])
    #config_env["d_ff"] = int(2 ** best_trial.config["d_ff"])
    #config_env["attn_dropout"] = best_trial.config["attn_dropout"]
    #config_env["fc_dropout"] = best_trial.config["fc_dropout"]
    #config_env["dropout"] = best_trial.config["dropout"]
    config_env["lr"] = best_trial.config["lr"]
    
    loss_fn = getattr(loss_factory, config_env["loss_name"])()
    print(loss_fn)    
    
    valid_r = evaluate(model=None,
            loss_fn = loss_fn,
            config = config_env,
            data_mean=None,
            data_scale = None,
            col_names=None,
            output_path=best_trial.checkpoint.value,
            tag="val",
            model_path=checkpoint_path)
    
    print("********Valid: " + str(dict(valid_r)))

    # Incremental train best model with validdata
    if not config_env["is_debug"]:
        print("Incremental train best model with validdata...")
        Incremental_train_validdata(config_env, checkpoint_path)
        


def task_arg_parser(argv, usage=None):
    """
    :param argv:
    :return:

    TODO: add param 'n_concurrent_trials'
    """
    import argparse

    parser = argparse.ArgumentParser(prog='main', usage=usage)

    parser.add_argument('--node_list', type=str, help="""[optional] List of node ip address:
                        for run distributed Ray applications on some local nodes available on premise.
                        the first node will be choose to be the head node.
                        """)
    parser.add_argument('--host_file', type=str, help="""[optional] same as node_list, host file path, List of node ip address.
                        """)
    parser.add_argument('--remote', action="store_true", default=False,
                        help="True, submit job outside the Ray cluster, False, submit inside the ray cluster")
    parser.add_argument('--train_data', type=str, help='path of train data')
    parser.add_argument('--test_data', type=str, help='[optional] path of test data')
    parser.add_argument('--output_path', type=str, help='[optional] output path for model to save')
    parser.add_argument('--time_budget', type=int, default=60,
                        help='constrains the wall-clock time (seconds) used by the AutoML process. default: 60')
    parser.add_argument('--resources_per_trial', type=str,
                        help='resources dict like: {"cpu": 1, "gpu": gpus_per_trial}, number of cpus, gpus for each trial; 0.5 means two training jobs can share one gpu')
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument('--num_samples', type=int, help='maximal number of trials')
    
    parser.add_argument("--input_len", type=int, default=288)
    parser.add_argument("--output_len", type=int, default=288)
    parser.add_argument("--start_col", type=int, default=0)
    parser.add_argument("--var_len", type=int, default=12)
    parser.add_argument("--var_out", type=int, default=1)
    parser.add_argument("--day_len", type=int, default=144)
    
    parser.add_argument("--train_days", type=int, default=100)
    parser.add_argument("--sample_step", type=int, default=1)
    parser.add_argument("--val_days", type=int, default=31)
    
    parser.add_argument("--model_name", type=str, choices=['Transformer', 'TST'], default='TST')
    
    parser.add_argument("--loss_name", type=str, choices=['FilterMSELoss', 'SCORELoss', 'SCORELoss2'], default='FilterMSELoss')
    parser.add_argument("--log_per_steps", type=int, default=100)
    
    parser.add_argument("--n_layers", type=int, default=2)
    parser.add_argument("--max_seq_len", type=int, default=512)
    parser.add_argument("--n_heads", type=int, default=16)
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--d_ff", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--attn_dropout", type=float, default=0.0)
    parser.add_argument("--fc_dropout", type=float, default=0.0)
    
    parser.add_argument("--epoch", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--patient", type=int, default=5)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--capacity", type=int, default=134)
    parser.add_argument("--model_path", type=str, default="checkpoints/TST")
    parser.add_argument('--is_debug', action="store_true", default=False,
                        help="True, is_debug mod")  
    parser.add_argument('--real_value', action="store_true", default=False,
                        help="True, real_value mod")
    parser.add_argument(
        "--num-workers",
        "-n",
        type=int,
        default=2,
        help="Sets number of workers for training.",
    )
    parser.add_argument(
        "--num-gpus-per-worker",
        type=int,
        default=0,
        help="Sets number of gpus each worker uses.",
    )
    parser.add_argument(
        "--cluster",
        action="store_true",
        default=False,
        help="enables multi-node tuning",
    )
    parser.add_argument(
        "--workers-per-node",
        type=int,
        help="Forces workers to be colocated on machines if set.",
    )      

    args = parser.parse_args(argv)
    args = vars(args)
    if args['node_list']:
        args['node_list'] = eval(args['node_list'])
    elif args['host_file'] and os.path.exists(args['host_file']):
        with open(args['host_file'], "r") as f:
            lines = f.readlines()
            args['node_list'] = [line.split(' ')[0].strip() for line in lines]
            print("node_list in host_file: ", args['node_list'])

    if args['resources_per_trial']:
        args['resources_per_trial'] = eval(args['resources_per_trial'])
        for k, v in args['resources_per_trial'].items():
            args['resources_per_trial'][k] = float(v)
    
    config = prep_env()
    config.update(args)
    print("configs:", config)
    
    return config


def task_manager(argv, usage=None):
    """
    job manager: job args parser, job submit
    :param argv:
    :param usage:
    :return:
    """
    args = task_arg_parser(argv, usage=usage)
    print("parse args: '{}'".format(args))

    try:
        # Startup Ray
        if args['node_list'] and len(args['node_list']) > 1:
            from autotabular import RayCluster
            ray_op = RayCluster()
            node_list, remote = ray_op.startup(node_list=args['node_list'], remote=args['remote'])
            args['node_list'] = node_list
            args['remote'] = remote

        print("node_list: ", args['node_list'])
        if args['node_list']==None or len(args['node_list']) <= 1:
            args["workers_per_node"] = None
            #print("workers_per_node: ", args["workers_per_node"])
            
        # run task
        automl_tune(args)

    except Exception as e:
        print("except:", e)
    finally:
        # Cleanup Ray
        if args['node_list'] and len(args['node_list']) > 1:
            from autotabular import RayCluster
            ray_op = RayCluster()
            ray_op.clean_up(args['node_list'], args['remote'])


def main(argv):
    usage = '''
    example1:
        python3 ./tune_kdd_tsai.py --output_path /nfs/volume-807-1/fanshuangxi/kdd_tsai.outputs --time_budget 300 --resources_per_trial '{"cpu":1,"gpu":1}' --num_samples 500 --num-workers 2 --num-gpus-per-worker 1 --input_len 288 --train_days 214 --sample_step 1 --val_days 31 --model_name TST --loss_name FilterMSELoss --max_seq_len 512 --n_layers 2 --n_heads 4 --d_model 64 --d_ff 64 --dropout 0.0 --attn_dropout 0.0 --fc_dropout 0.0 --epoch 5 --batch_size 64 --patient 2 --lr 0.0001 --model_path checkpoints/TST
            or
        sh run.sh ./tune_kdd_tsai.py --host_file /etc/HOROVOD_HOSTFILE --output_path /nfs/volume-807-1/fanshuangxi/kdd_tsai.outputs --time_budget 300 --resources_per_trial '{"cpu":1,"gpu":1}' --num_samples 500 --num-workers 4 --num-gpus-per-worker 1 --input_len 288 --train_days 214 --val_days 31 --model_name TST --loss_name FilterMSELoss --max_seq_len 512 --n_layers 2 --n_heads 4 --d_model 64 --d_ff 64 --dropout 0.0 --attn_dropout 0.0 --fc_dropout 0.0 --epoch 5 --batch_size 64 --patient 2 --lr 0.0001 --model_path checkpoints/TST
            '''
    print("argv:", argv)
    task_manager(argv, usage)


if __name__ == "__main__":
    main(sys.argv[1:])

