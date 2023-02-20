# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import gc
gc.collect()
import os.path

import torch, time
import torch.nn.functional as F
import numpy as np

from torch.utils.data import DataLoader
import torch.nn as nn
from prepare import prep_env
import loss as loss_factory
from wpf_dataset import PGL4WPFDataset
import optimization as optim
from metrics import regressor_scores, regressor_detailed_scores
from utils import save_model, _create_if_not_exist, load_model


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


def train_and_evaluate(config):
    if torch.cuda.is_available():
        torch.cuda.set_device(0)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print("device:", device)

    print("load data ...")
    train_data = PGL4WPFDataset(
        config["data_path"],
        filename=config["filename"],
        size=[config["input_len"], config["output_len"]],
        flag='train',
        total_days=config["total_days"],
        train_days=config["train_days"],
        val_days=config["val_days"],
        test_days=config["test_days"]
    )

    data_scale_path = config["data_scale_path"]
    try:
        scale_state_dict = torch.load(data_scale_path)

        data_mean = scale_state_dict['data_mean'].to(device)
        data_scale = scale_state_dict['data_scale'].to(device)
    except Exception as e:
        data_mean = torch.Tensor(train_data.data_mean)
        data_scale = torch.Tensor(train_data.data_scale)

        state = {'data_mean': data_mean, 'data_scale': data_scale}
        torch.save(state, data_scale_path)
        print("save data_scale to %s" % data_scale_path)

    data_mean = data_mean.cuda()
    data_scale = data_scale.cuda()

    train_data_loader = DataLoader(
        train_data,
        batch_size=config["batch_size"],
        shuffle=True,
        drop_last=True)
    col_names = dict([(v, k) for k, v in enumerate(train_data.get_raw_df()[0].columns)])

    gpus = torch.cuda.device_count()
    device_ids = []
    for i in range(gpus): device_ids.append(i)
    print('used the gpu devices: ', device_ids, gpus)
    if config["model_name"] == "TST":
        #from tsai.all import TSTPlus
        from tsai.models.TSTPlus import TSTPlus
        model = TSTPlus(config["var_len"], config["output_len"], config["input_len"],
                        max_seq_len=config["max_seq_len"],
                        n_layers=config["n_layers"],
                        n_heads=config["n_heads"],
                        d_model=config["d_model"],
                        d_ff=config["d_ff"],
                        attn_dropout=config["attn_dropout"],
                        dropout=config["dropout"],
                        fc_dropout=config["fc_dropout"])
    elif config["model_name"] == "SeqCNN":
        from cnn_tst import SeqCNN
        model = SeqCNN(config, fc_output=True)
    elif config["model_name"] == "CnnTst":
        from cnn_tst import CnnTst
        model = CnnTst(config)

    model.to(device)
    # model = TSTPlus(dls.vars, dls.c, dls.len, dropout=.3)
    # model = TST(dls.vars, dls.c, dls.len, dropout=.1, fc_dropout=.8)
    # model.to(device)
    loss_fn = getattr(loss_factory, config["loss_name"])()
    print(loss_fn)

    opt = optim.get_optimizer(model=model, learning_rate=config["lr"])
    # lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, config.epoch)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt,
                                                           factor=0.1,
                                                           mode='min',
                                                           patience=5,
                                                           min_lr=0,
                                                           verbose=True)

    from datetime import datetime
    current_time = datetime.now().strftime('%Y%m%d%H%M%S')
    output_path = os.path.join(config["checkpoints"], config["model_name"], current_time)
    _create_if_not_exist(output_path)

    cat_var_len = config["cat_var_len"]

    global_step = 0
    best_score = np.inf
    patient = 0
    valid_records = []

    for epoch in range(config["epoch"]):
        torch.cuda.empty_cache()
        model.train()

        t_time = time.time()
        t_loss = []
        for id, (batch_x, batch_y) in enumerate(train_data_loader):

            if (id + epoch) % config["sample_step"]:
                continue

            torch.cuda.empty_cache()
            # batch_x, batch_y = data_augment(batch_x.type(torch.float32), batch_y.type(torch.float32))
            batch_x, batch_y = batch_x.type(torch.float32).to(device), batch_y.type(torch.float32).to(device)

            if not config["real_value"]:
                batch_x[..., cat_var_len:] = (batch_x[..., cat_var_len:] - data_mean[..., cat_var_len:]) / data_scale[..., cat_var_len:]

            # X: [B,L,C] | [B,N,L,C] => [B,L,C]
            X = batch_x.reshape(-1, batch_x.shape[-2], batch_x.shape[-1])  # [B,L,C]

            if config["model_name"] == "TST":
                # X: [B,C,L] | [B,N,L,C] => [B,L,C] => [B,C,L]  just for "TST" model
                X = X.permute(0, 2, 1) # [B,C,L]
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
                print("Step %s Train score-Loss: %s " % (global_step, train_loss.item()))
        print('**********Train Epoch {}: averaged Loss: {:.6f}, use time {:.3f} min'.format(epoch, np.mean(t_loss),
                                                                                            (time.time() - t_time) / 60))

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

        lr_scheduler.step(valid_r['score'])

        best_score = min(valid_r['score'], best_score)
        # test_records.append(test_r)
        if best_score == valid_r['score']:
            patient = 0
            print("save model to: ", output_path)
            save_model(output_path=output_path, model=model, steps=epoch, opt=opt, lr_scheduler=lr_scheduler)
        else:
            patient += 1
            if patient > config["patient"]:
                break
    best_epochs = min(enumerate(valid_records), key=lambda x: x[1]["score"])[0]
    print("Best valid Epoch %s" % best_epochs)
    print("Best valid score %s" % valid_records[best_epochs])
    # print("Best valid test-score %s" % test_records[best_epochs])
    
    del train_data
    # Incremental train best model with validdata
    if not config["is_debug"]:
        print("Incremental train best model with validdata...")
        Incremental_train_validdata(config, output_path, data_mean, data_scale)


def Incremental_train_validdata(config, model_path, data_mean, data_scale):
    if torch.cuda.is_available():
        torch.cuda.set_device(0)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print("device:", device)

    valid_data = PGL4WPFDataset(
        config["data_path"],
        filename=config["filename"],
        size=[config["input_len"], config["output_len"]],
        flag="val",
        total_days=config["total_days"],
        train_days=config["train_days"],
        val_days=config["val_days"],
        test_days=config["test_days"])
    valid_data_loader = DataLoader(
        valid_data,
        batch_size=config["batch_size"],
        shuffle=False,
        drop_last=True)

    col_names = dict([(v, k) for k, v in enumerate(valid_data.get_raw_df()[0].columns)])
    model = None
    if config["model_name"] == "TST":
        from tsai.all import TSTPlus
        model = TSTPlus(config["var_len"], config["output_len"], config["input_len"],
                        max_seq_len=config["max_seq_len"],
                        n_layers=config["n_layers"],
                        n_heads=config["n_heads"],
                        d_model=config["d_model"],
                        d_ff=config["d_ff"],
                        attn_dropout=config["attn_dropout"],
                        dropout=config["dropout"],
                        fc_dropout=config["fc_dropout"])
    elif config["model_name"] == "SeqCNN":
        from cnn_tst import SeqCNN
        model = SeqCNN(config, fc_output=True)
    elif config["model_name"] == "CnnTst":
        from cnn_tst import CnnTst
        model = CnnTst(config)

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
    print("model_step:", model_step)

    gpus = torch.cuda.device_count()
    device_ids = []
    for i in range(gpus): device_ids.append(i)
    print('used the gpu devices: ', device_ids, gpus)
    model = nn.DataParallel(model, device_ids=device_ids).cuda()
    # model = model.to('cuda:0')
    loss_fn = getattr(loss_factory, config["loss_name"])()
    print(loss_fn)

    cat_var_len = config["cat_var_len"]
    from datetime import datetime
    output_path = model_path + "_alldata"
    _create_if_not_exist(output_path)
    global_step = 0
    best_score = np.inf
    patient = 0
    valid_records = []
    for epoch in range(model_step+1):
        torch.cuda.empty_cache()
        model.train()

        t_time = time.time()
        t_loss = []
        for id, (batch_x, batch_y) in enumerate(valid_data_loader):
            # print(batch_x.shape, batch_y.shape)
            if (id + epoch) % config["sample_step"]:
                continue

            torch.cuda.empty_cache()
            # if id==0: print('before model',torch.cuda.memory_allocated()/1024/1024)

            opt.zero_grad()
            # batch_x, batch_y = data_augment(batch_x.type(torch.float32), batch_y.type(torch.float32))
            batch_x, batch_y = batch_x.type(torch.float32).to(device), batch_y.type(torch.float32).to(device)

            if not config["real_value"]:
                batch_x[..., cat_var_len:] = (batch_x[..., cat_var_len:] - data_mean[..., cat_var_len:]) / data_scale[..., cat_var_len:]

            # X: [B,L,C] | [B,N,L,C] => [B,L,C]
            X = batch_x.reshape(-1, batch_x.shape[-2], batch_x.shape[-1])  # [B,L,C]

            if config["model_name"] == "TST":
                # X: [B,C,L] | [B,N,L,C] => [B,L,C] => [B,C,L]  just for "TST" model
                X = X.permute(0, 2, 1) # [B,C,L]
                # y = batch_y.reshape(-1, batch_y.shape[-2], batch_y.shape[-1])[:,:,-1:].permute(0, 2, 1)

            pred_y = model(X)

            pred_y = pred_y.reshape(-1, batch_y.shape[-3], batch_y.shape[-2])

            if config["real_value"]:
                train_loss = loss_fn(pred_y, batch_y[:, :, :, -1], batch_y, col_names)
            else:
                train_loss = loss_fn(pred_y, (batch_y[:, :, :, -1] - data_mean[:, :, :, -1]) / data_scale[:, :, :, -1],
                                     batch_y, col_names)

            t_loss.append(train_loss.item())
            train_loss.backward()
            opt.step()
            global_step += 1
            if global_step % config["log_per_steps"] == 0 or global_step == 1:
                print("Step %s Train score-Loss: %s " % (global_step, train_loss.item()))
        print('**********Train Epoch {}: averaged Loss: {:.6f}, use time {:.3f} min'.format(epoch, np.mean(t_loss), (
                    time.time() - t_time) / 60))
        # print('before del',torch.cuda.memory_allocated()/1024/1024)
        # del pred_y, batch_y, batch_x, train_data, train_data_loader
        print('after del', torch.cuda.memory_allocated() / 1024 / 1024)

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

        # lr_scheduler.step(valid_r['score'])

        best_score = min(valid_r['score'], best_score)

    print("Incremental train model with validdata saved to: ", output_path)
    save_model(output_path, model, steps=global_step, opt=opt, lr_scheduler=lr_scheduler)


def evaluate(model,
             loss_fn,
             config,
             data_mean,
             data_scale,
             col_names,
             output_path,
             tag="val"):

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print("device:", device)

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
    valid_data_loader = DataLoader(
        valid_data,
        batch_size=config["batch_size"],
        shuffle=False,
        drop_last=True)

    cat_var_len = config["cat_var_len"]
    step = 0
    pred_batch = []
    gold_batch = []
    input_batch = []
    losses = []
    print('val loaded data', torch.cuda.memory_allocated() / 1024 / 1024)

    with torch.no_grad():
        for id, (batch_x, batch_y) in enumerate(valid_data_loader):
            #batch_x = batch_x.type(torch.float32).to(device)
            #batch_y = batch_y.type(torch.float32).to(device)
            batch_x = batch_x.type(torch.float32).cuda()
            batch_y = batch_y.type(torch.float32).cuda()
            if not config["real_value"]:
                batch_x[..., cat_var_len:] = (batch_x[..., cat_var_len:] - data_mean[..., cat_var_len:]) / data_scale[..., cat_var_len:]

            X = batch_x.reshape(-1, batch_x.shape[-2], batch_x.shape[-1])  # [B,L,C]

            if config["model_name"] == "TST":
                # X: [B,C,L] | [B,N,L,C] => [B,L,C] => [B,C,L]  just for "TST" model
                X = X.permute(0, 2, 1) # [B,C,L]
                # y = batch_y.reshape(-1, batch_y.shape[-2], batch_y.shape[-1])[:,:,-1:].permute(0, 2, 1)

            pred_y = model(X)
            #print("batch_y.shape[-3]: {}, batch_y.shape[-2]: {}".format(batch_y.shape[-3], batch_y.shape[-2]))
            pred_y = pred_y.reshape(-1, batch_y.shape[-3], batch_y.shape[-2])
            #print("pred_y.shape: {}".format(pred_y.shape))
                
            if config["real_value"]:
                __loss = loss_fn(pred_y, batch_y[:, :, :, -1], batch_y, col_names)
            else:
                __loss = loss_fn(pred_y, (batch_y[:, :, :, -1] - data_mean[:, :, :, -1]) / data_scale[:, :, :, -1], batch_y, col_names)
                #print("data_scale.shape {}, data_scale[:, :, :, -1].shape{}".format(data_scale.shape, data_scale[:, :, :, -1].shape))
                pred_y = F.relu(pred_y * data_scale[:, :, :, -1] + data_mean[:, :, :, -1])

                #print("pred_y.shape: {}".format(pred_y.shape))

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
    """
    visualize_prediction(
        np.sum(input_batch, 0) / 1000.,
        np.sum(pred_batch, 0) / 1000.,
        np.sum(gold_batch, 0) / 1000., tag, output_path)
    """

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


if __name__ == "__main__":
    import argparse
    
    usage = '''
    example1:
        python3 ./main_tsai.py --input_len 288 --train_days 20 --sample_step 4 --val_days 31 --model_name TST --loss_name FilterMSELoss --max_seq_len 512 --conv_layers 2 --n_layers 2 --n_heads 4 --d_model 64 --d_ff 64 --dropout 0.0 --attn_dropout 0.0 --fc_dropout 0.0 --num_workers 4 --epoch 2 --batch_size 2 --patient 2 --lr 0.0001 
            '''
            
    parser = argparse.ArgumentParser(description='main', usage=usage)
    parser.add_argument("--input_len", type=int, default=288)
    parser.add_argument("--output_len", type=int, default=288)
    parser.add_argument("--start_col", type=int, default=0)
    parser.add_argument("--var_len", type=int, default=13)
    parser.add_argument("--var_out", type=int, default=1)

    parser.add_argument("--day_len", type=int, default=144)
    
    parser.add_argument("--train_days", type=int, default=100)
    parser.add_argument("--sample_step", type=int, default=1)
    parser.add_argument("--val_days", type=int, default=31)

    parser.add_argument("--model_name", type=str, choices=['TST', 'SeqCNN', 'CnnTst'], default='CnnTst')
    parser.add_argument("--tst_mod", type=str, choices=['tst', 'tst_plus'], default='tst_plus')
    
    parser.add_argument("--loss_name", type=str, choices=['FilterMSELoss', 'SCORELoss', 'SCORELoss2'], default='FilterMSELoss')

    parser.add_argument("--conv_layers", type=int, default=3)
    parser.add_argument("--pool_mod",  type=str, choices=['max', 'avg'], default='max')
    parser.add_argument("--n_layers", type=int, default=2)

    parser.add_argument("--log_per_steps", type=int, default=100)

    parser.add_argument("--max_seq_len", type=int, default=512)

    parser.add_argument("--n_heads", type=int, default=16)
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--d_ff", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--attn_dropout", type=float, default=0.0)
    parser.add_argument("--fc_dropout", type=float, default=0.0)
    
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--epoch", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--patient", type=int, default=5)
    parser.add_argument("--lr_max", type=float, default=0.0001)
    parser.add_argument("--capacity", type=int, default=134)
    parser.add_argument("--model_path", type=str, default="checkpoints/TST")
    parser.add_argument('--is_debug', action="store_true", default=False,
                        help="True, is_debug mod")
    parser.add_argument('--real_value', action="store_true", default=False,
                        help="True, real_value mod")
    args = parser.parse_args()
    args = vars(args)
    
    config = prep_env()
    
    config.update(args)
    
    print("configs:", config)

    train_and_evaluate(config)
