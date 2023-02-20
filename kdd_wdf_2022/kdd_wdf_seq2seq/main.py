
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
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch.cuda.amp as amp

def data_augment(X, y, p=0.8, alpha=0.5, beta=0.5):
    """Regression SMOTE
    """
    fix_X, X = X[:, :, :, :2], X[:, :, :, 2:]
    fix_y, y = y[:, :, :, :2], y[:, :, :, 2:]
    batch_size = X.shape[0]
    random_values = torch.rand([batch_size])
    idx_to_change = random_values < p

    # ensure that first element to switch has probability > 0.5
    random_betas = np.random.beta(alpha, beta, batch_size) / 2 + 0.5
    random_betas = torch.Tensor(random_betas).reshape([-1, 1, 1, 1])
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

    if config["model_name"] == 'ASTGCN':
        from astgcn import ASTGCN as network
    elif config["model_name"] == 'GruAtt':
        from gru_att import GruAtt as network
    elif config["model_name"] == 'Transformer':
        from transformer import Transformer as network
    elif config["model_name"] == 'Autoformer':
        from autoformer import Autoformer as network
    elif config["model_name"] == 'WPFModel':
        from orinal_model import WPFModel as network
    elif config["model_name"] == "SatGruAtt":
        from satgruatt import SatGruAtt as network
    model = network(config)

    gpus = torch.cuda.device_count()
    device_ids = []
    for i in range(gpus): device_ids.append(i)
    print('used the gpu devices: ', device_ids, gpus)
    model = nn.DataParallel(model, device_ids=device_ids).cuda()
    # model = model.to('cuda:0')
    loss_fn = getattr(loss_factory, config["loss_name"])()
    print(loss_fn)
    opt = optim.get_optimizer(model=model, learning_rate=config["lr"])
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, config["epoch"])
    # lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt,
    #                                                        factor=0.1,
    #                                                        mode='min',
    #                                                        patience=5,
    #                                                        min_lr=0,
    #                                                        verbose=True)
    from datetime import datetime
    current_time = datetime.now().strftime('%Y%m%d%H%M%S')
    output_path = os.path.join(config["checkpoints"], config["model_name"], current_time)
    _create_if_not_exist(output_path)
    global_step = 0
    best_score = np.inf
    patient = 0
    valid_records = []

    scaler = amp.GradScaler()
    for epoch in range(config["epoch"]):
        torch.cuda.empty_cache()
        model.train()
        train_data = PGL4WPFDataset(
            config["data_path"],
            filename=config["filename"],
            size=[config["input_len"], config["output_len"]],
            flag='train',
            total_days=config["total_days"],
            train_days=config["train_days"],
            val_days=config["val_days"],
            test_days=config["test_days"])
        data_mean = torch.Tensor(train_data.data_mean).cuda()
        data_scale = torch.Tensor(train_data.data_scale).cuda()
        # torch.save(data_mean, "./data_mean.pt")
        # torch.save(data_scale, "./data_scale.pt")
        train_data_loader = DataLoader(
            train_data,
            batch_size=config["batch_size"],
            shuffle=True,
            drop_last=True)
        col_names = dict([(v, k) for k, v in enumerate(train_data.get_raw_df()[0].columns)])
        del train_data
        t_time = time.time()
        t_loss = []
        for id, (batch_x, batch_y) in enumerate(train_data_loader):
            # print(batch_x.shape, batch_y.shape)
            torch.cuda.empty_cache()
            opt.zero_grad()
            batch_x, batch_y = data_augment(batch_x.type(torch.float32), batch_y.type(torch.float32))
            # batch_x, batch_y = batch_x.type(torch.float32), batch_y.type(torch.float32)
            batch_x, batch_y = batch_x.cuda(), batch_y.cuda()
            with torch.cuda.amp.autocast():
                pred_y = model((batch_x[..., 2:] - data_mean) / data_scale, batch_x[:, 0, :, :2])
                if config["real_value"]:
                    train_loss = loss_fn(pred_y, batch_y[:, :, :, -1], batch_y, col_names)
                else:
                    train_loss = loss_fn(pred_y, (batch_y[:, :, :, -1] - data_mean[:, :, :, -1]) / data_scale[:, :, :, -1], batch_y, col_names)
            t_loss.append(train_loss.item())
            scaler.scale(train_loss).backward()
            scaler.step(opt)
            scaler.update()
            global_step += 1
            if global_step % config["log_per_steps"] == 0 or global_step == 1:
                print("Step %s Train filter-Loss: %s " % (global_step, train_loss.item() ) )
        print('**********Train Epoch {}: averaged Loss: {:.6f}, use time {:.3f} min'.format(epoch, np.mean(t_loss), (time.time() - t_time) / 60 ))
        print('before del',torch.cuda.memory_allocated()/1024/1024)
        del pred_y, batch_y, batch_x, train_data_loader
        print('after del',torch.cuda.memory_allocated()/1024/1024)
        lr_scheduler.step()
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
        save_model(output_path, model, steps=epoch, opt=opt, lr_scheduler=lr_scheduler)
        if best_score == valid_r['score']:
            patient = 0
        else:
            patient += 1
            if patient > config["patient"]:
                break
    best_epochs = min(enumerate(valid_records), key=lambda x: x[1]["score"])[0]
    print("Best valid Epoch %s" % best_epochs)
    print("Best valid score %s" % valid_records[best_epochs])


def visualize_prediction(input_batch, pred_batch, gold_batch, tag, out_path):
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    plt.figure()
    for i in range(1, 5):
        ax = plt.subplot(2, 2, i)
        ax.plot(
            np.concatenate(
                [input_batch[288 * (i - 1)], gold_batch[288 * (i - 1)]]),
            label="gold")
        ax.plot(
            np.concatenate(
                [input_batch[288 * (i - 1)], pred_batch[288 * (i - 1)]]),
            label="pred")
        ax.legend()
    plt.savefig(os.path.join(out_path, tag + "_vis.png"))
    plt.close()


def evaluate(model,
             loss_fn,
             config,
             data_mean,
             data_scale,
             col_names,
             output_path,
             tag="val"):
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
    step = 0
    pred_batch = []
    gold_batch = []
    input_batch = []
    losses = []

    with torch.no_grad():
        for id, (batch_x, batch_y) in enumerate(valid_data_loader):
            batch_x = batch_x.type(torch.float32).cuda()
            batch_y = batch_y.type(torch.float32).cuda()
            with torch.cuda.amp.autocast():
                pred_y = model((batch_x[..., 2:] - data_mean) / data_scale, batch_x[:, 0, :, :2])
            if config["real_value"]:
                __loss = loss_fn(pred_y, batch_y[:, :, :, -1], batch_y, col_names)
            else:
                __loss = loss_fn(pred_y, (batch_y[:, :, :, -1] - data_mean[:, :, :, -1]) / data_scale[:, :, :, -1], batch_y, col_names)
                pred_y = F.relu(pred_y * data_scale[:, :, :, -1] + data_mean[:, :, :, -1])

            losses.append(__loss.item())

            input_batch.append(batch_x[:, :, :, -1].cpu().numpy())
            pred_batch.append(pred_y.cpu().numpy())
            gold_batch.append(batch_y[..., -1].cpu().numpy())
            step += 1
    del pred_y, batch_y, batch_x, valid_data_loader
    valid_raw_df = valid_data.get_raw_df()
    del valid_data
    pred_batch = np.concatenate(pred_batch, axis=0).transpose((1, 0, 2))
    gold_batch = np.concatenate(gold_batch, axis=0).transpose((1, 0, 2))
    input_batch = np.concatenate(input_batch, axis=0).transpose((1, 0, 2))

    visualize_prediction(
        np.sum(input_batch, 0) / 1000.,
        np.sum(pred_batch, 0) / 1000.,
        np.sum(gold_batch, 0) / 1000., tag, output_path)
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
    config = prep_env()

    train_and_evaluate(config)
