
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
from add_day_dataset import PGL4WPFDataset
import optimization as optim
from metrics import regressor_scores, regressor_detailed_scores
# from test_metrics import regressor_detailed_scores
from utils import save_model, _create_if_not_exist
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def data_augment(X, y, p=0.8, alpha=0.5, beta=0.5):
    """Regression SMOTE
    """
    fix_y, y = y[:, :, :, :2], y[:, :, :, 2:]
    bs, n, d, t, f = X.shape
    X = X.reshape(bs, n, d * t, f)
    random_values = torch.rand([bs])
    idx_to_change = random_values < p

    # ensure that first element to switch has probability > 0.5
    random_betas = np.random.beta(alpha, beta, bs) / 2 + 0.5
    random_betas = torch.Tensor(random_betas).reshape([-1, 1, 1, 1])
    index_permute = torch.randperm(bs)

    X[idx_to_change] = random_betas[idx_to_change] * X[idx_to_change]
    X[idx_to_change] += (
        1 - random_betas[idx_to_change]) * X[index_permute][idx_to_change]

    y[idx_to_change] = random_betas[idx_to_change] * y[idx_to_change]
    y[idx_to_change] += (
        1 - random_betas[idx_to_change]) * y[index_permute][idx_to_change]
    return X.reshape(bs, n, d, t, f), torch.cat([fix_y, y], dim=-1)


def train_and_evaluate(config):
    if torch.cuda.is_available():
        torch.cuda.set_device(0)

    if config["model_name"] == 'ChebyNet':
        from astgcn2 import ChebyNet as network
    elif config["model_name"] == 'logTransformer':
        from transformer import logTransformer as network
    elif config["model_name"] == 'SatGruAtt':
        from satgruatt import SatGruAtt as network
    elif config["model_name"] == "Autoformer":
        from autoformer import Autoformer as network
    elif config["model_name"] == "Autoformer2":
        from autoformer_2 import Autoformer as network
    elif config["model_name"] == "Autoformer3":
        from autoformer3 import Autoformer as network
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
    # lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.1, mode='min', patience=5, min_lr=0, verbose=True)
    from datetime import datetime
    current_time = datetime.now().strftime('%Y%m%d%H%M%S')
    output_path = os.path.join(config["checkpoints"], config["model_name"], current_time)
    _create_if_not_exist(output_path)
    global_step = 0
    best_score = np.inf
    patient = 0
    valid_records = []
    for epoch in range(config["epoch"]):
        torch.cuda.empty_cache()
        model.train()
        train_data = PGL4WPFDataset(
            config["data_path"],
            filename=config["filename"],
            size=[config["input_len"], config["output_len"]],
            flag='train',
            day_seq=config["day_seq_len"],
            total_days=config["total_days"],
            train_days=config["train_days"],
            val_days=config["val_days"],
            test_days=config["test_days"])

        data_mean = torch.Tensor(train_data.data_mean).cuda()
        data_scale = torch.Tensor(train_data.data_scale).cuda()
        train_data_loader = DataLoader(
            train_data,
            batch_size=config["batch_size"],
            shuffle=True,
            drop_last=True)
        col_names = dict([(v, k) for k, v in enumerate(train_data.get_raw_df()[0].columns)])
        graph = train_data.graph
        del train_data
        t_time = time.time()
        t_loss = []
        for id, (day_seq, batch_y) in enumerate(train_data_loader):
            torch.cuda.empty_cache()
            opt.zero_grad()
            day_seq, batch_y = data_augment(day_seq.type(torch.float32), batch_y.type(torch.float32))
            day_seq, batch_y = day_seq.cuda(), batch_y.cuda()
            day_seq = ((day_seq - data_mean.unsqueeze(2)) / data_scale.unsqueeze(2))
            if config["model_name"] in ['ChebyNet', 'WPFModel', 'SatGruAtt']:
                pred_y = model( day_seq, graph)
            else:
                pred_y = model(day_seq)
            # calculate loss
            if config["real_value"]:
                train_loss = loss_fn(pred_y, batch_y[:, :, :, -1], batch_y, col_names)
            else:
                train_loss = loss_fn(pred_y, (batch_y[:, :, :, -1] - data_mean[:, :, :, -1]) / data_scale[:, :, :, -1],
                                     batch_y, col_names)
            t_loss.append(train_loss.item())
            train_loss.backward()
            opt.step()
            # scaler.update()
            global_step += 1
            if global_step % config["log_per_steps"] == 0 or global_step == 1:
                print("Step %s Train filter-Loss: %s " % (global_step, train_loss.item() ) )
        print('**********Train Epoch {}: averaged Loss: {:.6f}, use time {:.3f} min'.format(epoch, np.mean(t_loss), (time.time() - t_time) / 60 ))
        print('before del',torch.cuda.memory_allocated()/1024/1024)
        del pred_y, batch_y, train_data_loader, day_seq
        print('after del',torch.cuda.memory_allocated()/1024/1024)
        lr_scheduler.step()
        valid_r = evaluate(model,
                           graph,
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


def visualize_prediction( pred_batch, gold_batch, tag, out_path):
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    plt.figure()
    for i in range(1, 5):
        ax = plt.subplot(2, 2, i)
        ax.plot(
                gold_batch[288 * (i - 1)],
            label="gold")
        ax.plot(
                pred_batch[288 * (i - 1)],
            label="pred")
        ax.legend()
    plt.savefig(os.path.join(out_path, tag + "_vis.png"))
    plt.close()


def evaluate(model,
             graph,
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
        day_seq=config["day_seq_len"],
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
    losses = []

    with torch.no_grad():
        for id, (day_seq, batch_y) in enumerate(valid_data_loader):
            batch_y = batch_y.type(torch.float32).cuda()
            day_seq = day_seq.type(torch.float32).cuda()
            day_seq = ((day_seq - data_mean.unsqueeze(2)) / data_scale.unsqueeze(2))
            if config["model_name"] in ['ChebyNet', 'WPFModel', 'SatGruAtt']:
                pred_y = model(day_seq, graph) #b, n, 288
            else:
                pred_y = model(day_seq)
            if config["real_value"]:
                __loss = loss_fn(pred_y, batch_y[:, :, :, -1], batch_y, col_names)
            else:
                __loss = loss_fn(pred_y, (batch_y[:, :, :, -1] - data_mean[:, :, :, -1]) / data_scale[:, :, :, -1], batch_y, col_names)
                pred_y = F.relu(pred_y * data_scale[:, :, :, -1] + data_mean[:, :, :, -1])

            losses.append(__loss.item())
            pred_batch.append(pred_y.cpu().numpy()) #b, n, 288
            gold_batch.append(batch_y[..., -1].cpu().numpy()) #b, n, 288
            step += 1
    del pred_y, batch_y, valid_data_loader, day_seq
    valid_raw_df = valid_data.get_raw_df()
    del valid_data
    pred_batch = np.concatenate(pred_batch, axis=0).transpose((1, 0, 2)) #n, b, 288
    gold_batch = np.concatenate(gold_batch, axis=0).transpose((1, 0, 2))

    visualize_prediction(
        np.sum(pred_batch, 0) / 1000.,
        np.sum(gold_batch, 0) / 1000., tag, output_path)
    _mae, _rmse = regressor_detailed_scores(pred_batch, gold_batch,
                                            valid_raw_df, 134, 288)

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
