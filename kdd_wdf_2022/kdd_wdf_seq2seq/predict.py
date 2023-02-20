
import torch
import torch.nn.functional as F
import numpy as np

from wpf_dataset import PGL4WPFDataset, TestPGL4WPFDataset
from utils import load_model
from prepare import prep_env

def forecast(config):
    # graph = graph.tensor(train_data.graph)
    if config["model_name"] == 'ASTGCN':
        from astgcn import ASTGCN as network
    elif config["model_name"] == 'GruAtt':
        from gru_att import GruAtt as network
    elif config["model_name"] == 'Transformer':
        from transformer import Transformer as network
    elif config["model_name"] == 'Autoformer':
        from autoformer import Autoformer as network
    elif config["model_name"] == "SatGruAtt":
        from satgruatt import SatGruAtt as network
    elif config["model_name"] == 'WPFModel':
        from orinal_model import WPFModel as network
    model = network(config)
    global_step = load_model(config["model_path"], model)
    model.eval()
    with torch.no_grad():
        test_x_ds = TestPGL4WPFDataset(filename=config["path_to_test_x"])
        test_x = torch.Tensor( test_x_ds.get_data()[:, :, -config["input_len"]:, :])
        data_mean = torch.load("./data_mean.pt").to(test_x.device)
        data_scale = torch.load("./data_scale.pt").to(test_x.device)
        if config["model_name"] in ['ASTGCN', 'WPFModel']:
            graph = np.loadtxt('./adj_matrix.csv')
            pred_y = model((test_x[..., 2:] - data_mean) / data_scale, test_x[:, 0, :, :2], graph)
        else:
            pred_y = model((test_x[..., 2:] - data_mean) / data_scale, test_x[:, 0, :, :2])
        if config["real_value"] == False:
            pred_y = F.relu(pred_y * data_scale[:, :, :, -1] + data_mean[:, :, :, -1])
        pred_y = pred_y.cpu().numpy()
        pred = np.transpose(pred_y, [1, 2, 0])
    return pred

if __name__ == "__main__":
    config = prep_env()
    pred, gt, raw_data = forecast(config)  #, valid_data, test_data)
    print(pred.shape, gt.shape, len(raw_data), len(raw_data[0]))
