
import torch, os
import torch.nn.functional as F
import numpy as np

from add_day_dataset import TestPGL4WPFDataset
from utils import load_model
from prepare import prep_env

def forecast(config):
    # graph = graph.tensor(train_data.graph)
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
    global_step = load_model(config["model_path"], model)
    model.eval()
    with torch.no_grad():
        test_x_ds = TestPGL4WPFDataset(filename=config["path_to_test_x"])
        day_seq = test_x_ds.get_data()
        day_seq = torch.Tensor(day_seq)
        print(day_seq.shape)
        data_mean = torch.load(os.path.join(config["checkpoints"], "./data_mean.pt")).to(day_seq.device)
        data_scale = torch.load(os.path.join(config["checkpoints"], "./data_scale.pt")).to(day_seq.device)
        day_seq = ((day_seq - data_mean.unsqueeze(2)) / data_scale.unsqueeze(2))
        if config["model_name"] in ['ASTGCN', 'WPFModel', 'SatGruAtt']:
            graph = np.loadtxt(os.path.join(config["checkpoints"],'./adj_matrix.csv'))
            pred_y = model(day_seq, graph)
        else:
            pred_y = model(day_seq)
        if config["real_value"] == False:
            pred_y = F.relu(pred_y * data_scale[:, :, :, -1] + data_mean[:, :, :, -1])
        pred_y = pred_y.cpu().numpy()
        pred = np.transpose(pred_y, [1, 2, 0])
    return pred

if __name__ == "__main__":
    config = prep_env()
    pred, gt, raw_data = forecast(config)  #, valid_data, test_data)
    print(pred.shape, gt.shape, len(raw_data), len(raw_data[0]))
