import glob
import os.path

import torch
import torch.nn.functional as F
import numpy as np

from wpf_dataset import PGL4WPFDataset, TestPGL4WPFDataset
from metrics import regressor_scores, regressor_detailed_scores
from utils import save_model, _create_if_not_exist, load_model
import matplotlib.pyplot as plt
from prepare import prep_env
from datetime import datetime

def forecast(config):
    time = datetime.now()

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print("device:", device)

    model_type = 'pkl'
    print("model path:", config["checkpoints"])
    model = None
    if model_type=='pkl':
        print("load model from pkl")
        model = load_model(config["checkpoints"])
    else:
        print("load model from state_dict")
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
        
        global_step = load_model(config["checkpoints"], model)
        
    print('load model use time: ', (datetime.now()- time))
    model.to(device)
    model.eval()

    cat_var_len = config["cat_var_len"]

    with torch.no_grad():
        time = datetime.now()
        test_x_ds = TestPGL4WPFDataset(filename=config["path_to_test_x"])
        print('load TestPGL4WPFDataset use time: ', (datetime.now() - time))

        test_x = torch.Tensor(test_x_ds.get_data()[:, :, -config["input_len"]:, :]).to(device)
        # data_mean = torch.Tensor(test_x_ds.data_mean).to(test_x.device)
        # data_scale = torch.Tensor(test_x_ds.data_scale).to(test_x.device)
        #data_mean = torch.load(os.path.join(config["checkpoints"], "./data_mean.pt")).to(device)
        #data_scale = torch.load(os.path.join(config["checkpoints"], "./data_scale.pt")).to(device)

        print("test_x.shape:", test_x.shape)

        if not config["real_value"]:
            scale_state_dict = torch.load(os.path.join(config["checkpoints"], "./data_scale"))
            data_mean = scale_state_dict['data_mean'].to(device)
            data_scale = scale_state_dict['data_scale'].to(device)

            test_x[..., cat_var_len:] = (test_x[..., cat_var_len:] - data_mean[..., cat_var_len:]) / data_scale[...,
                                                                                                       cat_var_len:]

        X = test_x.reshape(-1, test_x.shape[-2], test_x.shape[-1])  # [B,L,C]

        if config["model_name"] == "TST":
            # X: [B,C,L] | [B,N,L,C] => [B,L,C] => [B,C,L]  just for "TST" model
            X = X.permute(0, 2, 1)  # [B,C,L]
            # y = batch_y.reshape(-1, batch_y.shape[-2], batch_y.shape[-1])[:,:,-1:].permute(0, 2, 1)

        pred_y = model(X)

        # print("batch_y.shape[-3]: {}, batch_y.shape[-2]: {}".format(batch_y.shape[-3], batch_y.shape[-2]))
        print("pred_y.shape:", pred_y.shape)
        pred_y = pred_y.reshape(-1, test_x.shape[-3], config["output_len"])
        # print("pred_y.shape: {}".format(pred_y.shape))
        print("pred_y.shape:", pred_y.shape)

        #print("pred_y[0,:]:", pred_y[0,:])
        print('to predict use time: ', (datetime.now() - time))
        if config["real_value"] == False:
            if device == "cpu":
                pred_y = pred_y.cpu()
            pred_y = F.relu(pred_y * data_scale[:, :, :, -1] + data_mean[:, :, :, -1])
        pred_y = pred_y.cpu().numpy()
        pred = np.transpose(pred_y, [1, 2, 0])
        #print("pred[0,:]:", pred[0,:])

    return pred


if __name__ == "__main__":
    config = prep_env()
    pred= forecast(config)  #, valid_data, test_data)
    print(pred.shape)
