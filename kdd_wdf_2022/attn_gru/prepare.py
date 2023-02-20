import torch

def prep_env():
    settings = {
        "path_to_test_x": "./data/sdwpf_baidukddcup2022_test_toy/test_x",
        "path_to_test_y": "./data/sdwpf_baidukddcup2022_test_toy/test_y",
        #"data_path": "./data",
        #"data_scale_path": "./data/data_scale",
        "data_path": "/nfs/volume-807-1/fanshuangxi/dev/TS/kdd_wind_turbines/kdd_wdf_gnn/data",
        "data_scale_path": "./checkpoints/data_scale",

        "filename": "wtbdata_245days.csv",
        "task": "MS",
        "target": "Patv",
        "checkpoints": "checkpoints",
        #"input_len": 288,
        "input_len": 288,
        "output_len": 288,
        "start_col": 0,
        "var_len": 13,
        "cat_var_len": 3,
        "var_out": 1,
        "out_dim": 1,
        "day_len": 144,
        "train_days": 214,
        "sample_step": 1,
        #"train_days": 100,
        "val_days": 31,
        "test_days": 0,
        "total_days": 245,
        "model_name": 'GruAtt', # optinal: [ Transformer, GruAtt, GruAtt_v2, ASTGCN]
        "is_attention": False,
        "attention_mod": 'seq2cat',  # ['seq2cat', 'self']
        "projection_mod": 'linear',  # ['linear', 'ff']
        
        "loss_name": 'FilterMSELoss',
        #"real_value": False,
        "real_value": False,
        "log_per_steps": 100,

        "num_layers": 2,
        "dropout_att": 0.2,
        "dropout_proj": 0.5,
        "nhead": 4,
        "hidden_dims": 64,

        "num_workers": 5,
        #"epoch": 20,
        "epoch": 5,
        #"batch_size": 64,
        "batch_size": 8,
        #"patient": 5,
        "patient": 2,
        "lr": 1e-4,
        "gpu": 0,
        "capacity": 134,
        "pred_file": "predict.py",
        #"model_path": " checkpoints/ASTGCN/20220707184706",
        "model_path": " checkpoints/GruAtt/202207081010",
        "framework": "pytorch",
        "is_debug": True
    }

    if torch.cuda.is_available():
        settings["use_gpu"] = True
        torch.cuda.set_device(int(settings["gpu"]))
    else:
        settings["use_gpu"] = False
        settings["device"] = 'cpu'

    # if args.use_gpu and args.use_multi_gpu:
    #     args.devices = args.devices.replace(' ', '')
    #     device_ids = args.devices.split(',')
    #     args.device_ids = [int(id_) for id_ in device_ids]
    #     args.gpu = args.device_ids[0]
    #     settings.update(
    #         {
    #             "use_gpu": args.use_gpu,
    #             "devices": args.devices,
    #             "device_ids": args.device_ids,
    #             "gpu": args.gpu,
    #             "use_multi_gpu": args.use_multi_gpu
    #          }
    #     )
    # else:
    #     settings.update(
    #         {
    #             "use_gpu": args.use_gpu,
    #             "gpu": args.gpu,
    #             "use_multi_gpu": args.use_multi_gpu
    #          }
    #     )

    print("Experimental settings are: \n{}".format(str(settings)))
    return settings