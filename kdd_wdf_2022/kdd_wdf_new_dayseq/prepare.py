import torch

def prep_env():
    settings = {
        "path_to_test_x": "../data/sdwpf_baidukddcup2022_test_toy/test_x",
        "path_to_test_y": "../data/sdwpf_baidukddcup2022_test_toy/test_y",
        "data_path": "../data",
        "filename": "wtbdata_245days.csv",
        "task": "MS",
        "target": "Patv",
        "checkpoints": "checkpoints",
        "input_len": 288,
        "output_len": 288,
        "day_seq_len": 14,
        "start_col": 0,
        "var_len": 12,
        "var_out": 1,
        "day_len": 144,
        "train_days": 214,
        "val_days": 31,
        "test_days": 0,
        "total_days": 245,
        "model_name": 'Autoformer', # optinal: [ ChebyNet, logTransformer, SatGruAtt, Autoformer]
        "loss_name": 'FilterMSELoss',
        "real_value": False,
        "log_per_steps": 200,

        "num_layers": 2,
        "dropout": 0.3,
        "nhead": 4,
        "hidden_dims": 64,

        "num_workers": 5,
        "epoch": 30,
        "batch_size": 12,
        "patient": 8,
        "lr": 0.0001,
        "gpu": 0,
        "capacity": 134,
        "pred_file": "predict.py",
        "model_path": "checkpoints/Autoformer2/20220719154259",
        "framework": "pytorch",
        "is_debug": True
    }

    if torch.cuda.is_available():
        settings["use_gpu"] = True
        torch.cuda.set_device(int(settings["gpu"]))
    else:
        settings["use_gpu"] = False
        settings.device = 'cpu'

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