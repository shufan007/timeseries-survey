import time
import os
import sys
import numpy as np
import flaml
from ray import tune

import torch.nn.functional as F
from sklearn.metrics import precision_score, recall_score, confusion_matrix,  roc_curve, precision_recall_curve, accuracy_score, roc_auc_score, f1_score, mean_absolute_error, mean_squared_error

import torch
from tsai.all import *

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# print("BASE_DIR: ", BASE_DIR)
sys.path.append(BASE_DIR)

computer_setup()

train_file_num = 1
test_file_num = 1

def mape2(y_pred, y_true):
    #y_true = y_true.cpu().data.numpy()
    #y_pred = y_pred.cpu().data.numpy()

    #return  mape(y_true, decode_label(y_pred))
    return  mape(y_true, y_pred)



def get_battery_data(file_name, x_key='seq_d', input_len=12, label_index=-1):
    """
    # Output data format: [# samples x # variables x sequence length] 
    
    """
    samples = torch.load(file_name)
    #geo = self.samples[0]
    #version = self.samples[1]
    raw_label_idx = 2
    raw_label = samples[raw_label_idx][:, label_index]
    y = raw_label
    
    feature_names = ['flat_seq_c', 'flat_seq_d', 'flat_d', 'hh_seq_c', 'hh_seq_d', 'hh_seq_label', 'seq_c', 'seq_d', 'seq_label']
    
    x_index = raw_label_idx + 1 + feature_names.index(x_key)
    
    if x_index >= input_len-3:
        X = samples[x_index][:,:,-1,:].transpose(1,2) 
    elif x_index >= input_len-6 and x_index < input_len-3:
        X = samples[x_index].transpose(1,2) 
    else:
        X = samples[x_index]
    
    # Output data format: [# samples x # variables x sequence length] 
    return  X,y


def train_learner(config, data_path, valid_size=0.2, isTune=True, **kwargs):
    
    print("mpdel train ...")
    
    start0 = time.time()
    
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print("device:", device)

    if os.path.isfile(data_path):
        file_list = [data_path]
    elif os.path.isdir(data_path):
        file_list = [os.path.join(data_path, file) for file in os.listdir(data_path)]
    
    bs = [2 ** config["batch_size"], 128]
    n_epochs = int(round(config["n_epochs"]))
    
    
    #tfms  = [None, [None]]
    tfms = [None, TSRegression()]
    #tfms = [None, TSForecasting()]
    
    file_num = train_file_num
    file_cnt = 0
    learn = None
    for file in file_list[0:file_num]:
        file_cnt += 1
        start = time.time()
        
        X,y = get_battery_data(file, x_key='seq_d', input_len=12, label_index=-1)
        
        print('\n data load Elapsed time:', time.time() - start) 
        
        """
        # train valid split
        """

        n_sample = 1000
        X, y = X[0:n_sample], y[0:n_sample]
        
        #splits = get_splits(y, valid_size=valid_size, stratify=True, random_state=23, shuffle=True)
        splits = get_splits(y, valid_size=valid_size, stratify=False, shuffle=False)    
        
        """
        # combine_split_data
        valid_size = 0.2
        n_total = X_total.shape[0]
        n_sample = 1000

        n_valid = int(n_sample* valid_size)
        n_train = n_sample - n_valid

        X_train, X_valid = X_total[0:n_train], X_total[n_train:n_total]
        y_train, y_valid = y_total[0:n_train], y_total[n_train:n_total]

        X, y, splits = combine_split_data([X_train, X_valid], [y_train, y_valid])

        """        

        """
        prepare data set, TSDataLoader
        """
        
        dsets = TSDatasets(X, y, tfms=tfms, splits=splits, inplace=True)
        dls   = TSDataLoaders.from_dsets(dsets.train, dsets.valid, bs=bs, device=device, batch_tfms=TSStandardize())
        #dls.show_batch()

        if not learn:
            #model = InceptionTime(dls.vars, dls.c)
            model = TSTPlus(dls.vars, dls.c, dls.len, 
                            n_layers = config["n_layers"],
                            n_heads = 2**config["n_heads"],
                            dropout = config["dropout"],
                            fc_dropout = config["fc_dropout"])
            #model = TSTPlus(dls.vars, dls.c, dls.len, dropout=.3)
            #model = TST(dls.vars, dls.c, dls.len, dropout=.1, fc_dropout=.8)
            #model.to(device)            
        else:
            model = learn.model
        
        loss_func = MSELossFlat()
        #loss_func = TweedieLoss()
        
        # 如果使用自定义metric, 无法序列化 Can't pickle <function mape2
        #learn = Learner(dls, model, loss_func=loss_func, metrics=[mae, mape, mape2]) 
        learn = Learner(dls, model, loss_func=loss_func, metrics=[loss_func, mae, mape])

        start = time.time()
        
        learn.fit_one_cycle(n_epochs, lr_max=config["lr_max"])
        
        print('\n Elapsed time:', time.time() - start)    

        """
        train by TSRegressor

        batch_tfms = TSStandardize(by_sample=True)
        #reg = TSRegressor(X, y, splits=splits, path='models', arch=TSTPlus, batch_tfms=batch_tfms, metrics=rmse, cbs=ShowGraph(), verbose=True)
        reg = TSRegressor(X, y, splits=splits, path='models', arch=TSTPlus, batch_tfms=batch_tfms, loss_func=loss_func, metrics=mape, cbs=ShowGraph(), verbose=True)

        reg.fit_one_cycle(5, 3e-4)
        #reg.export("reg.pkl")
        """

        """
        # train by TSForecaster

        from tsai.all import *
        #ts = get_forecasting_time_series("Sunspots").values
        #X, y = SlidingWindow(60, horizon=1)(ts)

        #X1, y1 = SlidingWindow(49, horizon=1)(y)
        X1, y1 = SlidingWindow(4, horizon=1)(y)

        splits = TimeSplitter(valid_size=0.2)(y1) 
        batch_tfms = TSStandardize()
        fcst = TSForecaster(X1, y1, splits=splits, path='models', batch_tfms=batch_tfms, bs=512, arch=TSTPlus, metrics=mae, cbs=ShowGraph())
        fcst.fit_one_cycle(10, 1e-3)
        fcst.export("fcst.pkl")
        """
        
        if not isTune:
            path = os.path.join(checkpoint_dir, "checkpoint")
            torch.save(
                (learn.model.state_dict(), learn.opt_func.state_dict()), path)
            continue

        # Here we save a checkpoint. It is automatically registered with
        # Ray Tune and will potentially be passed as the `checkpoint_dir`
        # parameter in future iterations.
        #with tune.checkpoint_dir(step=epoch) as checkpoint_dir:
        #with distributed_checkpoint_dir(step=epoch) as checkpoint_dir:
        with tune.checkpoint_dir(step=file_cnt) as checkpoint_dir:
            path = os.path.join(checkpoint_dir, "checkpoint")
            #torch.save((learn.model.state_dict(), learn.opt_func.state_dict()), path)
            learn.export(path)
        
        """
        tune.report(vars()[learn.metrics[0].name]=learn.metrics[0].value, 
                    vars()[learn.metrics[1].name]=learn.metrics[1].value,
                   vars()[learn.metrics[2].name]=learn.metrics[2].value)        
        """
        
        tune.report(mae=learn.metrics[0].value, 
                    mape=learn.metrics[1].value)
        
        print("learn.metrics[0].name: %.3f, learn.metrics[1].name: %.3f" % 
              (learn.metrics[0].value, learn.metrics[1].value)) 

    print('\n Total time:', time.time() - start0) 
    
    #return learn
    return {"mae":learn.metrics[0].value, "mape":learn.metrics[1].value}


def model_eval(learner, data_path, **kwargs):
    
    print("model eval...")
    start0 = time.time()

    if os.path.isfile(data_path):
        file_list = [data_path]
    elif os.path.isdir(data_path):
        file_list = [os.path.join(data_path, file) for file in os.listdir(data_path)]
    
    file_num = test_file_num

    metric_sum = 0
    for file_name in file_list[0:file_num]:
        
        start = time.time()
        
        X,y = get_battery_data(file_name, x_key='seq_d', input_len=12, label_index=-1)
        
        raw_preds, target, preds = learner.get_X_preds(X, y)
        
        metric_sum += mape(target, torch.tensor(preds))

    print("mape on valid data:", metric_sum/len(file_list[0:file_num]))
    print('\nElapsed time:', time.time() - start0) 
    


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
    train_data = args["train_data"]
    test_data = args["test_data"]
    time_budget_s = args["time_budget"]
    num_samples = args["num_samples"]
    output_path="./logs"
    if args["output_path"]:
        output_path = args["output_path"]
    resources_per_trial = args["resources_per_trial"]

    """
    Search space
    """
    max_num_epoch = 4
    config = {
        "n_layers": tune.randint(2, 4),
        #"d_model": tune.randint(5, 7),  # log transformed with base 2
        "n_heads": tune.randint(2, 4),  # log transformed with base 2
        #"d_ff": tune.randint(6, 8),  # log transformed with base 2
        "dropout": tune.loguniform(1e-1, 3e-1),
        "fc_dropout": tune.loguniform(6e-1, 9e-1),
        "lr_max": tune.loguniform(1e-4, 1e-3),
        "n_epochs": tune.loguniform(1, max_num_epoch),
        "batch_size": tune.randint(4, 7)  # log transformed with base 2
    }

    np.random.seed(7654321)

    start_time = time.time()
    result = flaml.tune.run(
        tune.with_parameters(train_learner, data_path=train_data),
        config=config,
        metric="mae",
        mode="min",
        low_cost_partial_config={"n_epochs": 1},
        max_resource=max_num_epoch,
        min_resource=1,
        scheduler="asha",  # need to use tune.report to report intermediate results in train_cifar
        resources_per_trial=resources_per_trial,
        local_dir=output_path,
        num_samples=num_samples,
        time_budget_s=time_budget_s,
        use_ray=True)

    print(f"#trials={len(result.trials)}")
    print(f"time={time.time() - start_time}")
    try:
        #best_trial = result.get_best_trial("loss", "min", "all")

        best_trial = result.get_best_trial("mae", "min", "all")
        print("Best trial config: {}".format(best_trial.config))
        print("Best trial final validation mae: {}".format(
            best_trial.metric_analysis["mae"]["min"]))
        print("Best trial final validation metric: mae:{}, mape:{}".format(
            best_trial.metric_analysis["mae"]["min"], best_trial.metric_analysis["mape"]["min"]))
        
    except Exception as e:
        print("except:", e)
        

    """
    test 
    """
    print("test on best_trained_model ...")
    
    
    if test_data and best_trial:
        
        """
        X,y = get_battery_data(file, x_key='seq_d', input_len=12, label_index=-1)
        
        # load model
        best_trained_model = TSTPlus(X.shape[1], 1, X.shape[2], 
                            n_layers = best_trial.config["n_layers"],
                           n_heads = best_trial.config["n_heads"])
        
        device = "cpu"
        if torch.cuda.is_available():
            device = "cuda:0"
            if torch.cuda.device_count() > 1:
                best_trained_model = nn.DataParallel(best_trained_model)
                
        print("device:", device)
        best_trained_model.to(device) 
        
        checkpoint_path = os.path.join(best_trial.checkpoint.value, "checkpoint")
        
        model_state, optimizer_state = torch.load(checkpoint_path)
        best_trained_model.load_state_dict(model_state, strict=False)
        test_acc = _test_accuracy(best_trained_model, testset=testset, batch_size=128, device=device)
        print("Best trial test set accuracy: {}".format(test_acc))
        """

        checkpoint_path = os.path.join(best_trial.checkpoint.value, "checkpoint")
        
        from tsai.inference import load_learner
        
        learn = load_learner(checkpoint_path)

        model_eval(learn, test_data)
        


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
    parser.add_argument('--remote', type=str, default='False', help="True, submit job outside the Ray cluster, False, submit inside the ray cluster")
    parser.add_argument('--train_data', type=str, help='path of train data')
    parser.add_argument('--test_data', type=str, help='[optional] path of test data')
    parser.add_argument('--output_path', type=str, help='[optional] output path for model to save')
    parser.add_argument('--time_budget', type=int, default=60,
                        help='constrains the wall-clock time (seconds) used by the AutoML process. default: 60')
    parser.add_argument('--resources_per_trial', type=str,
                        help='resources dict like: {"cpu": 1, "gpu": gpus_per_trial}, number of cpus, gpus for each trial; 0.5 means two training jobs can share one gpu')
    parser.add_argument('--num_samples', type=int, help='maximal number of trials')

    args = parser.parse_args(argv)
    args = vars(args)
    if args['node_list']:
        args['node_list'] = eval(args['node_list'])
    elif args['host_file']:
        with open(args['host_file'], "r") as f:
            lines = f.readlines()
            args['node_list'] = [line.split(' ')[0].strip() for line in lines]
            print("node_list in host_file: ", args['node_list'])
    if args['remote']:
        assert (args['remote'].lower() in ['true', 'false', '0', '1'])
        if (args['remote'].lower() == 'true') or (args['remote'] == '1'):
            args['remote'] = True
        else:
            args['remote'] = False

    if args['resources_per_trial']:
        args['resources_per_trial'] = eval(args['resources_per_trial'])
        for k, v in args['resources_per_trial'].items():
            args['resources_per_trial'][k] = float(v)

    return args
            

def task_manager(argv, usage=None, use_ray=True):
    """
    job manager: job args parser, job submit
    :param argv:
    :param usage:
    :return:
    """
    args = task_arg_parser(argv, usage=usage)
    print("parse args: '{}'".format(args))

    if not use_ray:
        automl_tune(args)
        return
        
    try:
        # Startup Ray
        if args['node_list'] and len(args['node_list']) > 1:
            from autotabular import RayCluster
            ray_op = RayCluster()
            node_list, remote = ray_op.startup(node_list=args['node_list'], remote=args['remote'])
            args['node_list'] = node_list
            args['remote'] = remote

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
        python3 ./examples/tune_battery_cost_tsai.py --node_list '["10.186.2.241"]' --train_data /nfs/volume-807-1/houmin/hdh/15_train --test_data /nfs/volume-807-1/houmin/hdh/15_test --output_path /nfs/volume-807-1/fanshuangxi/test.outputs --time_budget 300 --resources_per_trial '{"cpu":1,"gpu":1}' --num_samples 500
            or
        sh run.sh ./examples/tune_battery_cost_tsai.py --host_file /etc/HOROVOD_HOSTFILE --train_data /nfs/volume-807-1/houmin/hdh/15_train --test_data /nfs/volume-807-1/houmin/hdh/15_test --output_path /nfs/volume-807-1/fanshuangxi/test.outputs --time_budget 300 --resources_per_trial '{"cpu":1,"gpu":1}' --num_samples 20
            '''
    print("argv:", argv)
    # task_manager(argv, usage)
    task_manager(argv, usage, use_ray=False)


if __name__ == "__main__":
    main(sys.argv[1:])
    