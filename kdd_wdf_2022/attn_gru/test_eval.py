# -*-Encoding: utf-8 -*-
################################################################################
#
# Copyright (c) 2022 Baidu.com, Inc. All Rights Reserved
#
################################################################################
"""
Description: Evaluate the performance
Authors: Lu,Xinjiang (luxinjiang@baidu.com)
Date:    2022/04/24
"""
import os
import sys
import time
import traceback
import tempfile
import zipfile
import numpy as np
import metrics
# -*-Encoding: utf-8 -*-
################################################################################
#
# Copyright (c) 2022 Baidu.com, Inc. All Rights Reserved
#
################################################################################
"""
Description: Wind turbine test set
Authors: Lu,Xinjiang (luxinjiang@baidu.com)
Date:    2022/04/20
"""
import numpy as np
import pandas as pd
from copy import deepcopy

# -*-Encoding: utf-8 -*-
################################################################################
#
# Copyright (c) 2022 Baidu.com, Inc. All Rights Reserved
#
################################################################################
"""
Description: Some useful metrics
Authors: Lu,Xinjiang (luxinjiang@baidu.com)
Date:    2022/03/10
"""
import os
import traceback
import numpy as np
import pandas as pd


class MetricsError(Exception):
    """
    Desc:
        Customize the Exception
    """
    def __init__(self, err_message):
        Exception.__init__(self, err_message)


def is_valid_prediction(prediction, idx=None):
    """
    Desc:
        Check if the prediction is valid
    Args:
        prediction:
        idx:
    Returns:
        A boolean value
    """
    try:
        if prediction.ndim > 1:
            nan_prediction = pd.isna(prediction).any(axis=1)
            if nan_prediction.any():
                if idx is None:
                    msg = "NaN in predicted values!"
                else:
                    msg = "NaN in predicted values ({}th prediction)!".format(idx)
                raise MetricsError(msg)
        if prediction.size == 0:
            if idx is None:
                msg = "Empty prediction!"
            else:
                msg = "Empty predicted values ({}th prediction)! ".format(idx)
            raise MetricsError(msg)
    except ValueError as e:
        traceback.print_exc()
        if idx is None:
            raise MetricsError("Value Error: {}. ".format(e))
        else:
            raise MetricsError("Value Error: {} in {}th prediction. ".format(e, idx))
    return True


def mae(pred, gt, run_id=0):
    """
    Desc:
        Mean Absolute Error
    Args:
        pred:
        gt: ground truth vector
        run_id:
    Returns:
        MAE value
    """
    _mae = -1
    if is_valid_prediction(pred, idx=run_id):
        if pred.shape != gt.shape:
            raise MetricsError("Different shapes between Prediction ({}) and Ground Truth ({}) "
                               "in {}th prediction! ".format(pred.shape, gt.shape, run_id))
        _mae = np.mean(np.abs(pred - gt))
    return _mae


def mse(pred, gt, run_id=0):
    """
    Desc:
        Mean Square Error
    Args:
        pred:
        gt: ground truth vector
        run_id:
    Returns:
        MSE value
    """
    _mse = -1
    if is_valid_prediction(pred, idx=run_id):
        if pred.shape != gt.shape:
            raise MetricsError("Different shapes between Prediction ({}) and Ground Truth ({}) "
                               "in {}th prediction! ".format(pred.shape, gt.shape, run_id))
        _mse = np.mean((pred - gt) ** 2)
    return _mse


def rmse(pred, gt, run_id=0):
    """
    Desc:
        Root Mean Square Error
    Args:
        pred:
        gt: ground truth vector
        run_id:
    Returns:
        RMSE value
    """
    _mse = mse(pred, gt, run_id=run_id)
    if _mse < 0:
        return -1
    return np.sqrt(_mse)


def regressor_scores(prediction, gt, idx=0):
    """
    Desc:
        Some common metrics for regression problems
    Args:
        prediction:
        gt: ground truth vector
        idx:
    Returns:
        A tuple of metrics
    """
    _mae = mae(prediction, gt, run_id=idx)
    _rmse = rmse(prediction, gt, run_id=idx)
    return _mae, _rmse


def turbine_scores(pred, gt, raw_data, examine_len, idx=0):
    """
    Desc:
        Calculate the MAE and RMSE of one turbine
    Args:
        pred: prediction for one turbine
        gt: ground truth
        raw_data: the DataFrame of one wind turbine
        examine_len:
        idx:
    Returns:
        The averaged MAE and RMSE
    """
    nan_cond = pd.isna(raw_data).any(axis=1)
    invalid_cond = (raw_data['Patv'] < 0) | \
                   ((raw_data['Patv'] == 0) & (raw_data['Wspd'] > 2.5)) | \
                   ((raw_data['Pab1'] > 89) | (raw_data['Pab2'] > 89) | (raw_data['Pab3'] > 89)) | \
                   ((raw_data['Wdir'] < -180) | (raw_data['Wdir'] > 180) | (raw_data['Ndir'] < -720) |
                    (raw_data['Ndir'] > 720))
    indices = np.where(~nan_cond & ~invalid_cond)
    prediction = pred[indices]
    targets = gt[indices]
    # NOTE: Before calculating the metrics, the unit of the outcome (e.g. predicted or true) power
    #       should be converted from Kilo Watt to Mega Watt first.
    _mae, _rmse = -1, -1
    if np.any(prediction) and np.any(targets):
        _mae, _rmse = regressor_scores(prediction[-examine_len:] / 1000, targets[-examine_len:] / 1000, idx=idx)
    return _mae, _rmse


def check_zero_prediction(prediction, idx=None):
    """
    Desc:
       If zero prediction, return -1
    Args:
        prediction:
        idx:
    Returns:
        An integer indicating status
    """
    if not np.any(prediction):
        if idx is None:
            msg = "Zero prediction!"
        else:
            msg = "Zero predicted values ({}th prediction)! ".format(idx)
        print(msg)
        return -1
    return 0


def is_zero_prediction(predictions, identifier, settings):
    """
    Desc:
        Check if zero prediction for all turbines in a wind farm
    Args:
        predictions:
        identifier:
        settings:
    Returns:
        False if otherwise
    """
    wind_farm_statuses = []
    for i in range(settings["capacity"]):
        prediction = predictions[i]
        status = check_zero_prediction(prediction, idx=identifier)
        wind_farm_statuses.append(status)
    statuses = np.array(wind_farm_statuses)
    non_zero_predictions = statuses[statuses == 0]
    non_zero_ratio = non_zero_predictions.size / settings["capacity"]
    if non_zero_ratio < settings["min_non_zero_ratio"]:
        msg = "{:.2f}% turbines with zero predicted values " \
              "(in {}th prediction)!".format((1 - non_zero_ratio) * 100, identifier)
        raise MetricsError(msg)
    return False


def check_identical_prediction(prediction, min_std=0.1, min_distinct_ratio=0.1, idx=None):
    """
    Desc:
        Check if the prediction is with the same values
    Args:
        prediction:
        min_std:
        min_distinct_ratio:
        idx:
    Returns:
        An integer indicating the status
    """
    try:
        if np.min(prediction) == np.max(prediction):
            if idx is None:
                msg = "All predicted values are the same as {:.4f}!".format(np.min(prediction))
            else:
                msg = "All predicted values are same as {:.4f} ({}th prediction)!".format(np.min(prediction), idx)
            print(msg)
            return -1
        if np.std(prediction) <= min_std:
            prediction = np.ravel(prediction)
            distinct_prediction = set(prediction)
            distinct_ratio = len(distinct_prediction) / np.size(prediction)
            samples = list(distinct_prediction)[:3]
            samples = ",".join("{:.5f}".format(s) for s in samples)
            if distinct_ratio < min_distinct_ratio:
                if idx is None:
                    msg = "{:.2f}% of predicted values are same! Some predicted values are: " \
                          "{},...".format((1 - distinct_ratio) * 100, samples)
                else:
                    msg = "{:.2f}% of predicted values are same ({}th run)! " \
                          "Some predicted values are:" \
                          "{},...".format((1 - distinct_ratio) * 100, idx, samples)
                print(msg)
                return -1
    except ValueError as e:
        traceback.print_exc()
        if idx is None:
            raise MetricsError("Value Error: {}. ".format(e))
        else:
            raise MetricsError("Value Error: {} in {}th prediction. ".format(e, idx))
    return 0


def is_identical_prediction(predictions, identifier, settings):
    """
    Desc:
        Check if the predicted values are identical for all turbines
    Args:
        predictions:
        identifier:
        settings:
    Returns:
        False
    """
    farm_check_statuses = []
    for i in range(settings["capacity"]):
        prediction = predictions[i]
        status = check_identical_prediction(prediction, min_distinct_ratio=settings["min_distinct_ratio"],
                                            idx=identifier)
        farm_check_statuses.append(status)
    statuses = np.array(farm_check_statuses)
    variational_predictions = statuses[statuses == 0]
    variation_ratio = variational_predictions.size / settings["capacity"]
    if variation_ratio < settings["min_distinct_ratio"]:
        msg = "{:.2f}% turbines with (almost) identical predicted values " \
              "({}th prediction)!".format((1 - variation_ratio) * 100, identifier)
        raise MetricsError(msg)
    return False


def regressor_detailed_scores(predictions, gts, raw_df_lst, settings):
    """
    Desc:
        Some common metrics
    Args:
        predictions:
        gts: ground truth vector
        raw_df_lst:
        settings:
    Returns:
        A tuple of metrics
    """
    path_to_test_x = settings["path_to_test_x"]
    tokens = os.path.split(path_to_test_x)
    identifier = int(tokens[-1][:-6]) - 1
    all_mae, all_rmse = [], []
    all_latest_mae, all_latest_rmse = [], []
    if not is_identical_prediction(predictions, identifier, settings) and \
            not is_zero_prediction(predictions, identifier, settings):
        pass
    for i in range(settings["capacity"]):
        prediction = predictions[i]
        if not is_valid_prediction(prediction, idx=identifier):
            continue
        gt = gts[i]
        raw_df = raw_df_lst[i]
        _mae, _rmse = turbine_scores(prediction, gt, raw_df, settings["output_len"], idx=identifier)
        if _mae != _mae or _rmse != _rmse:  # In case NaN is encountered
            continue
        if -1 == _mae or -1 == _rmse:       # In case the target is empty after filtering out the abnormal values
            continue
        all_mae.append(_mae)
        all_rmse.append(_rmse)
        latest_mae, latest_rmse = turbine_scores(prediction, gt, raw_df, settings["day_len"], idx=identifier)
        all_latest_mae.append(latest_mae)
        all_latest_rmse.append(latest_rmse)
    total_mae = np.array(all_mae).sum()
    total_rmse = np.array(all_rmse).sum()
    if total_mae < 0 or total_rmse < 0:
        raise MetricsError("{}th prediction: summed MAE ({:.2f}) or RMSE ({:.2f}) is negative, "
                           "which indicates too many invalid values "
                           "in the prediction! ".format(identifier, total_mae, total_rmse))
    if len(all_mae) == 0 or len(all_rmse) == 0 or total_mae == 0 or total_rmse == 0:
        raise MetricsError("No valid MAE or RMSE for "
                           "all of the turbines in {}th prediction! ".format(identifier))
    total_latest_mae = np.array(all_latest_mae).sum()
    total_latest_rmse = np.array(all_latest_rmse).sum()
    return total_mae, total_rmse, total_latest_mae, total_latest_rmse

# Turn off the SettingWithCopyWarning
pd.set_option('mode.chained_assignment', None)

class TestData(object):
    """
        Desc: Test Data
    """
    def __init__(self,
                 path_to_data,
                 task='MS',
                 target='Patv',
                 start_col=3,       # the start column index of the data one aims to utilize
                 farm_capacity=134
                 ):
        self.task = task
        self.target = target
        self.start_col = start_col
        self.data_path = path_to_data
        self.farm_capacity = farm_capacity
        self.df_raw = pd.read_csv(self.data_path)
        self.total_size = int(self.df_raw.shape[0] / self.farm_capacity)
        # Handling the missing values
        self.df_data = deepcopy(self.df_raw)
        self.df_data.replace(to_replace=np.nan, value=0, inplace=True)

    def get_turbine(self, tid):
        begin_pos = tid * self.total_size
        border1 = begin_pos
        border2 = begin_pos + self.total_size
        if self.task == 'MS':
            cols = self.df_data.columns[self.start_col:]
            data = self.df_data[cols]
        else:
            raise Exception("Unsupported task type ({})! ".format(self.task))
        seq = data.values[border1:border2]
        df = self.df_raw[border1:border2]
        return seq, df

    def get_all_turbines(self):
        seqs, dfs = [], []
        for i in range(self.farm_capacity):
            seq, df = self.get_turbine(i)
            seqs.append(seq)
            dfs.append(df)
        return seqs, dfs


class LoaderError(Exception):
    """
    Desc:
        Customize the Exception
    """

    def __init__(self, err_message):
        Exception.__init__(self, err_message)


class EvaluationError(Exception):
    """
    Desc:
        Customize the Exception for Evaluation
    """

    def __init__(self, err_message):
        Exception.__init__(self, err_message)


class Loader(object):
    """
    Desc:
        Dynamically Load a Module
    """

    def __init__(self):
        """
        """
        pass

    @staticmethod
    def load(path):
        """
        Args:
            path to the script
        """
        try:
            items = os.path.split(path)
            sys.path.append(os.path.join(*items[:-1]))
            ip_module = __import__(items[-1][:-3])
            return ip_module
        except Exception as error:
            traceback.print_exc()
            raise LoaderError("IMPORT ERROR: {}, load module [path: {}]!".format(error, path))


def performance(settings, idx, prediction, ground_truth, ground_truth_df):
    """
    Desc:
        Test the performance on the whole wind farm
    Args:
        settings:
        idx:
        prediction:
        ground_truth:
        ground_truth_df:
    Returns:
        MAE, RMSE and Accuracy
    """
    overall_mae, overall_rmse, _, overall_latest_rmse = \
        regressor_detailed_scores(prediction, ground_truth, ground_truth_df, settings)
    # A convenient customized relative metric can be adopted
    # to evaluate the 'accuracy'-like performance of developed model for Wind Power forecasting problem
    if overall_latest_rmse < 0:
        raise EvaluationError("The RMSE of the last 24 hours is negative ({}) in the {}-th prediction"
                              "".format(overall_latest_rmse, idx))
    acc = 1 - overall_latest_rmse / settings["capacity"]
    return overall_mae, overall_rmse, acc


TAR_DIR = os.path.normpath(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../test_y.zip'))
PRED_DIR = os.path.normpath(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../test_x.zip'))
DATA_DIR = os.path.normpath(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../data'))
REQUIRED_ENV_VARS = [
    "pred_file",
    "checkpoints",
    "start_col",
    "framework"
]
SUPPORTED_FRAMEWORKS = [
    "base", "paddlepaddle", "pytorch", "tensorflow"
]
NUM_MAX_RUNS = 1
MAX_TIMEOUT = 3600 * 10  # 10 hours
MIN_TIME = 3  # 3 secs
MIN_NOISE_LEVEL = 0.001  # 0.1 %


def exec_predict_and_test(envs, test_file, forecast_module, flag='predict'):
    """
    Desc:
        Do the prediction or get the ground truths
    Args:
        envs:
        test_file:
        forecast_module:
        flag:
    Returns:
        A result dict
    """
    print("test_file: ",test_file)
    with tempfile.TemporaryDirectory() as tmp_dir:
        with zipfile.ZipFile(test_file) as test_f:
            test_f.extractall(path=tmp_dir)
            items = os.listdir(tmp_dir)
            assert len(items) == 1, "More than one test files encountered in the tmp dir! "
            assert str(items[0]).endswith('.csv'), "Test data does not end with csv! "
            path_to_test_file = os.path.join(tmp_dir, items[0])
            print('path_to_test_file:  ', path_to_test_file)
            if 'predict' == flag:
                envs["path_to_test_x"] = path_to_test_file
                return {
                    "prediction": forecast_module.forecast(envs)
                }
            elif flag == 'test':
                test_data = TestData(path_to_data=path_to_test_file, start_col=envs["start_col"])
                turbines, raw_turbines = test_data.get_all_turbines()
                test_ys = []
                for turbine in turbines:
                    test_ys.append(turbine[:envs["output_len"], -envs["out_var"]:])
                return {
                    "ground_truth_y": np.array(test_ys), "ground_truth_df": raw_turbines
                }
            else:
                raise EvaluationError("Unsupported evaluation task (only 'predict' or 'test' is acceptable)! ")


def predict_and_test(envs, path_to_data, forecast_module, idx, flag='predict'):
    """
    Desc:
        Prediction or get the ground truths
    Args:
        envs:
        path_to_data:
        forecast_module:
        idx:
        flag:
    Returns:
        A dict
    """

    with tempfile.TemporaryDirectory() as tmp_dir:
        print("_____tmp_dir:", tmp_dir)
        # with zipfile.ZipFile(path_to_data) as test_f:
        #     test_f.extractall(path=tmp_dir)
        #
        #     items = os.listdir(tmp_dir)
        #     assert 1 == len(items), "More than one items in {}".format(tmp_dir)
        #     print("_____1111tmp_dir:", tmp_dir)
        #     tmp_dir = os.path.join(tmp_dir, items[0])
        #     print("_____222tmp_dir:", tmp_dir)
        #     items = os.listdir(tmp_dir)
        #     files = sorted(items)
        #     path_to_test_file = os.path.join(tmp_dir, files[idx])
        return exec_predict_and_test(envs, path_to_data, forecast_module, flag)


def evaluate(path_to_src_dir):
    """
    Desc:
        The main entrance for the evaluation
    args:
        path_to_src_dir:
    Returns:
        A dict indicating performance
    """
    print("path_to_src_dir:  ", path_to_src_dir)

    begin_time = time.time()
    left_time = MAX_TIMEOUT
    start_test_time = begin_time
    # Set up the initial environment
    path_to_prep_script = os.path.join(path_to_src_dir, "prepare.py")
    if not os.path.exists(path_to_prep_script):
        raise EvaluationError("The preparation script, i.e. 'prepare.py', does NOT exist! ")
    prep_module = Loader.load(path_to_prep_script)
    envs = prep_module.prep_env()

    eval_path = os.path.normpath(os.path.dirname(os.path.realpath(__file__)))
    eval_dir = os.path.split(eval_path)[-1]
    print("eval_dir: ", eval_dir)
    # if envs["framework"] not in eval_dir or eval_dir not in envs["framework"]:
    #     raise Exception("The claimed framework ({}) is NOT the framework "
    #                     "you used ({})!".format(envs["framework"], eval_dir))

    for req_key in REQUIRED_ENV_VARS:
        if req_key not in envs:
            raise EvaluationError("Key error: '{}'. The variable {} "
                                  "is missing in the prepared experimental settings! ".format(req_key, req_key))

    if "is_debug" not in envs:
        envs["is_debug"] = False

    if envs["framework"] not in SUPPORTED_FRAMEWORKS:
        raise EvaluationError("Unsupported machine learning framework: {}. "
                              "The supported frameworks are 'base', 'paddlepaddle', 'pytorch', "
                              "and 'tensorflow'".format(envs["framework"]))

    envs["data_path"] = DATA_DIR
    envs["filename"] = "wtbdata_245days.csv"
    envs["location_filename"] = "sdwpf_baidukddcup2022_turb_location.csv"
    envs["day_len"] = 144
    envs["capacity"] = 134
    envs["output_len"] = 288
    envs["out_var"] = 1
    envs["pred_file"] = os.path.join(path_to_src_dir, envs["pred_file"])
    envs["checkpoints"] = os.path.join(path_to_src_dir, envs["checkpoints"])
    envs["min_distinct_ratio"] = 0.1
    envs["min_non_zero_ratio"] = 0.5

    if envs["is_debug"]:
        end_load_test_set_time = time.time()
        print("Load test_set (test_ys) in {} secs".format(end_load_test_set_time - start_test_time))
        start_test_time = end_load_test_set_time

    maes, rmses, accuracies = [], [], []
    forecast_module = Loader.load(envs["pred_file"])

    start_forecast_time = start_test_time
    end_forecast_time = start_forecast_time
    PRED_DIR = "./data/sdwpf_baidukddcup2022_test_toy/test_x/test_x.zip"
    TAR_DIR = "./data/sdwpf_baidukddcup2022_test_toy/test_y/test_y.zip"
    for i in range(NUM_MAX_RUNS):

        pred_res = predict_and_test(envs, PRED_DIR, forecast_module, i, flag='predict')
        prediction = pred_res["prediction"]
        print('prediction->: ', prediction.shape)


        gt_res = predict_and_test(envs, TAR_DIR, forecast_module, i, flag='test')
        gt_ys = gt_res["ground_truth_y"]
        gt_turbines = gt_res["ground_truth_df"]  # 134, 288, 1
        print('true: ',gt_ys.shape)

        if envs["is_debug"]:
            end_forecast_time = time.time()
            print("\nElapsed time for {}-th prediction is: "
                  "{} secs \n".format(i, end_forecast_time - start_forecast_time))
            start_forecast_time = end_forecast_time

        tmp_mae, tmp_rmse, tmp_acc = performance(envs, i, prediction, gt_ys, gt_turbines)
        #
        if tmp_acc <= 0:
            # Accuracy is lower than Zero, which means that the RMSE of this prediction is too large,
            # which also indicates that the performance is probably poor and not robust
            print('\n\tThe {}-th prediction -- '
                  'RMSE: {}, MAE: {}, and Accuracy: {}'.format(i, tmp_mae, tmp_rmse, tmp_acc))
            raise EvaluationError("Accuracy ({:.3f}) is lower than Zero, which means that "
                                  "the RMSE (in latest 24 hours) of the {}th prediction "
                                  "is too large!".format(tmp_acc, i))
        else:
            print('\n\tThe {}-th prediction -- '
                  'RMSE: {}, MAE: {}, Score: {}, '
                  'and Accuracy: {:.4f}%'.format(i, tmp_rmse, tmp_mae, (tmp_rmse + tmp_mae) / 2, tmp_acc * 100))
        maes.append(tmp_mae)
        rmses.append(tmp_rmse)
        accuracies.append(tmp_acc)

        cost_time = time.time() - begin_time
        left_time -= cost_time
        cnt_left_runs = NUM_MAX_RUNS - (i + 1)
        if i > 1 and left_time < MIN_TIME * (cnt_left_runs + 1):
            # After three runs, we will check how much time remain for your code:
            raise EvaluationError("TIMEOUT! "
                                  "Based on current running time analysis, it's not gonna happen that "
                                  "your model can run {} predictions in {:.2f} secs! ".format(cnt_left_runs, left_time))
        begin_time = time.time()

    avg_mae, avg_rmse, total_score = -1, -1, 65535
    # TODO: more invalid predictions should be taken into account ...
    if len(maes) == NUM_MAX_RUNS:
        if np.std(np.array(rmses)) < MIN_NOISE_LEVEL or np.std(np.array(maes)) < MIN_NOISE_LEVEL \
                or np.std(np.array(accuracies)) < MIN_NOISE_LEVEL:
            # Basically, this is not going to happen most of the time, if so, something went wrong
            raise EvaluationError("Std of rmses ({:.4f}) or std of maes ({:.4f}) or std of accs ({:.4f}) "
                                  "is too small! ".format(np.std(np.array(rmses)), np.std(np.array(maes)),
                                                          np.std(np.array(accuracies))))
        avg_mae = np.array(maes).mean()
        avg_rmse = np.array(rmses).mean()
        total_score = (avg_mae + avg_rmse) / 2
        print('\n --- Final MAE: {}, RMSE: {} ---'.format(avg_mae, avg_rmse))
        print('--- Final Score --- \n\t{}'.format(total_score))

    if envs["is_debug"]:
        print("\nElapsed time for prediction is {} secs\n".format(end_forecast_time - start_test_time))
        end_test_time = time.time()
        print("\nTotal time for evaluation is {} secs\n".format(end_test_time - start_test_time))

    if total_score > 0:
        return {
            "score": -1. * total_score, "ML-framework": envs["framework"]
        }
    else:
        raise EvaluationError("Invalid score ({}) returned. ".format(total_score))


def eval(submit_file):
    """
    Desc:
        The interface for the system call
    Args:
        submit_file:
    Returns:
        A dict indicating the score and the machine learning framework
    """
    # Check suffix of the submitted file
    if not submit_file.endswith('.zip'):
        raise Exception("Submitted file does not end with zip ！")

    try:
        with tempfile.TemporaryDirectory() as tmp_dir:
            print("tmp_dir: ", tmp_dir)
            # Extract files
            # Handle exceptions
            with zipfile.ZipFile(submit_file) as src_f:
                # for filename in src_f.namelist():
                #     print('File:', filename)
                src_f.extractall(path=tmp_dir)
                items = os.listdir(tmp_dir)
                print(items)
                if 1 == len(items):
                    tmp_dir = os.path.join(tmp_dir, items[0])
                    items = os.listdir(tmp_dir)
                if 0 == len(items):
                    raise Exception("Zip file is empty! ")
                return evaluate(tmp_dir)
    except Exception as error:
        submit_file = os.path.split(submit_file)[-1]
        msg = "Err: {}! ({})".format(error, submit_file)
        raise Exception(msg)


if __name__ == '__main__':
    eval('../kdd_wdf_test5.zip')