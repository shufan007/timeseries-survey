
# -*-Encoding: utf-8 -*-
################################################################################
#
# Copyright (c) 2022 Baidu.com, Inc. All Rights Reserved
#
################################################################################
"""
Description: Evaluate the performance
Authors: Lu,Xinjiang (luxinjiang@baidu.com)
Date:    2022/03/10
"""
import os
import sys
import time
import traceback
import numpy as np
import metrics, torch
from prepare import prep_env
from wpf_dataset import TestPGL4WPFDataset

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
            print("IMPORT ERROR: ", error)
            print("Load module [path %s] error: %s" % (path, traceback.format_exc()))
            traceback.print_exc()
            return None


def evaluate(settings):
    # type: (dict) -> float
    """
    Desc:
        Test the performance on the whole wind farm
    Args:
        settings:
    Returns:
        A score
    """
    start_forecast_time = time.time()
    forecast_module = Loader.load(settings["pred_file"])
    settings["path_to_test_x"] = './tests/test_x/0001in.csv'
    settings["path_to_test_y"] = './tests/test_y/0001out.csv'
    predictions = forecast_module.forecast(settings)
    test_y_ds = TestPGL4WPFDataset(filename=settings["path_to_test_y"])
    grounds = np.array(test_y_ds.get_data()[0, :, -settings["output_len"]:, :])
    raw_data_lst = test_y_ds.get_raw_df()

    end_forecast_time = time.time()
    if settings["is_debug"]:
        print("\nElapsed time for prediction is: {} secs\n".format(end_forecast_time - start_forecast_time))

    preds = np.array(predictions)
    gts = np.array(grounds)
    print(preds.shape, gts.shape)
    preds = np.sum(preds, axis=0)
    gts = np.sum(gts, axis=0)

    day_len = settings["day_len"]
    day_acc = []
    for idx in range(0, preds.shape[0]):
        acc = 1 - metrics.rmse(preds[idx, -day_len:], gts[idx, -day_len:]) / (settings["capacity"] * 1000)
        if acc != acc:
            continue
        day_acc.append(acc)
    day_acc = np.array(day_acc).mean()
    print('Accuracy:  {:.4f}%'.format(day_acc * 100))

    print(predictions.shape)
    overall_mae, overall_rmse = metrics.regressor_detailed_scores(predictions, grounds, raw_data_lst, settings["capacity"], settings["output_len"])

    print('\n \t RMSE: {}, MAE: {}'.format(overall_rmse, overall_mae))

    if settings["is_debug"]:
        end_test_time = time.time()
        print("\nElapsed time for evaluation is {} secs\n".format(end_test_time - end_forecast_time))

    total_score = (overall_mae + overall_rmse) / 2
    return total_score


if __name__ == "__main__":
    # Set up the initial environment
    # Current settings for the model
    envs = prep_env()
    score = evaluate(envs)
    print('\n --- Overall Score --- \n\t{}'.format(score))
