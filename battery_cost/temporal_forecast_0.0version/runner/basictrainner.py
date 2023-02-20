import numpy as np
import torch, time, copy, os, sys
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
file_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(file_dir+'/')
from lib.dataloader import SegData, SegData_orignal, get_feature
from lib.metric import All_Metrics, TweedieLoss, WMAPE, MAE, MAPE

class Trainer(object):
    def __init__(self, config, model, model_name, loss, optimizer, dataset, current, day, logger, lr_scheduler=None):
        super(Trainer, self).__init__()
        self.config = config
        self.model = model
        self.model_name = model_name
        self.loss = loss
        self.optimizer = optimizer
        self.train_dir = config.data.train_dir
        self.test_dir = config.data.test_dir
        self.lr_scheduler = lr_scheduler
        self.train_len = config.data.train_len
        self.test_len = config.data.test_len
        self.val_len = 6
        self.best_path = self.config.log.log_dir
        self.loss_figure_path = os.path.join(self.config.log.log_dir, 'trainning_loss.png')
        self.dataset = dataset
        self.current = bool(current)
        self.day = bool(day)
        self.log_dir = config.log.log_dir
        self.logger = logger
        self.logger.info('Experiment log path in: {}'.format(config.log.log_dir))
        if lr_scheduler:
            self.logger.info('Applying learning rate decay.')
        self.logger.info("Argument: %r", config)
        self.orignal = False
        if model_name == 'gru_att':
            self.orignal = True


    def val_epoch(self, epoch):
        self.model.eval()
        Tweedie = TweedieLoss(p=1.4)
        self.logger.info("the validation file is: " + os.path.join(self.test_dir, '/part-%s' % self.val_len))
        if self.orignal:
            val_data = SegData_orignal(self.test_dir + '/part-%s' % self.val_len, 12, self.config.data.label_index)
        else:
            val_data = SegData(self.test_dir + '/part-%s' % self.val_len, 12, label_index=self.config.data.label_index, current=self.current, day=self.day)
        val_dataloader = DataLoader(val_data, batch_size=self.config.train.batch_size, shuffle=True, drop_last=True)
        del val_data
        total_val_loss = 0
        y_pred, y_true = [], []
        with torch.no_grad():
            for idx, batch in enumerate(val_dataloader):
                if self.orignal:
                    input, y = batch
                else:
                    cur_features, h17_seq_features, day_features, y = batch
                del batch
                if self.config.data.label_index == 2:
                    y /= 1000
                y = y.reshape(-1, 1).float().cuda()
                if self.orignal:
                    flat_seq_c, flat_seq_d, flat_d, hh_seq_c, hh_seq_d, hh_seq_label, seq_c, seq_d, seq_label = get_feature(
                        f=input, use_gpu=1, feature_types=9)
                    pred = self.model(flat_seq_c, flat_seq_d, flat_d, hh_seq_c, hh_seq_d, hh_seq_label,
                                      seq_c[:, :, -1, :],
                                      seq_d[:, :, -1, :], seq_label[:, :, -1, :])
                elif self.model_name in ['Informer', 'logspare_transformer', 'TFT']:
                    pred = self.model(h17_seq_features, cur_features, day_features)
                else:
                    pred = self.model(h17_seq_features)


                if self.config.train.loss == 'tweedie':
                    pred = torch.exp(pred)

                loss = (self.loss(pred, y)).mean()
                if not torch.isnan(loss):
                    total_val_loss += loss.item()

                y_pred.append(pred)
                y_true.append(y)
        y_true = torch.cat(y_true, dim=0)
        y_pred = torch.cat(y_pred, dim=0)

        val_loss = total_val_loss / len(val_dataloader)
        self.logger.info('**********val Epoch {}: averaged Loss ({}): {:.6f}'.format(epoch, self.config.train.loss, val_loss))

        wmape, mape, tweedie = WMAPE(y_true, y_pred).item(), MAPE(y_true, y_pred).item(), (Tweedie(y_pred, y_true)).mean().item()
        self.logger.info("**********val Average, wmape: {:.4f}, mape: {:.4f}, tweedie: {:.4f}".format(wmape, mape, tweedie))
        mae_0, mae_0_10, mae_10_20, mae_20 = MAE(y_true, y_pred)
        self.logger.info(
            "**********val Average, mae_0: {:.4f}, mae_0_10: {:.4f}, mae_10_20: {:.4f}, mae_20: {:.4f}".format(mae_0.item(),
                                                                                                         mae_0_10.item(),
                                                                                                         mae_10_20.item(),
                                                                                                         mae_20.item()))

        mae_thresh = None
        mae, rmse, rrse = All_Metrics(y_pred, y_true, mae_thresh)
        self.logger.info("**********val Average, MAE: {:.4f}, RMSE: {:.4f}, RRSE: {:.4f}".format(mae, rmse, rrse))

        return val_loss

    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        ll = 0
        for i in range(int(self.train_len)):
            if self.orignal:
                train_data = SegData_orignal(self.train_dir + '/part-%s' % i, 12, self.config.data.label_index)
            else:
                train_data = SegData(self.train_dir + '/part-%s' % i, 12, label_index=self.config.data.label_index, current=self.current, day=self.day)
            train_loader = DataLoader(train_data, batch_size=self.config.train.batch_size, shuffle=True, drop_last=True)
            del train_data
            train_loss = 0
            for batch_idx, batch in enumerate(train_loader):
                self.optimizer.zero_grad()
                if self.orignal:
                    input, y = batch
                else:
                    cur_features, h17_seq_features, day_features, y = batch
                del batch
                if self.config.data.label_index == 2:
                    y /= 1000
                y = y.reshape(-1, 1).float().cuda()
                if self.orignal:
                    flat_seq_c, flat_seq_d, flat_d, hh_seq_c, hh_seq_d, hh_seq_label, seq_c, seq_d, seq_label = get_feature(
                        f=input, use_gpu=1, feature_types=9)
                    pred = self.model(flat_seq_c, flat_seq_d, flat_d, hh_seq_c, hh_seq_d, hh_seq_label,
                                      seq_c[:, :, -1, :],
                                      seq_d[:, :, -1, :], seq_label[:, :, -1, :])
                elif self.model_name in ['Informer', 'logspare_transformer', 'TFT']:
                    pred = self.model(h17_seq_features, cur_features, day_features)
                else:
                    pred = self.model(h17_seq_features)

                if self.config.train.loss in ['tweedie']:
                    pred = torch.exp(pred)
                loss = (self.loss(pred, y)).mean()
                train_loss += loss.item()
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                # log information
                if batch_idx % self.config.log.log_step == 0:
                    self.logger.info('Train Epoch {}: the part {}: {}/{} Loss: {:.6f}'.format(
                        epoch, i, batch_idx, len(train_loader), loss.item()))
            ll += len(train_loader)
            del train_loader
            #self.logger.info('Finished training the part %s of the train-data-set' % i)
            total_loss += train_loss
        train_epoch_loss = total_loss / ll
        self.logger.info('**********Train Epoch {}: averaged Loss ({}): {:.6f}'.format(epoch, self.config.train.loss, train_epoch_loss))
        # learning rate decay
        if self.config.train.lr_decay:
            self.lr_scheduler.step()
        return train_epoch_loss

    def train(self):
        best_model = None
        best_loss = float('inf')
        not_improved_count = 0
        lossdf = pd.DataFrame()
        epoch_list = []
        train_loss_list = []
        val_loss_list = []
        start_time = time.time()
        torch.cuda.empty_cache()
        for epoch in range(1, self.config.train.epochs+1):
            torch.cuda.empty_cache()
            epoch_list += [epoch]

            train_epoch_loss = self.train_epoch(epoch )
            val_epoch_loss = self.val_epoch(epoch)

            train_loss_list.append(train_epoch_loss)
            val_loss_list.append(val_epoch_loss)

            if train_epoch_loss > 1e6:
                self.logger.warning('Gradient explosion detected. Ending...')
                break
            if val_epoch_loss < best_loss:
                best_loss = val_epoch_loss
                not_improved_count = 0
                best_state = True
            else:
                not_improved_count += 1
                best_state = False
            # early stop
            if self.config.train.early_stop:
                if not_improved_count == self.config.train.early_stop_patience:
                    self.logger.info("Validation performance didn\'t improve for {} epochs. " 
                                     "Training stops.".format(self.config.train.early_stop_patience))
                    break
            # save the best state
            if best_state == True:
                self.logger.info('*********************************Current best model saved!')
                best_model = copy.deepcopy(self.model.state_dict())
                torch.save(best_model, os.path.join(self.best_path, 'best_model.model'))
                self.logger.info("Saving current best model to " + os.path.join(self.best_path, 'best_model.model'))

        training_time = time.time() - start_time
        self.logger.info("Total training time: {:.4f}min, best loss: {:.6f}".format((training_time / 60), best_loss))
        # save the best model to file
        if not self.config.base.debug:
            torch.save(best_model, os.path.join(self.best_path, 'best_model.pth'))
            self.logger.info("Saving current best model to " + os.path.join(self.best_path, 'best_model.pth'))
        # plot loss
        lossdf['epoch'] = epoch_list
        lossdf['train'] = train_loss_list
        lossdf['val'] = val_loss_list
        line1, = plt.plot(lossdf['epoch'], lossdf['train'])
        line2, = plt.plot(lossdf['epoch'], lossdf['val'])
        plt.title('the loss fig of training and validation')
        plt.legend(handles=[line1, line2], labels=["train", "val"], loc="upper left", fontsize=8)
        plt.xlabel("epoch")
        plt.ylabel("loss")
        plt.savefig(self.loss_figure_path)
        #plt.show()
        # test
        self.model.load_state_dict(best_model)
        self.test(self.config, self.model, self.model_name, self.test_dir, self.logger, self.dataset,  self.current, self.day, self.log_dir)

    @staticmethod
    def test(config, model, model_name, test_dir, logger, dataset=None, current=False, day=False, log_dir='./'):
        model.eval()
        y_pred, y_true = [], []
        orignal = False
        if model_name == 'gru_att':
            orignal = True
        with torch.no_grad():
            #for i in range(int(config.data.test_len)):
            for i in range(6, 7):
                try:
                    if orignal:
                        test_data = SegData_orignal(test_dir + '/part-%s' % i, 12, config.data.label_index)
                    else:
                        test_data = SegData(test_dir + '/part-%s' % i, 12, label_index=config.data.label_index, current=current, day=day)
                    logger.info("the test file is: " + os.path.join(test_dir, '/part-%s' % i))
                except:
                    continue
                test_loader = DataLoader(test_data, batch_size=int(config.train.batch_size), shuffle=True, drop_last=True)
                for idx, batch in enumerate(test_loader):
                    if orignal:
                        input, y = batch
                    else:
                        cur_features, h17_seq_features, day_features, y = batch
                    del batch
                    if config.data.label_index == 2:
                        y /= 1000
                    y = y.reshape(-1, 1).float().cuda()
                    if orignal:
                        flat_seq_c, flat_seq_d, flat_d, hh_seq_c, hh_seq_d, hh_seq_label, seq_c, seq_d, seq_label = get_feature(
                            f=input, use_gpu=1, feature_types=9)
                        pred = model(flat_seq_c, flat_seq_d, flat_d, hh_seq_c, hh_seq_d, hh_seq_label,
                                          seq_c[:, :, -1, :],
                                          seq_d[:, :, -1, :], seq_label[:, :, -1, :])
                    elif model_name in ['Informer', 'logspare_transformer', 'TFT']:
                        pred = model(h17_seq_features, cur_features, day_features)
                    else:
                        pred = model(h17_seq_features)

                    y_pred.append(pred)
                    y_true.append(y)
        y_true = torch.cat(y_true, dim=0)
        y_pred = torch.cat(y_pred, dim=0)
        if config.train.loss == 'tweedie':
            y_pred = torch.exp(y_pred)
        np.save(os.path.join(log_dir, 'test_true_result.npy'), y_true.cpu().numpy())
        np.save(os.path.join(log_dir, 'test_pred_result.npy'), y_pred.cpu().numpy())
        Tweedie = TweedieLoss(p=1.4)
        mae_thresh = None
        wmape, mape, tweedie = WMAPE(y_true, y_pred).item(), MAPE(y_true, y_pred).item(), (Tweedie(y_pred, y_true)).mean().item()
        logger.info("Average Horizon, wmape: {:.4f}, mape: {:.4f}, tweedie: {:.4f}".format(wmape, mape, tweedie))
        mae_0, mae_0_10, mae_10_20, mae_20 = MAE(y_true, y_pred)
        logger.info("Average Horizon, mae_0: {:.4f}, mae_0_10: {:.4f}, mae_10_20: {:.4f}, mae_20: {:.4f}".format(mae_0.item(),
                                                                                mae_0_10.item(), mae_10_20.item(), mae_20.item()))

        mae, rmse, rrse = All_Metrics(y_pred, y_true, mae_thresh)
        logger.info("Average Horizon, MAE: {:.4f}, RMSE: {:.4f}, RRSE: {:.4f}".format(mae, rmse, rrse))