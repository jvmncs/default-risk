import torch
from tensorboardX import SummaryWriter
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix
import numpy as np
import os
'''
TensorbardX logger for default-risk kaggle competition

Example:
# set up tensorboard logger to save at end of each epoch
logger = LoggerX('test_mnist', 'mnist_data', save_freq=len(train_loader))
# initialize model metrics with batch size and number of expected batches
model_metrics = CalculateMetrics(batch_size=args.batch_size, batches_per_epoch=len(train_loader))
# for each batch
scores_dict = model_metrics.update_scores(label, pred)
# log the metrics to tensorboard X, assessing best model based on track_score
logger.log(model, optimizer, loss.item(),
           track_score=scores_dict['weighted_acc']/model_metrics.bn,
           scores_dict=scores_dict,
           epoch=epoch, bn=model_metrics.bn,
           batches_per_epoch=model_metrics.batches_per_epoch, phase='train')
'''

class CalculateMetrics(object):
    '''
    Calculating AUC, weighted accuracy and confusion matrix for imbalanced
    binary home default risk problem
    https://www.kaggle.com/c/home-credit-default-risk/
    '''

    def __init__(self, batch_size, batches_per_epoch):
        super(CalculateMetrics, self).__init__()
        self.w_accuracy = 0
        self.auc = 0
        self.fn = 0
        self.fp = 0
        self.tp = 0
        self.tn = 0
        self.loss = 0
        self.scores_dict = {}
        self.batch_size = batch_size
        self.bn = 1 # current batch number
        self.batches_per_epoch = batches_per_epoch

    @staticmethod
    def _update_scores_dict(scores_dict, acc, auc, tn, fn, tp, fp):

        scores_dict['weighted_acc'] = acc
        scores_dict['auc'] = auc
        scores_dict['tn'] = tn
        scores_dict['fn'] = fn
        scores_dict['tp'] = tp
        scores_dict['fp'] = fp

        return scores_dict

    @staticmethod
    def calculate_sample_weight(target):
        # weighting for unbalanced data
        # returns vector with weights applied to the target
        # assigns more weight to the positive label - under-represented in this competition
        num_pos = torch.nonzero(torch.from_numpy(target.squeeze())).size()[0]
        return target * 1 / (num_pos / target.shape[0])

    def update_scores(self, target, pred):
        '''
        Update the running scores and return batch metrics as well
        :param target: true values, torch tensor size (batch_size, 1)
        :param pred: predictions, assuming these are probabilities from a sigmoid activation, same size as target.
        :return: dictionary of metrics which is also an attribute
        '''
        # accepts torch tensors, calculates scores

        target = target.numpy().squeeze()
        pred = pred.numpy().squeeze()

        # calculate sample weights for the batch
        sample_weight = self.calculate_sample_weight(target).squeeze()

        # compute weighted accuracy
        w_accuracy = accuracy_score(target, pred, sample_weight=sample_weight)
        self.w_accuracy += w_accuracy

        # compute auc
        auc = roc_auc_score(target, pred)
        self.auc += auc

        # might be un necessary? Sometimes I like to see these when training as it may help spot issues more quickly
        try:
            tn, fp, fn, tp = confusion_matrix(target.round(), pred).ravel()
        except ValueError:  # indicates only one category predicted
            tn, fp, fn, tp = 0, 0, 0, 0

        # scores for current batch
        self.scores_dict = CalculateMetrics._update_scores_dict(
            self.scores_dict, np.round(auc, 3), np.round(w_accuracy, 3), tn, fn, tp, fp)

        if self.bn == self.batches_per_epoch:  # report final averages for each
            self.w_accuracy /= self.batches_per_epoch
            self.auc /= self.batches_per_epoch
            self.fn /= self.batches_per_epoch
            self.fp /= self.batches_per_epoch
            self.tn /= self.batches_per_epoch
            self.tp /= self.batches_per_epoch

        self.bn += 1

        return self.scores_dict


class LoggerX(object):
    '''
    Monitoring using tensorboardX
    includes writing custom metrics to TB, saving model checkpoints, and tqdm progress bar
    Automatically timestamps each separate run
    tensorboard --logdir ./runs
    '''
    def __init__(self, model_name, data_name, save_freq):
        '''
        :param model_name: directory where logs will be stored
        :param data_name: subdirectory to model_name where logs will be stored
        :param save_freq: how frequently (batches) to check for improved scores and save a checkpoint
        '''
        super(LoggerX, self).__init__()

        self.model_name = model_name
        self.data_name = data_name  # name for subdirectory in case multiple datasets (different augmentation, feature selection, etc) are used
        self.save_freq = save_freq

        self.comment = f'{model_name}_{data_name}'
        self.data_subdir = f'{model_name}/{data_name}'
        self.out_dir = f'./model_weights/{self.data_subdir}'  # creates a directory for model weights
        LoggerX._make_dir(self.out_dir)

        # TensorBoard
        self.writer = SummaryWriter(comment=self.comment)
        self.best_score = 0.5  # set to greater than zero to avoid saving low score results
        self.best_path = ''  # where the best model is saved

    def log(self, model, opt, loss, track_score, scores_dict, epoch, bn, batches_per_epoch, phase='train'):
        '''
        :param model: pytorch model for saving states
        :param opt: pytorch optimizer for saving states
        :param loss: loss value
        :param track_score: metric used for deciding best model
        :param scores_dict: dictionary of scores to record
        :param epoch: current epoch
        :param bn: current batch number
        :param batches_per_epoch: number of total batches
        :param phase: displays with metric name in TB, useful for noting training or validation metrics
        :return:
        '''
        step = LoggerX._step(epoch, bn, batches_per_epoch)
        self.writer.add_scalar(f'{self.model_name}/{phase}_loss', loss, step)

        for k in scores_dict.keys():
            self.writer.add_scalar(f'{self.model_name}/{phase}_{k}',
                                   scores_dict[k], step)

        if step % self.save_freq == 0:  # save freq can be set to batches per epoch if you only want to save at the end of an epoch
            old_path = self.best_path
            if track_score > self.best_score:
                try:
                    os.remove(old_path)
                except OSError:
                    pass
                self.best_score = track_score
                self.save_model_ckpt(epoch, track_score, opt, model)

    def display_status(self, pbar, epoch, lr, loss, w_score, auc):
        ''' for tqdm '''
        pbar.set_postfix(ep=epoch, lr=lr, loss=loss, weighted_acc=w_score, auc=auc)

    def save_model_ckpt(self, epoch, score, opt, model):
        ''' save optimizer and model states '''

        state = {'optimizer': opt.state_dict(),
                 'state_dict': model.state_dict()}
        save_path = os.path.join(self.out_dir, f'{epoch}_{score}.ckpt')
        torch.save(state, save_path)
        self.best_path = save_path
        print('Checkpoint saved to {}'.format(save_path))

    def close(self):
        self.writer.close()

    # Private Functionality
    @staticmethod
    def _step(epoch, n_batch, num_batches):
        return epoch * num_batches + n_batch

    @staticmethod
    def _make_dir(directory):
        try:
            os.makedirs(directory)
        except FileExistsError:
            pass
