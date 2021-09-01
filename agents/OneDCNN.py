#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  4 11:28:18 2021
@author: jay
Main Agent for our 1DCNN
"""
import numpy as np

from tqdm import tqdm 
import shutil

import h5py
import json

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
#plt.rcParams.update({'font.size': 16})
import scikitplot as skplt

from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc

from scipy import interp
import torch
from torch import nn
from torch.backends import cudnn
from torch.autograd import Variable

from torch.utils.data import Dataset, DataLoader

from datasets.auto_covariances import covariance_dataloader
from agents.base import BaseAgent
from graphs.models.onedcnn import OneDCNN
#from graphs.models.onedcnn2 import NeuralNet
#from graphs.models.resnet_model import resnet34
#from graphs.models.resnet_m2 import resnet50

from graphs.losses.cross_entropy import CrossEntropyLoss
from sklearn.metrics import roc_auc_score
'''
with h5py.File('data/0-4source_20snr_100k.h5','r') as hdf: #Read hdf5 file and converts into a numpy aray
    ls=list(hdf.keys())
    print('Dataset List: \n', ls)
    X =  np.array(hdf.get('extracted_x'))#extrated_x
    y = np.array(hdf.get('target_y'))#target_y
'''

from tensorboardX import SummaryWriter
from utils.metrics import AverageMeter, AverageMeterList, cls_accuracy, check_accuracy
from utils.misc import print_cuda_statistics
from utils.train_utils import adjust_learning_rate

cudnn.benchmark = True


    
class OneDCNNAgent(BaseAgent):
    def __init__(self, config):
        super().__init__(config)
        # Create an instance from the Model
        self.model = OneDCNN(self.config)
        #self.model = resnet50(config)
        #self.model = resnet34(self.config)
        #self.model = NeuralNet(self.config)
        # Create an instance from the data loader
        self.data_loader = covariance_dataloader(self.config)
        # Create instance from the loss
        self.loss = CrossEntropyLoss()
        # Create instance from the optimizer
        self.optimizer = torch.optim.SGD(self.model.parameters(),
                                         lr=self.config.learning_rate,
                                         momentum=float(self.config.momentum),
                                         weight_decay=self.config.weight_decay,
                                         nesterov=True)
        # initialize my counters
        self.current_epoch = 0
        self.current_iteration = 0
        self.best_valid_acc = 0
        self.train_accuracy = 0
        self.valid_accuracy = 0
        #Initialize my accuracyl ists
        self.training_accuracy_list = []
        self.validation_accuracy_list = []
        # Check is cuda is available or not
        self.is_cuda = torch.cuda.is_available()
        # Construct the flag and make sure that cuda is available
        self.cuda = self.is_cuda & self.config.cuda

        if self.cuda:
            self.device = torch.device("cuda")
            torch.cuda.manual_seed_all(self.config.seed)
            torch.cuda.set_device(self.config.gpu_device)
            self.logger.info("Operation will be on *****GPU-CUDA***** ")
            #print_cuda_statistics()
        else:
            self.device = torch.device("cpu")
            torch.manual_seed(self.config.seed)
            self.logger.info("Operation will be on *****CPU***** ")

        self.model = self.model.to(self.device)
        self.loss = self.loss.to(self.device)
        # Model Loading from the latest checkpoint if not found start from scratch.
        self.load_checkpoint(self.config.checkpoint_file)
        # Tensorboard Writer
        self.summary_writer = SummaryWriter(log_dir=self.config.summary_dir, comment='1DCNN')

    def save_checkpoint(self, filename='checkpoint.pth.tar', is_best=0):
        """
        Saving the latest checkpoint of the training
        :param filename: filename which will contain the state
        :param is_best: flag is it is the best model
        :return:
        """
        state = {
            'epoch': self.current_epoch,
            'iteration': self.current_iteration,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }
        # Save the state
        torch.save(state, self.config.checkpoint_dir + filename)
        # If it is the best copy it to another file 'model_best.pth.tar'
        if is_best:
            shutil.copyfile(self.config.checkpoint_dir + filename,
                            self.config.checkpoint_dir + 'model_best.pth.tar')

    def load_checkpoint(self, filename):
        filename = self.config.checkpoint_dir + filename
        try:
            self.logger.info("Loading checkpoint '{}'".format(filename))
            checkpoint = torch.load(filename)

            self.current_epoch = checkpoint['epoch']
            self.current_iteration = checkpoint['iteration']
            self.model.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])

            self.logger.info("Checkpoint loaded successfully from '{}' at (epoch {}) at (iteration {})\n"
                             .format(self.config.checkpoint_dir, checkpoint['epoch'], checkpoint['iteration']))
        except OSError as e:
            self.logger.info("No checkpoint exists from '{}'. Skipping...".format(self.config.checkpoint_dir))
            self.logger.info("**First time to train**")

    def run(self):
        """
        This function will the operator
        :return:
        """
        try:
            if self.config.mode == 'test':
                self.validate()
            else:
                self.train()

        except KeyboardInterrupt:
            self.logger.info("You have entered CTRL+C.. Wait to finalize")

    def train(self):
        """
        Main training function, with per-epoch model saving
        """
        for epoch in range(self.current_epoch, self.config.max_epoch):
            self.current_epoch = epoch
            self.train_one_epoch()
            self.training_accuracy_list.append(self.train_accuracy)

            valid_acc = self.validate()
            self.validation_accuracy_list.append(self.valid_accuracy)
            is_best = valid_acc > self.best_valid_acc
            if is_best:
                self.best_valid_acc = valid_acc
            self.save_checkpoint(is_best=is_best)

    def train_one_epoch(self):
        """
        One epoch training function
        """
        # Initialize tqdm
        tqdm_batch = tqdm(self.data_loader.train_loader, total=self.data_loader.train_iterations,
                          desc="Epoch-{}-".format(self.current_epoch))
        # Set the model to be in training mode
        self.model.train()
        # Initialize your average meters
        self.epoch_loss_tr = AverageMeter()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
        top1_acc = AverageMeter()
        top5_acc = AverageMeter()

        current_batch = 0
        for x, y in tqdm_batch:
            if self.cuda:
                x, y = x.cuda(non_blocking=self.config.async_loading), y.cuda(non_blocking=self.config.async_loading)

            # current iteration over total iterations
            progress = float(self.current_epoch * self.data_loader.train_iterations + current_batch) / (
                    self.config.max_epoch * self.data_loader.train_iterations)
            # progress = float(self.current_iteration) / (self.config.max_epoch * self.data_loader.train_iterations)
            x, y = Variable(x), Variable(y)
            y = y.squeeze_()
            y = torch.tensor(y, dtype=torch.long)

            lr = adjust_learning_rate(self.optimizer, self.current_epoch, self.config, batch=current_batch,
                                      nBatch=self.data_loader.train_iterations)
            # model
            
            pred = self.model(x)
            
            y = y.to(self.device)
	
            # loss
            cur_loss = self.loss(pred, y)
            if np.isnan(float(cur_loss.item())):
                raise ValueError('Loss is nan during training...')
            # optimizer
            self.optimizer.zero_grad()
            cur_loss.backward()
            self.optimizer.step()

           # top1, top5 = cls_accuracy(pred.data, y.data, topk=(1, 5))
            
            self.epoch_loss_tr.update(cur_loss.item())
           # top1_acc.update(top1.item(), x.size(0))
            #top5_acc.update(top5.item(), x.size(0))

            self.current_iteration += 1
            current_batch += 1

            self.summary_writer.add_scalar("epoch/loss", self.epoch_loss_tr.val, self.current_iteration)
            self.summary_writer.add_scalar("epoch/accuracy", self.train_accuracy, self.current_iteration)
        tqdm_batch.close()

        self.train_accuracy = check_accuracy(self.data_loader.train_loader, self.model, self.config.batch_size)        
        self.logger.info("Training at epoch-" + str(self.current_epoch) + " | " + "loss: " + str(
            self.epoch_loss_tr.val) + "- Training Acc: " + str(self.train_accuracy))

    def validate(self):
        """
        One epoch validation
        :return:
        """
        tqdm_batch = tqdm(self.data_loader.valid_loader, total=self.data_loader.valid_iterations,
                          desc="Valiation at -{}-".format(self.current_epoch))

        # set the model in training mode
        
        

        self.epoch_loss_val = AverageMeter()
        top1_acc = AverageMeter()
        top5_acc = AverageMeter()

        for x, y in tqdm_batch:
            if self.cuda:
                x, y = x.cuda(non_blocking=self.config.async_loading), y.cuda(non_blocking=self.config.async_loading)

            x, y = Variable(x), Variable(y)
            y = y.squeeze_()
            y = torch.tensor(y, dtype=torch.long)
            # model
            
            pred = self.model(x)
            # loss
            y = y.to(self.device)
            pred = pred.to(self.device)
            cur_loss = self.loss(pred, y)
            if np.isnan(float(cur_loss.item())):
                raise ValueError('Loss is nan during validation...')

           # top1, top5 = cls_accuracy(pred.data, y.data, topk=(1, 5))
            self.valid_accuracy = check_accuracy(self.data_loader.valid_loader, self.model,self.config.batch_size)
            self.epoch_loss_val.update(cur_loss.item())
            #top1_acc.update(top1.item(), x.size(0))
            #op5_acc.update(top5.item(), x.size(0))

        self.logger.info("Validation results at epoch-" + str(self.current_epoch) + " | " + "loss: " + str(
            self.epoch_loss_val.avg) + "- Accuracy: " + str(self.valid_accuracy))

        tqdm_batch.close()

        return top1_acc.avg

    def test(self):
        """
        One epoch test
        :return:
        """
        self.config.mode = 'test'
        self.data_loader = covariance_dataloader(self.config)
        
        tqdm_batch = tqdm(self.data_loader.test_loader, total=self.data_loader.test_iterations,
                          desc="Test at -{}-".format(self.current_epoch))

        # set the model in training mode
        self.model.eval()
        current_batch = 0
        self.epoch_loss_test = AverageMeter()

        for x, y in tqdm_batch:
            if self.cuda:
                x, y = x.cuda(non_blocking=self.config.async_loading), y.cuda(non_blocking=self.config.async_loading)

            # current iteration over total iterations
            progress = float(self.current_epoch * self.data_loader.test_iterations + current_batch) / (
                    self.config.max_epoch * self.data_loader.test_iterations)
            # progress = float(self.current_iteration) / (self.config.max_epoch * self.data_loader.train_iterations)
            x, y = Variable(x), Variable(y)
            y = y.squeeze_()
            y = torch.tensor(y, dtype=torch.long)

            lr = adjust_learning_rate(self.optimizer, self.current_epoch, self.config, batch=current_batch,
                                      nBatch=self.data_loader.test_iterations)
            # model
            
            pred = self.model(x)
            #device = "cpu"
            y = y.to('cuda')
            pred = pred.to('cuda')
            # loss
            cur_loss = self.loss(pred, y)
            if np.isnan(float(cur_loss.item())):
                raise ValueError('Loss is nan during testing, go check your code...')
            # optimizer
            self.optimizer.zero_grad()
            cur_loss.backward()
            self.optimizer.step()

           # top1, top5 = cls_accuracy(pred.data, y.data, topk=(1, 5))
            self.test_accuracy = check_accuracy(self.data_loader.test_loader, self.model, self.config.batch_size)
            self.epoch_loss_test.update(cur_loss.item())
           # top1_acc.update(top1.item(), x.size(0))
            #top5_acc.update(top5.item(), x.size(0))

            self.current_iteration += 1
            current_batch += 1

            self.summary_writer.add_scalar("epoch/loss", self.epoch_loss_test.val, self.current_iteration)
            self.summary_writer.add_scalar("epoch/accuracy", self.test_accuracy, self.current_iteration)
        tqdm_batch.close()

        self.logger.info("Testing at epoch-" + str(self.current_epoch) + " | " + "loss: " + str(
            self.epoch_loss_test.val) + "- Testing Acc: " + str(self.train_accuracy))
        

        return
    
    def plotter(self):
        
        fig = plt.figure(0)
        #plt.plot(self.validation_accuracy_list, label = 'Tr. acc.')
        plt.xlabel('Iterations  ')
        plt.ylabel('Accuracy') 
        #plt.title(' Training and validation accuracy')
        #fig.savefig(self.config.out_dir + 'taccuracy.eps', format = 'eps')


        #fig2 =  plt.figure(1)
       # plt.plot(self.training_accuracy_list, label = 'Val. acc.')
        plt.xlabel('Iterations  ')
        plt.ylabel('Accuracy')
        plt.legend()
        
        mytuple = list(zip(self.training_accuracy_list, self.validation_accuracy_list))
        dfplot = pd.DataFrame(mytuple, columns = ['Training_accuracy', 'Validation_accuracy'])
        dfplot2 = dfplot.reset_index().melt(id_vars = 'index')
        sns.set_style("darkgrid")
        dfplot2 = dfplot2.rename(columns = {"index": "Epochs", "value": "Accuracy"})
        figurep = sns.lineplot(data = dfplot2, x = 'Epochs', y = 'Accuracy', hue = 'variable', palette = ['red', 'blue'])
       # figurep.savefig(self.config.out_dir + 'aplots.png', dpi=1200)
        plt.savefig(self.config.out_dir + 'accuracyplots.png')
        plt.show()
        dfplot2.to_csv(self.config.out_dir + 'accuracy.csv')
#        plt.title(('Accuracy')
        #fig2.savefig(self.config.out_dir + 'vaccuracy.png')
        
    def cmap(self):
        
        nb_classes = self.config.num_classes
        confusion_matrix = torch.zeros(nb_classes, nb_classes)
        self.config.mode = 'test'
        self.data_loader = covariance_dataloader(self.config)
        with torch.no_grad():
            for i, data in enumerate(self.data_loader.test_loader, 0):
                inputs, labels = data
        #inputs = inputs.to(device)
       # classes = classes.to(device)
                outputs = self.model(inputs)
                _, preds = torch.max(outputs, 1)
                for t, p in zip(labels.view(-1), preds.view(-1)):
                        confusion_matrix[t.long(), p.long()] += 1

        print(confusion_matrix)
        print(confusion_matrix.diag()/confusion_matrix.sum(1))
        
        plt.figure(figsize=(15,10))
        class_names = [i for i in range(nb_classes)]
        sns.set(font_scale = 2)
        df_cm = pd.DataFrame(confusion_matrix, index=class_names, columns=class_names).astype(int)
        heatmap = sns.heatmap(df_cm, annot=True, fmt="d")
        
        heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right',fontsize=15)
        heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right',fontsize=15)
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        error = torch.ones(nb_classes)
    
        figure1 = heatmap.get_figure()    
        figure1.savefig(self.config.out_dir + 'svm_conf.png', dpi=400)
        

        TP = confusion_matrix.diag()
        for c in range(nb_classes):
            idx = torch.ones(nb_classes).byte()
            idx[c] = 0
            # all non-class samples classified as non-class
            TN = confusion_matrix[idx.nonzero()[:, None], idx.nonzero()].sum() #conf_matrix[idx[:, None], idx].sum() - conf_matrix[idx, c].sum()
            # all non-class samples classified as class
            FP = confusion_matrix[idx, c].sum()
            # all class samples not classified as class
            FN = confusion_matrix[c, idx].sum()
            se = TP[c]/(TP[c]+FN)
            sp = TN/(TN+FP) 
            p = TP[c]/(TP[c] + FP)
            
            error[c] = (FP + FN) / (TP[c] + TN + FP + FN) 
            
            print('Class {}\nTP {}, TN {}, FP {}, FN {}'.format(
                c, TP[c], TN, FP, FN))
            print('Specificty of this class is ', sp )
            print('Sensitivity of this class is', se)
            print('Precision is ara ara', p)
        
       
        
    def boxxer(self):
        
        self.config.mode = 'test'
        with torch.no_grad():
            for i, data in enumerate(self.data_loader.test_loader, 0):
                inputs, labels = data
                labels = labels.squeeze_()
                outputs = self.model(inputs)
                _, preds = torch.max(outputs, 1)
                preds = preds.cpu()
                labels = labels.cpu()
                y = preds.flatten().numpy()
        
        df = pd.DataFrame(data={'Predictions': y, 'Number of signals': labels})
        
        plt.figure()
        

        ax = sns.boxplot(x="Number of signals", y="Predictions", data=df)
        ax.get_figure().savefig(self.config.out_dir + 'box_plot.png', dpi=1200)
        plt.figure()
        ax2 = sns.swarmplot(x="Number of signals", y="Predictions", data=df)
        ax2.get_figure().savefig(self.config.out_dir + 'swarm_plot.png', dpi=1200)

         
        
                
    def roc(self):
        
        self.config.mode = 'test'
        with torch.no_grad():
            for i, data in enumerate(self.data_loader.test_loader, 0):
                inputs, labels = data
                labels = labels.squeeze_()
                outputs = self.model(inputs)
                _, preds = torch.max(outputs, 1)
                preds = preds.cpu()
                labels = labels.cpu()
                y = preds.flatten().numpy()
        
        nb_classes = self.config.num_classes

        class_names = [i for i in range(nb_classes)]
        y = label_binarize(y, class_names)
        labels = label_binarize(labels, class_names)
        
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for i in range(nb_classes):
            fpr[i], tpr[i], _ = roc_curve(labels[:, i], y[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
         
        all_fpr = np.unique(np.concatenate([fpr[i] for i in range(nb_classes)]))
        mean_tpr = np.zeros_like(all_fpr)
        for i in range(nb_classes):
            mean_tpr += interp(all_fpr, fpr[i], tpr[i])
        mean_tpr /= nb_classes
        fpr["macro"] = all_fpr
        tpr["macro"] = mean_tpr

  
        roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
        fig1 = plt.figure()
        plt.plot(fpr["macro"], tpr["macro"], "r-",label = "ROC curve" )
        plt.plot([0,1], [0,1], 'k--')
        plt.legend(loc='lower right')
        fig1 = plt.gcf()
        fig1.savefig(self.config.out_dir + "Macro.png")	
        macro_roc_auc_ovr = roc_auc_score(labels, y, multi_class="ovr",
                                  average="macro")
        weighted_roc_auc_ovr = roc_auc_score(labels, y, multi_class="ovr",
                                     average="weighted")
        print("One-vs-Rest ROC AUC scores:\n{:.6f} (macro),\n{:.6f} "
      "(weighted by prevalence)"
      .format(macro_roc_auc_ovr, weighted_roc_auc_ovr))  
        # Plot of a ROC curve for a specific class
        for i in range(nb_classes):
            #fig = plt.figure()
            plt.plot(fpr[i], tpr[i], label = 'Label %i' %i)
            #plt.plot(fpr[i], tpr[i],"r-", label='ROC (area = %0.2f)' % roc_auc[i])
            plt.plot([0, 1], [0, 1], 'k--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('ROC')
            plt.legend()
        fig1 = plt.gcf()
            #plt.show()
            #fig.savefig(self.config.out_dir + '%s.eps' % (str(nb_classes[i])), format = 'eps')
        fig1.savefig(self.config.out_dir + "Instant2{}.eps".format(i), format = 'eps')
            #sns.lineplot(x = fpr[i], y = tpr[i], palette = "flare")
            
            #fig.savefig(self.config.out_dir + str(nb_classes[i]) + 'th_roc.png')
            
            
            

        
    def store_scalars(self):
        
        values = {}
        values['exp-parameters'] = {'exp_name' : str(self.config.exp_name), 'numberofsig' : str(self.config.data_root), 'learning_rate' : str(self.config.learning_rate), 'batch_size' : str(self.config.batch_size)}
        values['loss'] = {'training' : str(self.epoch_loss_tr.val), 'validation' : str(self.epoch_loss_val.avg), 'testing' : str(self.epoch_loss_test.val) }
        values['accuracy'] = {'training' : str(self.train_accuracy), 'validation' : str(self.valid_accuracy), 'testing' : str(self.test_accuracy) }
        
        with open(self.config.summary_dir +"all_scalars.json", "w" ) as f:
            json.dump(values, f)

    def finalize(self):
        """
        Finalize all the operations of the 2 Main classes of the process the operator and the data loader
        :return:
        """
        self.logger.info("Please wait while finalizing the operation.. Thank you")
        self.save_checkpoint()
        self.summary_writer.export_scalars_to_json("{}all_scalars.json".format(self.config.summary_dir))
        self.summary_writer.close()
       
        self.data_loader.finalize()
        self.plotter()
        
        self.device = "cpu"
        self.test()
        self.cmap()
        self.boxxer()
        self.roc()
        self.store_scalars()
        
