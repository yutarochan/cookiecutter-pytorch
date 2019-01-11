'''
MNIST Agent
Author: Yuya Jeremy Ong (yuyajeremyong@gmail.com)
'''
from __future__ import print_function
import random
import numpy as np
from tqdm import tqdm

import torch
from torch import nn
from torch.backends import cudnn
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F

from agents.base import BaseAgent
from graphs.models.mnist import MNIST
from datasets.mnist import MNISTDataLoader

# from utils.misc import print_cuda_statistics

cudnn.benchmark = True

class MNISTAgent(BaseAgent):
    def __init__(self, config):
        super().__init__(config)

        # Initializze Model
        self.model = MNIST()

        # Initialize Data Loader
        # TODO Parameterize data loader based on configuration file.
        self.data_loader = MNISTDataLoader(config)

        # Define Loss Function
        self.loss = nn.NLLLoss()

        # Define Optimizer
        self.optimizer = optim.SGD(self.model.parameters(), lr=self.config.learning_rate, momentum=self.config.momentum)

        # Initialize Counter
        self.curr_epoch = 0
        self.curr_iter = 0
        self.best_metric = 0

        # Initialize CUDA Parameters
        self.is_cuda = torch.cuda.is_available()
        if self.is_cuda and not self.config.cuda:
            self.logger.info('Warning: You have a CUDA device available, you should enable CUDA for better performance.')

        self.cuda = self.is_cuda & self.config.cuda

        # Initialize PRNG Seed Values
        self.manual_seed = self.config.seed
        torch.cuda.manual_seed(self.manual_seed)
        # np.random.seed(0) # Set Numpy Seed

        # Set CUDA Device
        # TODO: Configure for multi parallel processing
        if self.cuda:
            self.device = torch.device('cuda')
            torch.cuda.set_device(self.config.gpu_device)
            self.model = self.model.to(self.device)
            self.loss = self.loss.to(self.device)

            self.logger.info('Model will use GPU CUDA')
            print_cuda_stats()
        else:
            self.device = torch.device('cpu')
            torch.manual_seed(self.manual_seed)
            self.logger.info('Model will use CPU')

        # Load Last Checkpoint (If Not Found - Start New)
        self.load_checkpoint(self.config.checkpoint_file)

        # Summary Writer
        self.summary_writer = None

    def load_checkpoint(self, file_name):
        pass

    def save_checkpoint(self, file_name='checkpoint.pth.tar', is_best=0):
        pass

    def run(self):
        try:
            self.train()
        except KeyboardInterrupt:
            self.logger.info('You have entered CTRL + C..., waiting to finalize model.')

    def train(self):
        for epoch in range(1, self.config.max_epoch + 1):
            self.train_one_epoch()
            self.validate()
            self.curr_epoch += 1

    def train_one_epoch(self):
        # Enable Model Training Mode
        self.model.train()

        # Iterate Training Process
        for batch_idx, (data, target) in enumerate(self.data_loader.train_loader):
            data, target = data.to(self.device), target.to(self.device)

            # Compute Loss
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = F.nll_loss(output, target)

            # Backprop Gradient
            loss.backward()
            self.optimizer.step()

            if batch_idx % self.config.log_interval == 0:
                self.logger.info('Train Epoch: {} [{}/{} ({:.0f}%)\tLoss: {:.6f}]'.format(
                    self.curr_epoch, batch_idx * len(data), len(self.data_loader.train_loader.dataset),
                    100. * batch_idx / len(self.data_loader.train_loader), loss.item()))

            self.curr_iter += 1

    def validate(self):
        self.model.eval()

        test_loss = 0
        correct = 0

        with torch.no_grad():
            for data, target, in self.data_loader.test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)

                test_loss += F.nll_loss(output, target, size_average=False).item()
                pred = output.max(1, keepdim=True)[1]
                correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(self.data_loader.test_loader.dataset)
        self.logger.info('\nTest Set: Average: Loss: {:.4f}, Accuracy: {}/{} {:.0f}%)\n'.format(
            test_loss, correct, len(self.data_loader.test_loader.dataset),
            100. * correct / len(self.data_loader.test_loader.dataset)))

    def finalize(self):
        pass
