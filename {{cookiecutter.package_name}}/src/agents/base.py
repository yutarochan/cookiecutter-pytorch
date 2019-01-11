'''
BaseEngine
Author: Yuya Jeremy Ong (yuyajeremyong@gmail.com)
'''
from __future__ import print_function
import logging

class BaseAgent:
    def __init__(self, config):
        self.config = config

        # TODO: Change logger system to monolog logger
        self.logger = logging.getLogger('Agent')

    def load_checkpoint(self, file_name):
        raise NotImplementedError

    def save_checkpoint(self, file_name='checkpoint.pth.tar', is_best=0):
        raise NotImplementedError

    def run(self):
        raise NotImplementedError

    def train(self):
        raise NotImplementedError

    def train_one_epoch(self):
        raise NotImplementedError

    def validate(self):
        raise NotImplementedError

    def finalize(self):
        raise NotImplementedError
