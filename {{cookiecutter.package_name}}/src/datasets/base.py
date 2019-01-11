'''
Base Data Loader
Author: Yuya Jeremy Ong (yuyajeremyong@gmail.com)
'''
from __future__ import print_function

class BaseDataLoader:
    def __init__(self, config):
        self.config = config
        if config.data_mode == 'imgs':
            raise NotImplementedError
        elif config.data_mode == 'numpy':
            raise NotImplementedError
        elif config.data_mode == "random":
            train_data = torch.randn(self.config.batch_size, self.config.input_channels, self.config.img_size, self.config.img_size)
            train_labels = torch.ones(self.config.batch_size).long()
            valid_data = train_data
            valid_labels = train_labels
            self.len_train_data = train_data.size()[0]
            self.len_valid_data = valid_data.size()[0]

            self.train_iterations = (self.len_train_data + self.config.batch_size - 1) // self.config.batch_size
            self.valid_iterations = (self.len_valid_data + self.config.batch_size - 1) // self.config.batch_size

            train = TensorDataset(train_data, train_labels)
            valid = TensorDataset(valid_data, valid_labels)

            self.train_loader = DataLoader(train, batch_size=config.batch_size, shuffle=True)
            self.valid_loader = DataLoader(valid, batch_size=config.batch_size, shuffle=False)
        else:
            raise Exception('Please specify in the json configuration file the data_mode')

    def plot_sample_per_epoch(self):
        raise NotImplementedError

    def make_gif(self, epochs):
        raise NotImplementedError

    def finalize(self):
        pass
