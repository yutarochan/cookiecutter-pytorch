'''
Monolog: Experiment Logger
Author: Yuya Jeremy Ong (yuyajeremyong@gmail.com)
'''
from __future__ import print_function
import os
import json
from datetime import datetime

import numpy as np
import pandas as pd
from imageio import imwrite

# Application Constants
_ROOT = os.path.abspath(os.path.dirname(__file__))

class Logger(object):
    # TODO: Use config parameters into config
    def __init__(self, name='default', save_dir=None, config=None):
        # Change Target Directory
        if save_dir is not None:
            global _ROOT
            _ROOT = save_dir

        # Setup Class Parameters
        self.metrics = []
        self.tags = {}
        self.name = name
        self.debug = config.debug if hasattr(config, 'debug') else False
        self.version = config.version if hasattr(config, 'version') else None
        self.autosave = config.autosave if hasattr(config, 'autosave') else True
        self.description = config.description if hasattr(config, 'description') else None
        self.create_git_tag = config.create_git_tag if hasattr(config, 'create_git_tag') else False
        self.exp_hash = '{}_v{}'.format(self.name, self.version)
        self.created_at = str(datetime.utcnow())

        # Update Verion Hash
        if self.version is None:
            old_version = self.__get_last_experiment_version()
            self.exp_hash = '{}_v{}'.format(self.name, old_version + 1)
            self.version = old_version + 1

        self.__init_cache_file_if_needed()

        # Generate New Log File
        if not self.debug:
            if self.version is not None:
                if not os.path.exists(self.__get_log_name()):
                    self.__create_exp_file(self.version)
                    self.save()
                else:
                    self.__load()
            else:
                old_version = self.__get_last_experiment_version()
                self.version = old_version
                self.__create_exp_file(self.version + 1)
                self.save()

            # Generate Git Tags
            if self.create_git_tag == True:
                desc = self.description if self.description is not None else 'No Description'
                tag_msg = 'Test tube exp: {} - {}'.format(self.name, desc)
                cmd = 'git tag -a exp_{} -m "{}"'.format(self.exp_hash, tag_msg)
                os.system(cmd)

    def argparse(self, argparser):
        parsed = vars(argparser)
        to_add = {}

        for k, v in parsed.items():
            if not callable(v): to_add[k] = v

        self.tag(to_add)

    def add_meta_from_hyperopt(self, hypo):
        meta = hypo.get_current_trial_meta()
        for tag in meta: self.tag(tag)

    ''' File IO Utilities '''
    def __init_cache_file_if_needed(self):
        exp_cache_file = self.get_data_path(self.name, self.version)
        if not os.path.isdir(exp_cache_file): os.makedirs(exp_cache_file)

    def __create_exp_file(self, version):
        exp_cache_file = self.get_data_path(self.name, self.version)
        path = '{}/meta.experiment'.format(exp_cache_file)
        open(path, 'w').close()
        self.version = version

        os.mkdir(self.get_media_path(self.name, self.version))

    def __get_last_experiment_version(self):
        try:
            exp_cache_file = os.sep.join(self.get_data_path(self.name, self.version).split(os.sep)[:-1])
            last_version = -1
            for f in os.listdir(exp_cache_file):
                if 'version_' in f:
                    file_parts = f.split('_')
                    version = int(file_parts[-1])
                    last_version = max(last_version, version)
            return last_version
        except Exception as e:
            return -1

    def __get_log_name(self):
        exp_cache_file = self.get_data_path(self.name, self.version)
        return '{}/meta.experiment'.format(exp_cache_file)

    def tag(self, tag_dict):
        if self.debug: return
        for k, v in tag_dict.items(): self.tags[k] = v
        if self.autosave == True: self.save()

    def log(self, metrics_dict):
        if self.debug: return
        if 'created_at' not in metrics_dict:
            metrics_dict['created_at'] = str(datetime.utcnow())

        self.__convert_numpy_types(metrics_dict)
        self.metrics.append(metrics_dict)
        if self.autosave == True: self.save()

    def __convert_numpy_types(self, metrics_dict):
        for k, v in metrics_dict.items():
            if v.__class__.__name__ == 'float32':
                metrics_dict[k] = float(v)

            if v.__class__.__name__ == 'float64':
                metrics_dict[k] = float(v)

    def save(self):
        if self.debug: return

        self.__save_images(self.metrics)
        metrics_file_path = self.get_data_path(self.name, self.version) + '/metrics.csv'
        meta_tags_path = self.get_data_path(self.name, self.version) + '/meta_tags.csv'

        obj = {
            'name': self.name,
            'version': self.version,
            'tags_path': meta_tags_path,
            'metrics_path': metrics_file_path,
            'autosave': self.autosave,
            'description': self.description,
            'created_at': self.created_at,
            'exp_hash': self.exp_hash
        }

        with open(self.__get_log_name(), 'w') as file: json.dump(obj, file, ensure_ascii=False)

        df = pd.DataFrame({'key': list(self.tags.keys()), 'value': list(self.tags.values())})
        df.to_csv(meta_tags_path, index=False)

        df = pd.DataFrame(self.metrics)
        df.to_csv(metrics_file_path, index=False)

    def __save_images(self, metrics):
        for i, metric in enumerate(metrics):
            for k, v in metric.items():
                img_extension = None
                img_extension = 'png' if 'png_' in k else img_extension
                img_extension = 'jpg' if 'jpg' in k else img_extension
                img_extension = 'jpeg' if 'jpeg' in k else img_extension

                if img_extension is not None:
                    img_name = '_'.join(k.split('_')[1:])
                    save_path = self.get_media_path(self.name, self.version)
                    save_path = '{}/{}_{}.{}'.format(save_path, img_name, i, img_extension)

                    if type(metric[k]) is not str: imwrite(save_path, metric[k])
                    metric[k] = save_path

    def __load(self):
        with open(self.__get_log_name(), 'r') as file:
            data = json.load(file)
            self.name = data['name']
            self.version = data['version']
            self.autosave = data['autosave']
            self.created_at = data['created_at']
            self.description = data['description']
            self.exp_hash = data['exp_hash']

        meta_tags_path = self.get_data_path(self.name, self.version) + '/meta_tags.csv'
        df = pd.read_csv(meta_tags_path)
        self.tags_list = df.to_dict(orient='records')
        self.tags = {}
        for d in self.tags_list:
            k, v = d['key'], d['value']
            self.tags[k] = v

        metrics_file_path = self.get_data_path(self.name, self.version) + '/metrics.csv'
        try:
            df = pd.read_csv(metrics_file_path)
            self.metrics = df.to_dict(orient='records')

            for metric in self.metrics:
                to_delete = []
                for k, v in metric.items():
                    try:
                        if np.isnan(v): to_delete.append(k)
                    except Exception as e:
                        pass

                for k in to_delete: del metric[k]
        except Exception as e:
            self.metrics = []

    def get_data_path(self, exp_name, exp_version):
        return os.path.join(_ROOT, exp_name, 'version_{}'.format(exp_version))

    def get_media_path(self, exp_name, exp_version):
        return os.path.join(self.get_data_path(exp_name, exp_version), 'media')

    ''' Object Overwrites '''
    def __str__(self):
        return 'Exp: {}, v: {}'.format(self.name, self.version)

    def __hash__(self):
        return 'Exp: {}, v: {}'.format(self.name, self.version)
