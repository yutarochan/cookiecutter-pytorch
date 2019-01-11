'''
Configuration Pipeline
Author: Yuya Jeremy Ong (yuyajeremyong@gmail.com)
'''
from __future__ import print_function
import os
import json
from pprint import pprint
from easydict import EasyDict

from monolog import Logger
# from utils.logger import Logger
# from utils.dirs import create_dirs

def get_config(json_file):
    with open(json_file, 'r') as config_file:
        try:
            config_dict = json.load(config_file)
            config = EasyDict(config_dict)
            return config, config_dict
        except ValueError as e:
            print("Invalid JSON File: " + e)
            exit(-1)

def parse_config(json_file):
    # Get Config Contents
    config, _ = get_config(json_file)

    try:
        # Initialize Logger Object
        # TODO: Change the target logger location based on the monolog system.
        exp = Logger(name=config.exp_name, save_dir='../logs', config=config)

        # Display Experiment Information
        print('='*80)
        print('Experiment Name: ' + config.exp_name)
        print('Version: ' + str(exp.version))
        print('='*80)
    except AttributeError:
        print("Missing parameter in config: exp_name")
        exit(-1)

    # Display Configuration Setting
    print("[Configuration Parameters]\n")
    pprint(config)
    print('-'*80)

    # create some important directories to be used for that experiment.
    # config.summary_dir = os.path.join("experiments", config.exp_name, "summaries/")
    # config.checkpoint_dir = os.path.join("experiments", config.exp_name, "checkpoints/")
    # config.out_dir = os.path.join("experiments", config.exp_name, "out/")
    # config.log_dir = os.path.join("experiments", config.exp_name, "logs/")
    # create_dirs([config.summary_dir, config.checkpoint_dir, config.out_dir, config.log_dir])

    # setup logging in the project
    # setup_logging(config.log_dir)

    # logging.getLogger().info("Hi, This is root.")
    # logging.getLogger().info("After the configurations are successfully processed and dirs are created.")
    # logging.getLogger().info("The pipeline of the project will begin now.")

    return config, exp
