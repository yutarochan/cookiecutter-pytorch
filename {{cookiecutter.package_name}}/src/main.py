'''
Pytorch Model Framework
Author: Yuya Jeremy Ong (yuyajeremyong@gmail.com)
'''
from __future__ import print_function
import argparse
from agents import *
from utils.config import *

def main():
    # Parse CLI Arguments
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('config', metavar='config_json_file', default='None', help='File path to the configuration file.')
    args = parser.parse_args()

    # Parse Config File
    config, logger = parse_config(args.config)

    # Generate Agent
    agent_class = globals()[config.agent]
    agent = agent_class(config)
    agent.run()
    agent.finalize()

if __name__ == '__main__':
    main()
