"""
----------------------------------
Start time : 2023-1-2
Building MS-IRTNet...

--------------By ZZW--------------
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import argparse
import yaml
import time
from Runners.Base_runners import Train

from test import Test
# Args
def get_parser():
    parser = argparse.ArgumentParser(description='SSC-Train')

    parser.add_argument('--sys_config', type=str, default='./Configs/MS-IRTNet.yaml')
    parser.add_argument('--data_config', type=str, default='./Configs/kitti_ssc_voxel_data.yaml')

    args_cfg = parser.parse_args()

    config_sys = yaml.safe_load(open(args_cfg.sys_config, 'r'))
    config_sys['System_Parameters']['time'] = time.strftime('%Yy%mm%dd-%Hh%Mm', time.localtime())
    config_sys['is_test'] = 'False'
    return config_sys

if __name__ == '__main__':
    print('-' * 35)
    print('-----------Start!!!---------------')
    sys_args = get_parser()
    print(sys_args['System_Parameters']['time'])

    # train
    #
    # trainer = Train(sys_args=sys_args)
    # trainer.run_train()

    #test
    #
    test = Test(sys_args=sys_args)
    test.fuse()