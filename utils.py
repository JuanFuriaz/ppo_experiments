"""
PPO Reinforcing learning with Pytorch on RacingCar v0
Author:
xtma (Github-user)
code:
https://github.com/xtma/pytorch_car_caring
"""
import visdom
import numpy as np
import argparse
import pickle
import gzip
import os

def store_data(data,  name = "data", datasets_dir="./data",):
    # save data
    if not os.path.exists(datasets_dir):
        os.mkdir(datasets_dir)
    data_file = os.path.join(datasets_dir, "".join([name, '.pkl.gzip']))
    f = gzip.open(data_file, 'wb')
    pickle.dump(data, f)


def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

class DrawLine():

    def __init__(self, env, title, xlabel=None, ylabel=None):
        self.vis = visdom.Visdom()
        self.update_flag = False
        self.env = env
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.title = title

    def __call__(
            self,
            xdata,
            ydata,
    ):
        if not self.update_flag:
            self.win = self.vis.line(
                X=np.array([xdata]),
                Y=np.array([ydata]),
                opts=dict(
                    xlabel=self.xlabel,
                    ylabel=self.ylabel,
                    title=self.title,
                ),
                env=self.env,
            )
            self.update_flag = True
        else:
            self.vis.line(
                X=np.array([xdata]),
                Y=np.array([ydata]),
                win=self.win,
                env=self.env,
                update='append',
            )
