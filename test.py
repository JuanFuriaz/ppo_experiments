"""
PPO Reinforcing learning with Pytorch on RacingCar v0

Juan Montoya

Version 1.3
TODO
-configure to load vae and infomax

!!JUST WORKING WITH IMG STACK 1 FOR SAVING IMAGES!!

Origina Author:
xtma (Github-user)
code:
https://github.com/xtma/pytorch_car_caring
"""
#python test.py --render --eps 3 --model-path param/ppo_net_params_vae32.pkl ndim 32 --vae True --rl-path pretrain_vae/pretrained_vae_32.ckpt --stop false
#python test.py --render --eps 3 --model-path param/ppo_net_params_imgstack_1.pkl --raw True --stop false
#python test_old.py --render --eps 3 --img-stack 4 --stop true

#python test_old.py --render --eps 3 --img-stack 4 --stop false --model-path param/ppo_net_params_1900.pkl --img-stack 4

import argparse

import numpy as np
import os
import gym
import torch
import torch.nn as nn
import gzip
import pickle
#from pretrain_vae.models import VAE
#from pretrain_infomax.models import InfoMax
from contin_vae.models import VAE
from contin_infomax.models import InfoMax
from train import Net
from utils import str2bool
from util import store_data
from argparse import Namespace
from torchvision.transforms import Compose, ToTensor, ToPILImage, Grayscale



parser = argparse.ArgumentParser(description='Test the PPO agent for the CarRacing-v0')
parser.add_argument('--action-repeat', type=int, default=8, metavar='N', help='repeat action in N frames (default: 12)')
parser.add_argument('--img-stack', type=int, default=1, metavar='N', help='stack N image in a state (default: 4)')
parser.add_argument('--seed', type=int, default=0, metavar='N', help='random seed (default: 0)')
parser.add_argument('--render', action='store_true', help='render the environment')
parser.add_argument('--save-data',action='store_true', help='Save data as zip file')
parser.add_argument('--eps', type=int, default=50, metavar='N', help='Episodes for testing')
parser.add_argument('--model-path',  type=str, default='param/ppo_net_params_imgstack_1.pkl', metavar='N', help='Give model path for Representational learning models')
parser.add_argument("--vae", type=str2bool, nargs='?', const=True, default=False, help='select vae')
parser.add_argument("--infomax", type=str2bool, nargs='?', const=True, default=False, help='select infomax')
parser.add_argument("--raw", type=str2bool, nargs='?', const=True, default=True, help='select raw pixel framework')
parser.add_argument("--stop", type=str2bool, nargs='?', const=True, default=True, help="Activate nice mode.")
parser.add_argument('--ndim', type=int, default=256, metavar='G', help='Dimension size shared for both VAE and CNN model')
parser.add_argument('--rl-path',  type=str, default='pretrain_vae/pretrained_vae_32_stack4.ckpt', metavar='N', help='Give model path for Representational learning models')

data_dict = {}
args = parser.parse_args()
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
torch.manual_seed(args.seed)
if use_cuda:
    torch.cuda.manual_seed(args.seed)


class Env():
    """
    Test environment wrapper for CarRacing 
    """

    def __init__(self):
        self.env = gym.make('CarRacing-v0')
        self.env.seed(args.seed)
        self.reward_threshold = self.env.spec.reward_threshold
        self.transform = Compose([ToPILImage(), Grayscale(), ToTensor()])

    def reset(self):
        self.counter = 0
        self.av_r = self.reward_memory()

        self.die = False
        img_rgb = self.env.reset()
        img_gray = self.rgb2gray(img_rgb)
        self.stack = [img_gray] * args.img_stack
        return np.array(self.stack)

    def step(self, action, steps =None, i_ep = None):
        total_reward = 0
        terminate = 0
        for i in range(args.action_repeat):
            img_rgb, reward, die, _ = self.env.step(action)
            img_gray = self.rgb2gray(img_rgb)
            # don't penalize "die state"
            if die:
                reward += 100
            # green penalty
            if np.mean(img_rgb[:, :, 1]) > 185.0:
                reward -= 0.05
            total_reward += reward
            # if no reward recently, end the episode
            done = True if self.av_r(reward) <= -0.1 else False
            if args.save_data:
                if done:
                    terminate = 1
                elif die:
                    terminate = 2
                if steps == 0:
                    data_dict[i_ep] = {steps: {"state": img_gray, "reward": reward, "actions": action, "terminate": terminate}}
                else:
                    data_dict[i_ep][steps] = {"state": img_gray, "reward": reward, "actions": action,
                                          "terminate": terminate}
                steps += 1
            if done or die:
                break
        #TODO:
        self.stack.pop(0)
        self.stack.append(img_gray)
        assert len(self.stack) == args.img_stack
        if args.save_data:
            return np.array(self.stack), total_reward, done, die, steps
        else:
            return np.array(self.stack), total_reward, done, die

    def render(self, *arg):
        self.env.render(*arg)

    def rgb2gray(self, rgb):
        """
        Transform images to gray and (in case selected) normalize
        Use Tensor
        """
        # rgb image -> gray [0, 1]
        """
        gray = np.dot(rgb[..., :], [0.299, 0.587, 0.114])
        if norm:
            # normalize
            gray = gray / 128. - 1.
        """
        gray = self.transform(rgb).to("cpu")
        gray = gray.squeeze(0)
        gray = gray.numpy()
        return gray

    @staticmethod
    def reward_memory():
        count = 0
        length = 100
        history = np.zeros(length)

        def memory(reward):
            nonlocal count
            history[count] = reward
            count = (count + 1) % length
            return np.mean(history)

        return memory

'''
class Net(nn.Module):
    """
    Actor-Critic Network for PPO
    """

    def __init__(self):
        super(Net, self).__init__()

        if args.vae:  #representational learning
            # load vae model
            self.vae = self.load_vae(args)

        elif args.infomax:
            # load infomax
            pass
        else:
            self.cnn_base = nn.Sequential(  # input shape (4, 96, 96)
                nn.Conv2d(args.img_stack, 8, kernel_size=4, stride=2),
                nn.ReLU(),  # activation
                nn.Conv2d(8, 16, kernel_size=3, stride=2),  # (8, 47, 47)
                nn.ReLU(),  # activation
                nn.Conv2d(16, 32, kernel_size=3, stride=2),  # (16, 23, 23)
                nn.ReLU(),  # activation
                nn.Conv2d(32, 64, kernel_size=3, stride=2),  # (32, 11, 11)
                nn.ReLU(),  # activation
                nn.Conv2d(64, 128, kernel_size=3, stride=1),  # (64, 5, 5)
                nn.ReLU(),  # activation
                nn.Conv2d(128, args.ndim, kernel_size=3, stride=1),  # (128, 3, 3)
                nn.ReLU(),  # activation
            )  # output shape (256, 1, 1)
            print("Raw pixel loaded")
            self.apply(self._weights_init)

        self.v  = nn.Sequential(nn.Linear(args.ndim, 100), nn.ReLU(), nn.Linear(100, 1))
        self.fc = nn.Sequential(nn.Linear(args.ndim, 100), nn.ReLU())
        self.alpha_head = nn.Sequential(nn.Linear(100, 3), nn.Softplus())
        self.beta_head = nn.Sequential(nn.Linear(100, 3), nn.Softplus())

    @staticmethod
    def _weights_init(m):
        """
         Weights initialization
        """
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
            nn.init.constant_(m.bias, 0.1)

    def forward(self, x):
        """
        Gives two function results
        1) Beta and Gamma for computing the distribution of the policy (using beta distribution)
        2) Value function for advantage term
        """
        # TODO: USE ANOTHE TYPE OF BASE

        if args.vae or args.infomax:  # representational learning
            # load vae model
            x = self.get_z(x)
        else:
            x = self.cnn_base(x)
            x = x.view(-1, 256)

        v = self.v(x)
        x = self.fc(x)
        alpha = self.alpha_head(x) + 1
        beta = self.beta_head(x) + 1

        return (alpha, beta), v

    # TODO: More general to infomax. Wating to see how to load it
    @staticmethod
    def load_vae(args_parser):
        hparams = Namespace(**torch.load(args_parser.rl_path)["hparams"])
        state_dict = torch.load(args.rl_path)["state_dict"]
        assert hparams.ndim == args_parser.ndim
        assert hparams.img_stack == args_parser.img_stack
        print("VAE Loaded")
        vae = VAE(hparams).to(device)  # Load VAE with parameters
        vae.load_state_dict(state_dict)  # Load weights
        # TODO: make nicer freezing parameters
        # for i, chld in enumerate(vae.children()):  #Freeze weights
        #    for params in chld.parameters():
        #        params.requires_grad = False
        return vae

    def get_z(self, x):
        mu, logvar = self.vae.encode(x)
        return self.vae.reparameterize(mu, logvar).to(device)
'''

class Agent():
    """
    Agent for testing
    """

    def __init__(self):
        self.net = Net().float().to(device)

    def select_action(self, state):
        state = torch.from_numpy(state).float().to(device).unsqueeze(0)
        with torch.no_grad():
            alpha, beta = self.net(state)[0]
        action = alpha / (alpha + beta)

        action = action.squeeze().cpu().numpy()
        return action

    def load_param(self):
        #TODO add for vae and InfoMax

        self.net.load_state_dict(torch.load(args.model_path))


if __name__ == "__main__":
    #Command for saving data for imant 13-02-2020
    #python test.py --save-data --model-path param/ppo_net_params_imgstack_1.pkl --img-stack 1 --action-repeat 8 --eps 50
    agent = Agent()
    agent.load_param()
    env = Env()



    training_records = []
    running_score = 0
    state = env.reset()
    for i_ep in range(args.eps):
        score = 0
        state = env.reset()
        steps = 0
        for t in range(1000):
            action = agent.select_action(state)
            action = action * np.array([2., 1., 1.]) + np.array([-1., 0., 0.])
            if args.save_data:
                state_, reward, done, die, steps = env.step(action, steps, i_ep)
                #print("Eps ", i_ep, "t ", t, "steps ", steps)
            else:
                state_, reward, done, die = env.step(action)  # Transform the actions so that in can read left turn
            if args.render:
                env.render()
            score += reward
            state = state_
            if args.stop:
                if die or done:
                    break

        print('Ep {}\tScore: {:.2f}\t'.format(i_ep, score))
    if args.save_data:
        name =  "".join([ "data_", args.model_path.split("/")[-1].split(".")[0], "_", str(args.eps)])
        store_data(data_dict, name)