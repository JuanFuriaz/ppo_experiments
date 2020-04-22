"""
PPO Reinforcing learning with Pytorch on RacingCar v0
Juan Montoya

V1.4
- Tensorboard
- New metrics Mean and Median for Tensordboard and vis
- RNN option (maybe need to restart h)
- freeze also for convNets
- csv log

Orignal Author:
xtma (Github-user)
code:
https://github.com/xtma/pytorch_car_caring
"""

import argparse

import numpy as np

import re
import gym
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Beta
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
#from pretrain_vae.models import VAE
#from pretrain_infomax.models import InfoMax
from contin_vae.models import VAE
from contin_infomax.models import InfoMax
from argparse import Namespace
from utils import DrawLine
from utils import str2bool
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor, ToPILImage, Grayscale
from torch.utils.tensorboard import SummaryWriter
import datetime
import csv
import os
import glob

parser = argparse.ArgumentParser(description='Train a PPO agent for the CarRacing-v0')
subparsers = parser.add_subparsers(description='Representational Learning Models ')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G', help='discount factor (default: 0.99)')
parser.add_argument('--ndim', type=int, required=True, metavar='G', help='Dimension size shared for both VAE and CNN model')
parser.add_argument('--action-repeat', type=int, default=8, metavar='N', help='repeat action in N frames (default: 8)')
parser.add_argument('--action-vec', type=int, default=0, metavar='N', help='Action vector for the fully conectected network')
parser.add_argument('--eps', type=int, default=4000, metavar='N', help='Episodes for training')
parser.add_argument('--terminate', action='store_true', help='Termination after the predefined threeshold')
parser.add_argument('--img-stack', type=int, default=1, metavar='N', help='stack N image in a state (default: 4)')
parser.add_argument('--seed', type=int, default=None, metavar='N', help='random seed (default: None)')
parser.add_argument('--render', action='store_true', help='render the environment')
parser.add_argument('--vis', action='store_true', help='use visdom')
parser.add_argument('--tb', action='store_true', help='Tensorboard')
parser.add_argument(
    '--log-interval', type=int, default=10, metavar='N', help='interval between training status logs (default: 10)')
parser.add_argument(
    '--buffer', type=int, default=5000, metavar='N', help='Buffer with Experience ')
parser.add_argument(
    '--batch', type=int, default=128, metavar='N', help='Batch size for sampling')
parser.add_argument(
    '--learning', type=float, default=0.001, metavar='N', help='Learning Rate')
# parser.add_argument("--vae", type=str2bool, nargs='?', const=True, default=False, help='select vae')
# parser.add_argument("--infomax", type=str2bool, nargs='?', const=True, default=False, help='select infomax')
# parser.add_argument("--raw", type=str2bool, nargs='?', const=True, default=True, help='select raw pixel framework')
parser.add_argument('--srl-model-type', type=str, help="specify state representation learning model type. Default of None implies raw pixel model",
        default=None, choices=['infomax', 'vae'])
parser.add_argument('--rnn',  action='store_true',  help='Use gated recurrend unit')
parser.add_argument("--reward-mod", action='store_false',  help='Engineered Reward turn off')
# parser.add_argument("--freeze", action='store_true', help='Freeze layers in representational models')
parser.add_argument("--freeze", type=str2bool, nargs='?', const=True, default=True, help='Freeze layers in representational models')
parser.add_argument('--rl-path',  type=str, default=None, metavar='N', help='Give model path for Representational learning models')
parser.add_argument('--title',  type=str, default=None, metavar='N', help='Name for the image title')
parser.add_argument('--debug',  action='store_true',  help='Debug on')
parser.add_argument('--logdir',  type=str, required=True, help='directory for tensorboard and csv')



args = parser.parse_args()
#print(args)
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

if args.debug:
    args = {'gamma': 0.99,
            'ndim': 32,
            'action_repeat': 8,
            'action_vec': 0,
            'eps': 4000,
            'terminate': False,
            'img_stack': 4,
            'seed': 0,
            'render': False,
            'vis': False,
            'tb': False,
            'log_interval': 10,
            'buffer': 10,
            'batch': 128,
            'learning': 0.001,
            'vae': False,
            'infomax': False,
            'raw': True,
            'rnn': True,
            'reward_mod': True,
            'freeze': False,
            'rl_path': 'pretrain_vae/pretrained_vae_32_stack4.ckpt',
            'title': 'debug',
            'debug': True}
    args = Namespace(**args)
    print("DEBUGGING MODUS")

print("Parameters: ")
print(args)
print("")

# set up versioning
version_dirs = glob.glob("%s/version_*" % args.logdir)
r = re.compile(r"\d+")
versions = [int(r.findall(d)[-1]) for d in version_dirs]
versions = sorted(versions)
new_version = versions[-1] + 1 if len(versions) > 0 else 1
new_logdir = "%s/version_%d" % (args.logdir, new_version)
args.logdir = new_logdir

if args.title is None:
    args.title = os.path.basename(args.rl_path).replace(".ckpt", "")
now_d = datetime.datetime.now().strftime("%d_%m_%Y")
dict_log = "".join([args.logdir, "/"])
# file_log = "".join([args.title, "_", str(args.seed)])
csv_log = "".join([dict_log, args.title, ".csv"])
if args.tb:
    writer = SummaryWriter("".join([dict_log]))
else:
    if not os.path.exists(dict_log):
        os.makedirs(dict_log)
    if os.path.isfile(csv_log):
        os.remove(csv_log)

if args.seed is None:
    args.seed = np.random.randint(2**32-1)
torch.manual_seed(args.seed)
np.random.seed(args.seed)

if use_cuda:
    torch.cuda.manual_seed(args.seed)

if args.action_vec > 0:
    transition = np.dtype([('s', np.float64, (args.img_stack, 96, 96)), ('a', np.float64, (3,)), ('a_logp', np.float64),
                           ('r', np.float64), ('s_', np.float64, (args.img_stack, 96, 96)),('a_v', np.float64, (3*(args.action_vec + 1),)) ])
else:
    transition = np.dtype([('s', np.float64, (args.img_stack, 96, 96)), ('a', np.float64, (3,)), ('a_logp', np.float64),
                       ('r', np.float64), ('s_', np.float64, (args.img_stack, 96, 96))])


class Env():
    """
    Environment wrapper for CarRacing
    """

    def __init__(self):
        """
        Create Env Racing Car
        """
        self.env = gym.make('CarRacing-v0')
        self.env.seed(args.seed)
        self.reward_threshold = self.env.spec.reward_threshold
        self.transform = Compose([ToPILImage(), Grayscale(), ToTensor()])

    def reset(self):
        """
        Restart values
        """
        self.counter = 0
        self.av_r = self.reward_memory()

        self.die = False
        img_rgb = self.env.reset()
        img_gray = self.rgb2gray(img_rgb)
        self.stack = [img_gray] * args.img_stack  # four frames for decision
        return np.array(self.stack)

    def step(self, action):
        """
        Steps giving rewards and (in case) terminate
        """
        total_reward = 0
        for i in range(args.action_repeat):
            img_rgb, reward, die, _ = self.env.step(action)
            # don't penalize "die state"
            if args.reward_mod:
                if die:
                    reward += 100
                # green penalty
                if np.mean(img_rgb[:, :, 1]) > 185.0:
                    reward -= 0.05
                total_reward += reward
                # if no reward recently, end the episode
                done = True if self.av_r(reward) <= -0.1 else False
            else:
                done = die
                total_reward += reward
            if done or die:
                break
        img_gray = self.rgb2gray(img_rgb)
        self.stack.pop(0)
        self.stack.append(img_gray)
        assert len(self.stack) == args.img_stack
        return np.array(self.stack), total_reward, done, die

    def render(self, *arg):
        """
        Show training in video
        """
        self.env.render(*arg)

    def rgb2gray(self, rgb):
        """
        Transform images to gray and (in case selected) normalize
        Use Tensor flow
        """
        gray = self.transform(rgb).to("cpu")
        gray = gray.squeeze(0)
        gray = gray.numpy()
        return gray

    @staticmethod
    def reward_memory():
        """
        Record reward for last 100 steps
        """
        count = 0
        length = 100
        history = np.zeros(length)

        def memory(reward):
            nonlocal count
            history[count] = reward
            count = (count + 1) % length
            return np.mean(history)

        return memory


class Net(nn.Module):
    """
    Actor-Critic Network for PPO
    """

    def __init__(self):
        super(Net, self).__init__()

        # use SRL encoder or init raw pixel encoder
        if args.rl_path is not None:
            self.srl_model = self.load_rl(args)
            if args.ndim is not None and args.ndim != self.srl_model.hparams.ndim:
                raise ValueError("Mismatch between SRL-Model ndim (%d) and RL-Agent ndim (%d)" %
                    (args.ndim, self.srl_model.hparams.ndim))
            self.ndim = self.srl_model.hparams.ndim
            self.img_stack = self.srl_model.hparams.img_stack
        else:
            self.srl_model = nn.Sequential(  # input shape (4, 96, 96)
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

        # optionally, freeze weights
        if args.freeze:
            print("Freezing ConvLayer Weights")
            for i, chld in enumerate(self.srl_model.children()):  # Freeze weights
                for params in chld.parameters():
                    params.requires_grad = False

        if args.srl_model_type is not None and args.action_vec > 0:
            self.v = nn.Sequential(nn.Linear(args.ndim + args.action_vec*3, 100), nn.ReLU(), nn.Linear(100, 1))
            self.fc = nn.Sequential(nn.Linear(args.ndim + args.action_vec*3, 100), nn.ReLU())

        else:
            if args.rnn:
                self.gru = nn.GRUCell(args.ndim, 100)
                self.v =  nn.Linear(100, 1)
            else:
                self.v = nn.Sequential(nn.Linear(args.ndim, 100), nn.ReLU(), nn.Linear(100, 1))
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
            nn.init.constant_(m.bias, 0)



    def forward(self, x):
        """
        Gives two function results
        1) Beta and Gamma for computing the distribution of the policy (using beta distribution)
        2) Value function for advantage term
        """
        if args.srl_model_type == "vae":  #representational learning
            # load vae model
            if args.action_vec > 0:
                x = torch.cat((self.get_z(x[0]), x[1]), dim=1)
            else:
                x = self.get_z(x)
        elif args.srl_model_type == "infomax":
            if args.action_vec > 0:
                x = torch.cat((self.srl_model.encoder(x[0]), x[1]), dim=1)
            else:
                x = self.srl_model.encoder(x)
        else:
            #TODO: Conv with action vector?
            x = self.cnn_base(x)
            x = x.view(-1, args.ndim)
        if args.rnn:
            #h = self.gru(x, self.h)
            h = self.gru(x)
            #self.h = h.detach()
            x = h
            #print(h.shape)
            v = self.v(x)
        else:
            v = self.v(x)
            x = self.fc(x)

        alpha = self.alpha_head(x) + 1
        beta = self.beta_head(x) + 1

        return (alpha, beta), v

    @staticmethod
    def load_rl(args_parser):
        hparams = Namespace(**torch.load(args_parser.rl_path)["hparams"])
        state_dict = torch.load(args.rl_path)["state_dict"]
        assert hparams.ndim == args_parser.ndim
       # assert hparams.img_stack == args_parser.img_stack
        if args.srl_model_type == "vae":
            rl = VAE(hparams).to(device)  # Load VAE with parameters
            print("VAE Loaded")
        elif args.srl_model_type == "infomax":
            rl  = InfoMax(hparams).to(device) # Load VAE with parameters
            print("InfoMax loaded")
        else:
            raise ValueError("Invalid SRL-model-type: %s" % args.srl_model_type)

        rl.load_state_dict(state_dict)  # Load weights
        return rl

    def get_z(self, x):
        mu, logvar = self.srl_model.encode(x)
        return self.srl_model.reparameterize(mu, logvar).to(device)

class Agent():
    """
    Agent for training
    """
    max_grad_norm = 0.5
    clip_param = 0.1  # epsilon in clipped loss
    ppo_epoch = 10
    buffer_capacity, batch_size = args.buffer, args.batch

    def __init__(self):
        self.training_step = 0
        self.net = Net().float().to(device)
        self.buffer = np.empty(self.buffer_capacity, dtype=transition)
        self.counter = 0

        self.optimizer = optim.Adam(self.net.parameters(), lr=args.learning)

    def select_action(self, state):
        if args.action_vec > 0:
            state = (torch.from_numpy(state[0]).float().to(device).unsqueeze(0), torch.from_numpy(state[1]).float().to(device).unsqueeze(0))
        else:
            state = torch.from_numpy(state).float().to(device).unsqueeze(0)
        #TODO CHANGE FOR VECTOR ACTIONS
        with torch.no_grad():
            alpha, beta = self.net(state)[0]
        dist = Beta(alpha, beta)
        action = dist.sample()
        a_logp = dist.log_prob(action).sum(dim=1)

        action = action.squeeze().cpu().numpy()
        a_logp = a_logp.item()
        return action, a_logp

    def save_param(self):
        torch.save(self.net.state_dict(), "".join(['param/ppo_net_params_',  args.title,  "_", str(args.ndim), '.pkl']))

    def store(self, transition):
        """
        Checks if buffer is full
        """
        self.buffer[self.counter] = transition
        self.counter += 1
        if self.counter == self.buffer_capacity:
            self.counter = 0
            return True
        else:
            return False

    def reset(self):   #reset for rnn
        self.net.h = torch.zeros(1, 100).to(device)

    def update(self):
        """
        Update policy gradient by using old batch of experience.
        This happens when the buffer is full
        """
        self.training_step += 1
        s = torch.tensor(self.buffer['s'], dtype=torch.float).to(device)
        a = torch.tensor(self.buffer['a'], dtype=torch.float).to(device)
        r = torch.tensor(self.buffer['r'], dtype=torch.float).to(device).view(-1, 1)
        s_ = torch.tensor(self.buffer['s_'], dtype=torch.float).to(device)
        if args.action_vec > 0: a_v = torch.tensor(self.buffer['a_v'], dtype=torch.float).to(device)

        """
        print("Weights before update: ")
        for k, v in self.net.state_dict().items():
            print("Layer {}".format(k))
            print(v.sum())
            print(v.mean(), v.median())
            print(v.max(), v.min())
        """
        if self.srl_model_type == "vae" and args.tb:
            z = self.net.get_z(s[0].unsqueeze_(0))
            dec2 = self.net.vae.decode(z).squeeze(0)
            imgs =  torch.cat((dec2,  s[0]), dim=2)
            img_grid = torchvision.utils.make_grid(imgs)
            writer.add_image("Encoder-Img", img_grid)

            #save_image(s[0].cpu(), 'test_img/' + args.title + "_" + str(args.ndim) + 'img_update_' + str(self.training_step) + '.png')

        old_a_logp = torch.tensor(self.buffer['a_logp'], dtype=torch.float).to(device).view(-1, 1)

        with torch.no_grad(): # Compute a vector with advantage terms
            if args.action_vec > 0:
                target_v = r + args.gamma * self.net((s_, a_v[:, 3:]))[1]
                adv = target_v - self.net((s_, a_v[:, :-3]))[1]
            else:
                target_v = r + args.gamma * self.net(s_)[1]
                adv = target_v - self.net(s)[1]
            # adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        for _ in range(self.ppo_epoch):
            # Compute update for mini batch
            for index in BatchSampler(SubsetRandomSampler(range(self.buffer_capacity)), self.batch_size, False):
                if args.action_vec > 0:
                    alpha, beta = self.net((s[index], a_v[index, :-3]))[0]
                else:
                    alpha, beta = self.net(s[index])[0]
                dist = Beta(alpha, beta)
                entropy = dist.entropy().mean()
                a_logp = dist.log_prob(a[index]).sum(dim=1, keepdim=True)
                ratio = torch.exp(a_logp - old_a_logp[index])    # old/new_policy for Trust Region Method

                surr1 = ratio * adv[index]
                surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * adv[index]  # Clip Ratio
                action_loss = -torch.min(surr1, surr2).mean()
                # Difference between prediction and real values
                if args.action_vec > 0:
                    value_loss = F.smooth_l1_loss(self.net((s[index], a_v[index, :-3]))[1], target_v[index])
                else:
                    value_loss = F.smooth_l1_loss(self.net(s[index])[1], target_v[index])
                #loss = action_loss + 2. * value_loss
                # Loss with Entropy
                loss = action_loss + 2. * value_loss - 0.001 * entropy

                self.optimizer.zero_grad()
                loss.backward()
                # nn.utils.clip_grad_norm_(self.net.parameters(), self.max_grad_norm)
                self.optimizer.step()

        if args.srl_model_type == "vae":
            z = self.net.get_z(s[0].unsqueeze_(0))
            dec2 =  self.net.vae.decode(z)
            save_image(dec2, 'test_img/' + args.title + "_" + str(args.ndim) +  '_dec_final_' + str(self.training_step) + '.png')


if __name__ == "__main__":
    agent = Agent()
    env = Env()

    if args.vis:
        draw_moving = DrawLine(env="car", title= "".join(["MovingAv", "_", now_d, "_", args.title]), xlabel="Episode", ylabel="Moving averaged episode reward")
        draw_mean = DrawLine(env="car", title="".join(["Mean", "_", now_d, "_", args.title]), xlabel="Episode",
                           ylabel="Moving averaged episode reward")
        draw_median = DrawLine(env="car", title="".join(["Median", "_", now_d, "_", args.title]), xlabel="Episode",
                               ylabel="Moving averaged episode reward")


    training_records = []
    score_l = []
    running_score = 0
    state = env.reset()
    for i_ep in range(1, args.eps+1):
        score = 0

        state = env.reset()
        # TODO:
        if args.rnn: agent.reset()
        act_vec = [np.zeros(3)] * (args.action_vec + 1)  # initialize stack for past actions
        for t in range(1000):
            if args.action_vec > 0:
                action, a_logp = agent.select_action((state, np.array(act_vec).flatten()[:-3]))
            else:
                action, a_logp = agent.select_action(state)
            act_vec.pop(0)       # remove oldest action and add new one
            act_vec.append(action)
            state_, reward, done, die = env.step(action * np.array([2., 1., 1.]) + np.array([-1., 0., 0.]))  # passing the values to negative steering
            if args.render:
                env.render()
            if args.action_vec > 0:
                if agent.store((state, action, a_logp, reward, state_, np.array(act_vec).flatten())):
                    print('updating with action vector')
                    agent.update()
            else:
                if agent.store((state, action, a_logp, reward, state_)):
                    print('updating')
                    agent.update()

            score += reward

            state = state_
            if done or die:
                break
        score_l.append(score)
        running_score = running_score * 0.99 + score * 0.01

        if i_ep % args.log_interval == 0:
            av = np.mean(score_l)
            md = np.median(score_l)
            std = np.std(score_l)
            min = np.min(score_l)
            max = np.max(score_l)

            if args.vis:
                draw_moving(xdata=i_ep, ydata=running_score)
                draw_mean(xdata=i_ep, ydata=av)
                draw_median(xdata=i_ep, ydata=md)
            if args.tb:
                writer.add_scalar("".join([now_d, "_","MovingAv", "_",  args.title]),
                                 running_score, i_ep)
                writer.add_scalar("".join([ now_d, "_","Mean", "_", args.title]),
                                  av, i_ep)
                writer.add_scalar("".join([ now_d, "_", "Median", "_",args.title]),
                                  md, i_ep)

            with open(csv_log, "a") as csvfile:
                csv.writer(csvfile).writerow([i_ep, av, md, std, min, max, running_score])

            print('Ep {}\tLast score: {:.2f}\tMoving average score: {:.2f}\tMean score: {:.2f}\tMedian score: {:.2f}'.format(i_ep, score, running_score, av, md))
            agent.save_param()
            score_l = []
        if running_score > env.reward_threshold and args.terminate:
            print("Solved! Running reward is now {} and the last episode runs to {}!".format(running_score, score))
            break
