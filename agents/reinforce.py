import math
import random
import shutil

import gym
import torch
from tensorboardX import SummaryWriter
from torch.backends import cudnn
from tqdm import tqdm

import logging

from graphs.losses.huber_loss import HuberLoss
from graphs.models.reinforce import Reinforce
from utils.env_utils import CartPoleEnv
from utils.misc import print_cuda_statistics
from utils.full_history import HistoryMemory, Transition

import numpy as np
class REINFORCEAgent:

    def __init__(self, config):
        self.config = config

        self.logger = logging.getLogger("REINFORCEAgent")

        # define models (policy and target)
        self.policy_model = Reinforce(self.config)
        # define memory
        self.history=HistoryMemory(self.config)

        # define optimizer
        self.optim = torch.optim.Adam(self.policy_model.parameters())

        # define environment
        self.env = gym.make('CartPole-v0').unwrapped
        self.cartpole = CartPoleEnv(self.config.screen_width)

        # initialize counter
        self.current_episode = 0
        self.current_iteration = 0
        self.episode_durations = []

        self.batch_size = self.config.batch_size

        # set cuda flag
        self.is_cuda = torch.cuda.is_available()
        if self.is_cuda and not self.config.cuda:
            self.logger.info("WARNING: You have a CUDA device, so you should probably enable CUDA")

        self.cuda = self.is_cuda & self.config.cuda

        if self.cuda:
            self.logger.info("Program will run on *****GPU-CUDA***** ")
          #  print_cuda_statistics()
            self.device = torch.device("cuda")
            torch.cuda.set_device(self.config.gpu_device)
        else:
            self.logger.info("Program will run on *****CPU***** ")
            self.device = torch.device("cpu")

        self.policy_model = self.policy_model.to(self.device)
    


        # Summary Writer
        self.summary_writer = SummaryWriter(log_dir=self.config.summary_dir, comment='REINFORCE')

    def load_checkpoint(self, file_name):
        filename = self.config.checkpoint_dir + file_name
        try:
            self.logger.info("Loading checkpoint '{}'".format(filename))
            checkpoint = torch.load(filename)

            self.current_episode = checkpoint['episode']
            self.current_iteration = checkpoint['iteration']
            self.policy_model.load_state_dict(checkpoint['state_dict'])
            self.optim.load_state_dict(checkpoint['optimizer'])

            self.logger.info("Checkpoint loaded successfully from '{}' at (epoch {}) at (iteration {})\n"
                  .format(self.config.checkpoint_dir, checkpoint['episode'], checkpoint['iteration']))
        except OSError as e:
            self.logger.info("No checkpoint exists from '{}'. Skipping...".format(self.config.checkpoint_dir))
            self.logger.info("**First time to train**")

    def save_checkpoint(self, file_name="checkpoint.pth.tar", is_best=0):
        state = {
            'episode': self.current_episode,
            'iteration': self.current_iteration,
            'state_dict': self.policy_model.state_dict(),
            'optimizer': self.optim.state_dict(),
        }
        # Save the state
        torch.save(state, self.config.checkpoint_dir + file_name)
        # If it is the best copy it to another file 'model_best.pth.tar'
        if is_best:
            shutil.copyfile(self.config.checkpoint_dir + file_name,
                            self.config.checkpoint_dir + 'model_best.pth.tar')

    def run(self):
        """
        This function will the operator
        :return:
        """
        try:
            self.train()

        except KeyboardInterrupt:
            self.logger.info("You have entered CTRL+C.. Wait to finalize")

    def select_action(self, state):
        """
        The action selection function, it either uses the model to choose an action or samples one uniformly.
        :param state: current state of the model
        :return:
        """
        if self.cuda:
            state = state.cuda()
        sample = random.random()

        self.current_iteration += 1
        action_prob = self.policy_model(state).squeeze(0) #return the action probability
        if self.cuda:
            action_prob=action_prob.cpu()
        action = np.random.choice(2, p=np.squeeze(action_prob.detach().numpy()))
        log_prob = torch.log(action_prob.squeeze(0)[action])
        return action, log_prob

    

    def compute_gradient(self):
        
        """
        performs a single step of optimization for the policy model
        :return:
        
        """
        transitions = self.history.memory
        one_batch=Transition(*zip(*transitions))
        

        # concatenate all batch elements into one
        
        reward_batch = torch.cat(one_batch.reward)  # [128]
        log_probs=one_batch.log_probs
        discounted_rewards = []
        '''

        https://medium.com/@thechrisyoon/deriving-policy-gradients-and-implementing-reinforce-f887949bd63

        '''

        for t in range(len(reward_batch)):
            Gt = 0 
            pw = 0
            for r in reward_batch[t:]:
                Gt = Gt + self.config.gamma**pw * r
                pw = pw + 1
            discounted_rewards.append(Gt)

        discounted_rewards = torch.tensor(discounted_rewards)
        discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-9) # normalize discounted rewards
        print(discounted_rewards)
        policy_gradient = []
        for log_prob, Gt in zip(log_probs, discounted_rewards):
            policy_gradient.append(-log_prob * Gt)

    

        # optimizer step
        self.optim.zero_grad()
        loss = torch.stack(policy_gradient).sum()
        loss.backward()
        self.optim.step()

        return loss

    def train(self):
        """
        Training loop based on the number of episodes
        :return:
        """
        for episode in tqdm(range(self.current_episode, self.config.num_episodes)):
            self.current_episode = episode
            
            # reset environment
            self.env.reset()
            self.train_one_epoch()
            
            #Reset History
            self.history=HistoryMemory(self.config)
          #  break

        self.env.render()
        self.env.close()

    def train_one_epoch(self):
        """
        One episode of training; it samples an action, observe next screen and optimize the model once
        :return:
        """
        episode_duration = 0
        prev_frame = self.cartpole.get_screen(self.env)
        curr_frame = self.cartpole.get_screen(self.env)
        # get state
        curr_state = curr_frame - prev_frame
        MAX_SEQUENCE=20000
        
        
        while(1):
            
            episode_duration += 1
            # select action
            action,log_prob = self.select_action(curr_state)
            # perform action and get reward
            _, reward, done, _ = self.env.step(action)

                
            if self.cuda:
                reward = torch.Tensor([reward]).to(self.device)
            else:
                reward = torch.Tensor([reward]).to(self.device)

            prev_frame = curr_frame
            curr_frame = self.cartpole.get_screen(self.env)
            # assign next state
            if done:
                next_state = None
            else:
                next_state = curr_frame - prev_frame

            # add this transition into memory
            self.history.push_transition(curr_state, action, next_state, reward,log_prob)

            curr_state = next_state


            if done:
                break
            if episode_duration==MAX_SEQUENCE:
                break
        #Policy Update
        curr_loss=self.compute_gradient()
        if curr_loss is not None:
            if self.cuda:
                     curr_loss = curr_loss.cpu()
            self.summary_writer.add_scalar("Policy Loss", curr_loss.detach().numpy(), self.current_episode)
        self.summary_writer.add_scalar("Training Episode Duration", episode_duration, self.current_episode)

    def validate(self):
        pass

    def finalize(self):
        """
        Finalize all the operations of the 2 Main classes of the process the operator and the data loader
        :return:
        """
        self.logger.info("Please wait while finalizing the operation.. Thank you")
        self.save_checkpoint()
        self.summary_writer.export_scalars_to_json("{}all_scalars.json".format(self.config.summary_dir))
        self.summary_writer.close()