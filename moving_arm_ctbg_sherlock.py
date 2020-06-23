import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
from numpy import random
import copy
import pickle
import datetime
from collections import deque
import CTBG_sherlock as CTBG
import threading
from tensorboard.plugins.hparams import api as hp

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

verbose = False
GUI = False
if GUI:
	import pygame
linux = True
manual_exit = True
if linux:
	manual_exit = False
#use_target_networks = True

manual_hparams = []
#Output from the agent is scaled by max_firing rate; this number is a relation between firing rate of the output neuron and how much the arm moves at that time_step. > 40 is too high without noise.
#Lower values mean more movement of the arm (can substitute for high noise?)
FORCE_SCALE = hp.HParam('force_scale',hp.Discrete([1]))
manual_hparams.append(FORCE_SCALE)


#How often to the run the 'train' subroutine
UPDATE_FREQ = hp.HParam('update_freq',hp.Discrete([10]))
manual_hparams.append(UPDATE_FREQ)

#How far away from the rewarded position you can be to get some scaled reward
TOLERANCE = hp.HParam('tolerance',hp.Discrete([100]))
manual_hparams.append(TOLERANCE)

#Number of steps in each session. Previously it has been found that this number needs to be > 1000 for proper exploration before a reset
MAX_STEPS = hp.HParam('max_steps',hp.Discrete([5]))
manual_hparams.append(MAX_STEPS)

#How many sessions before stopping the program
MAX_SESSIONS = hp.HParam('max_sessions',hp.Discrete([1000000]))
manual_hparams.append(MAX_SESSIONS)

#CDIV = how much to power the color vector; 255 means fully normalized; smaller values means color vector has higher magnitude
CDIV = hp.HParam('cdiv',hp.Discrete([255]))
manual_hparams.append(CDIV)

#VAL_SCALE = how high the reward is valued. scalar. High values mean higher valued reward. Idea here is that maybe this drives larger gradients?
VAL_SCALE = hp.HParam('val_scale',hp.Discrete([1]))
manual_hparams.append(VAL_SCALE)

#Learning rate for CTBG object in main agent
LR_CTBG = hp.HParam('lr_ctbg',hp.Discrete([0.001]))
#LR_CTBG_DRAW = 1
manual_hparams.append(LR_CTBG)

#Learning rate of critic (model free) in main agent. If too high, may not be as generalizable for various positions
LR_CRITIC = hp.HParam('lr_critic',hp.Discrete([0.001]))
#LR_CRITIC_DRAW = 1
manual_hparams.append(LR_CRITIC)
#Gamma = discount paramater when calculating Q value in critic network
#lower values are better for the simple task of just getting to one location because it is not a very multistep process
GAMMA = hp.HParam('gamma',hp.Discrete([0.05]))
#GAMMA_DRAW = 1
manual_hparams.append(GAMMA)


#striatum division size (multiply by 3 for total size)
UNIT_1 = hp.HParam('unit_1',hp.Discrete([100]))
manual_hparams.append(UNIT_1)

#other nuclei division size (multiply by 3 for total size)
UNIT_2 = hp.HParam('unit_2',hp.Discrete([100]))
manual_hparams.append(UNIT_2)

#premotor and motor cortex layer size
UNIT_3 = hp.HParam('unit_3',hp.Discrete([1000]))
manual_hparams.append(UNIT_3)

#critic units
CRITIC_UNITS = hp.HParam('critic_units',hp.Discrete([1000]))
manual_hparams.append(CRITIC_UNITS)

#Noise parameters
#noise = max(min(std_mc - self.noise_scale*(loss - self.tau),self.std_mc_init),noise_base)
#min of noise 1, max noise of the initial noise term (e.g. 90)
#loss is TDE (raw)
TAU = hp.HParam('tau',hp.Discrete([0.1]))
manual_hparams.append(TAU)
STD_MC = hp.HParam('std_mc',hp.Discrete([90]))
manual_hparams.append(STD_MC)
NOISE_SCALE = hp.HParam('noise_scale',hp.Discrete([0.009]))
manual_hparams.append(NOISE_SCALE)
NOISE_SCALE_2 = hp.HParam('noise_scale_2',hp.Discrete([0.008]))
manual_hparams.append(NOISE_SCALE_2)
NOISE_BASE = hp.HParam('noise_base',hp.Discrete([3.0]))
manual_hparams.append(NOISE_BASE)

#TDE scale. Scales the target_Q vs. current_Q. TDE = target_Q * tde_scale - current_Q
#Positive values give some bias to reward vs. prediction
TDE_SCALE = hp.HParam('tde_scale',hp.Discrete([1.02]))
manual_hparams.append(TDE_SCALE)

#Relevant for when the model free agent is trying to reach the MB agents goal. What success rate defines 'meeting the goal' and allows the MB agent to move on?
#100 out of the last 200 moves? You have to take into account that some goals cannot be reached in one step; so there are intermediary steps; this ratio cannot be too high due to that
#Keeping the goal needed somewhat low; 10 seems to be sufficient for showing that system has learned
#Currently, 'met goal' means you are either in the bin or 1 unit away
GOALS_MET_THRESH_1 = hp.HParam('goals_met_thresh_1',hp.Discrete([10]))
GOALS_MET_THRESH_2 = hp.HParam('goals_met_thresh_2',hp.Discrete([200]))
manual_hparams.append(GOALS_MET_THRESH_1)
manual_hparams.append(GOALS_MET_THRESH_2)

#Relevant for when the model free agent is trying to reach the MB agents goal.
#These terms define the reward the MF agent critic receives based on how close it is to the MB agents goal position
#There is no real reason to make this sparse...since it is internal
#GOAL_DIST_1; distances less than this number are rewarded; distances above are not
#using an L1 distance
#reward function -> reward = reward_base * exp(-L1_dist / dist_scale)
#-np.exp((dist - thresh_1)/neg_dist_scale) to give negative reward
GOAL_DIST_1 = hp.HParam('goal_dist_1',hp.Discrete([20]))
DIST_SCALE = hp.HParam('dist_scale',hp.Discrete([5]))
NEG_DIST_SCALE = hp.HParam('neg_dist_scale',hp.Discrete([100]))
REWARD_BASE = hp.HParam('reward_base',hp.Discrete([100]))
manual_hparams.append(GOAL_DIST_1)
manual_hparams.append(DIST_SCALE)
manual_hparams.append(NEG_DIST_SCALE)
manual_hparams.append(REWARD_BASE)

NUM_TO_FIX = hp.HParam('num_to_fix',hp.Discrete([10,100]))
manual_hparams.append(NUM_TO_FIX)



METRIC_ACCURACY = 'accuracy'

class moving_arm_env(threading.Thread):

	def __init__(self,hparams):
		super(moving_arm_env, self).__init__()
		self.hparams = hparams
	
		self.colors = [];
		self.reward_array = []; #[[[color],[[position,reward],[position2,reward2],..]],]
		
		self.current_color = [];
		self.num_colors = 1; #default 1, but this should be modifiable in future
		self.min_arm_position = -90;
		self.max_arm_position = 90;
		self.pos_range = 90;
		self.num_arms = 3; #default 3
		self.num_choices = 2; #move forward or backwards
		self.num_rewards_per_color = 1; #default 2
		self.num_reward_objects = 3; #how many different objects you can get. default 3
		
		
		self.n_step = 0;
		self.n_session = 0;
		
		self.max_steps = self.hparams[MAX_STEPS];
		self.max_sessions = self.hparams[MAX_SESSIONS];
		
		self.c_arm_pos = np.zeros((1,self.num_arms));
		#self.arm_lengths = np.ones((1,self.num_arms)); #can make settable in the future
		self.arm_lengths = np.array([[1,0.75,0.5]])
		self.arm_length = 50; #make settable
		
		self.last_action = [[0,0,0]];
		
		self.tolerance = float(self.hparams[TOLERANCE]);
		#print(self.tolerance)
		
		self.backup_reward_array = [];
		
		#self.fr_force_scale = 10.;
		self.fr_force_scale = float(self.hparams[FORCE_SCALE]);
		
		self.update_freq = self.hparams[UPDATE_FREQ]
		
		#min dist away the rewarded position can be from the edge
		#prevents the reward from being right at edge of max/min position
		#must be an integer
		self.edge = 10
		
		self.current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S%f")
		param_str = str({h.name: hparams[h] for h in hparams})
		if linux:
			log_dir = '/scratch/users/gchatt/logs/' + self.current_time
			self.summary_writer = tf.summary.create_file_writer(log_dir)
			
			with open(log_dir+'/header.txt','w') as header_file:
				header_file.write(param_str)
		else:
			log_dir = os.getcwd()+'\\logs\\' + self.current_time
			self.summary_writer = tf.summary.create_file_writer(log_dir)
			
			with open(log_dir+'\\header.txt','w') as header_file:
				header_file.write(param_str)
		
		self.last_reward = 0;
	
	def run(self):
		self.gen_new_env();
		self.start_round();
		with self.summary_writer.as_default():
			hp.hparams(self.hparams)
			tf.summary.scalar(METRIC_ACCURACY,self.last_reward,step=1)
		
	
	def gen_new_env(self):
		for c in range(self.num_colors):
			color = [random.randint(0,255),random.randint(0,255),random.randint(0,255)]; #get a color
			rewards = [];
			for n in range(self.num_rewards_per_color):
				reward_position = [];
				for x in range(self.num_arms):
					reward_position.append(random.randint(self.min_arm_position+self.edge,self.max_arm_position-self.edge)); #choose positions that are rewarded for that color
				reward_object_vector = [];
				max_amount = 20; #max amount you can have of an object
				min_amount = 10;
				for x in range(self.num_reward_objects):
					reward_object_vector.append(random.randint(min_amount,max_amount)); #assign rewards vectors to the positions rewarded for that color
				rewards.append([reward_position,reward_object_vector]); #add a tuple to the array	
			self.reward_array.append([color,rewards]);
	
	def save_env(self,moving_arm_env):
		with open(moving_arm_env,'wb') as f: #give filename to save as
			pickle.dump(self.reward_array);
			
	def load_env(self,moving_arm_env):
		with open(moving_arm_env,'rb') as f: #give filename to open
			self.reward_array = pickle.load(f);
			
	def start_round(self):
		#self.max_steps = max_steps; #can default as 100
		#self.max_sessions = max_sessions; #has to be less than number of colors. #default as 1
		self.backup_reward_array = copy.deepcopy(self.reward_array); #reward array can be altered if answers are found; store back up
		
		#initialize round
		self.n_step = 0
		self.n_step_total = 0
		self.n_session = 0
		self.current_color = self.reward_array[0][0] #color is first array in set of arrays; next are positions; next are rewards



		
		#agent = Agent_MB(self.min_arm_position,self.max_arm_position,self.num_arms,self.num_reward_objects)
		agent = Agent_3(self.min_arm_position,self.max_arm_position,self.num_arms,self.num_reward_objects,self.summary_writer,self.hparams)
		#set it's label (current time)
		agent.current_time = self.current_time
		
		#255.0 for now; this means full normalization of the color term
		cdiv = float(self.hparams[CDIV])
		if GUI:
			#init pygame
			h,w,border = 500,500,0;
			pygame.init()
			screen = pygame.display.set_mode((w+(2*border), h+(2*border)))
			running = True;
			c_color = (self.current_color[0],self.current_color[1],self.current_color[2])
			screen.fill(c_color);
			pygame.draw.lines(screen,(0,0,0),False,self.get_pygame_arm_points(self.c_arm_pos),5)
			done = False
			black = (0,0,0)
			gray = (120,120,120)
			while running:
				for event in pygame.event.get():
					if event.type == pygame.QUIT:
						running = False
				if not done:
				
					#Main action will be here
					action = agent.act(self.c_arm_pos[0],np.array([self.current_color])/cdiv);
					action = np.divide(action,self.fr_force_scale)
					#cp, cc, reward, done, done_2 = self.mb_step(goal);
					cp, cc, reward, done, done_2 = self.step(action[0]);
					agent.update_memory(self.c_arm_pos[0],np.array([self.current_color])/cdiv,reward)
					#agent.store_update(np.array(self.current_color)/255.,np.array(self.c_arm_pos)/self.pos_range,reward);
					if self.n_step_total % self.update_freq == 0:
						#print(self.n_step_total)
						agent.train();
					self.last_reward = agent.last_reward_avg
					#reset
					if(done_2):
						self.c_arm_pos = np.zeros((1,self.num_arms)); 
						#agent.c_arm_pos = np.zeros((1,self.num_arms));
						agent.last_action = [[0,0,0,0,0,0]];
						agent.last_action_tensor = tf.convert_to_tensor(np.array(agent.last_action,dtype='float32'));
						#agent.memory.clear();
					

					
					#update info / screen
					c_color = (self.current_color[0],self.current_color[1],self.current_color[2])
					screen.fill(c_color);
					p = self.get_pygame_arm_points(self.c_arm_pos);
					pygame.draw.lines(screen,black,False,p,5)
					#fix, that first [0] to [n_session]
					for x in self.reward_array[0][1]:
						rew_pos = x[0] #reward position in degrees
						rew_pts = self.get_pygame_arm_points([rew_pos]) #reward position in x,y coords
						
						pygame.draw.lines(screen,gray,False,rew_pts,2)
					pygame.display.update();
				else:
					self.end_round();
		#No GUI
		else:
			done = False
			while not done:
				action = agent.act(self.c_arm_pos[0],np.array([self.current_color])/cdiv);
				action = np.divide(action,self.fr_force_scale)
				cp, cc, reward, done, done_2 = self.step(action[0]);
				#self.avg_reward += reward / self.update_freq
				agent.update_memory(self.c_arm_pos[0],np.array([self.current_color])/cdiv,reward)
				if self.n_step_total % self.update_freq == 0: 
					agent.train()
				
				self.last_reward = agent.last_reward_avg
				#reset
				if(done_2):
					self.c_arm_pos = np.zeros((1,self.num_arms)); 
					#agent.c_arm_pos = np.zeros((1,self.num_arms));
					agent.last_action = [[0,0,0,0,0,0]];
					agent.last_action_tensor = tf.convert_to_tensor(np.array(agent.last_action,dtype='float32'));
					#agent.memory.clear();
			self.end_round();
				
	
	
	def mb_step(self,goal):
		done = False
		done_2 = False
		self.n_step += 1;
		self.n_step_total += 1
		self.move_arm_to(goal);
		reward = self.check_reward();
		if np.amax(reward) < 0:
			done_2 = True
		if self.n_step >= self.max_steps:
			self.n_step = 1;
			self.n_session += 1;
			done_2 = True
			
			if self.n_session >= self.max_sessions:
				self.n_session -= 1;
				done = True
		return self.c_arm_pos, self.current_color, reward, done, done_2
	
	def step(self,action):
		done = False
		done_2 = False
		self.n_step += 1;
		self.n_step_total += 1
		self.move_arm(action);
		reward = self.check_reward();
		#reward is negative if gets too close to boundary. Reset it.
		if np.amax(reward) < 0:
			done_2 = True
		if self.n_step >= self.max_steps:
			self.n_step = 1;
			self.n_session += 1;
			done_2 = True
			
			if self.n_session >= self.max_sessions:
				self.n_session -= 1;
				done = True
				#self.end_round(); #consider ending the round internally in the step function
			#else:
				#self.c_arm_pos = np.zeros((1,self.num_arms)); #reset arm position #in the future I will remove this to test out ability to move from anywhere
				#self.current_color = self.reward_array[n_session][0]; #IMPORTANT. FIX THIS LATER
		return self.c_arm_pos, self.current_color, reward, done, done_2
	
	def move_arm(self,action):
		pos_action = np.array([action[0],action[2],action[4]]);
		neg_action = np.array([action[1],action[3],action[5]]);
		#fix
		self.c_arm_pos = self.c_arm_pos + pos_action; #add numpy arrays #position in unit degrees; degree representing the position on a circle
		self.c_arm_pos = self.c_arm_pos - neg_action;
		self.c_arm_pos = np.clip(self.c_arm_pos,self.min_arm_position,self.max_arm_position); #array, min, max. Clip the values so the arm doesn't swing all the way around
	
	def move_arm_to(self,goal):
		self.c_arm_pos = copy.copy(goal)
	
	def check_reward(self):
		total_reward = np.zeros((1,self.num_reward_objects));
		#print(total_reward)
		max_pos = np.array([self.max_arm_position,self.max_arm_position,self.max_arm_position]);
		min_pos = np.array([self.min_arm_position,self.min_arm_position,self.min_arm_position]);
		max_dist = np.sum(np.abs(max_pos-min_pos))
		for r in self.reward_array:
			#print(r)
			if self.current_color == r[0]:
				i = 0;
				for p in r[1]: #positions for this color
					pos = np.array(p[0]); #make position a numpy array
					#dist = np.linalg.norm(self.c_arm_pos-pos,ord=1); #l1 norm
					dist = np.sum(np.abs(self.c_arm_pos-pos))
					#print(dist)
					#print(np.sum(np.abs(self.c_arm_pos-pos)))
					#print(self.c_arm_pos)
					#print(max_dist)
					#input('wait')
					#dist,max_dist = self.get_circle_distance(self.c_arm_pos,pos)
					#print('distance'+str(dist));
					if dist < self.tolerance:
						total_reward += ((1 - dist/max_dist) ** 3) * np.array(r[1][i][1]); #reward is discounted in relation to distance from optimal position
						#print(dist/max_dist)
						#print(total_reward);
						#print(np.sum(total_reward))
						#input('wait')
					# if dist == 0: #if it hits the spot perfectly, this position is removed from reward pool; it is no longe rewarding
						# r[2][i] = [0 for z in range(self.num_reward_objects)]
					i += 1;
		#If too close to the boundary, then reset (otherwise it will often get stuck there) and give a small negative reward
		boundary_dist = np.sum(np.abs(max_pos - np.abs(self.c_arm_pos)));
		boundary_punishment = 0.01 #scales the negative reward
		boundary_tolerance = 0
		if boundary_dist <= boundary_tolerance:
			total_reward = np.negative(np.ones((1,self.num_reward_objects)))*boundary_punishment;
		return total_reward
		
	def end_round(self):
		self.n_step = 0;
		self.n_session = 0;
		self.c_arm_pos = np.zeros((1,self.num_arms));
		self.reward_array = copy.deepcopy(self.backup_reward_array);# return the reward array to the back up stored at the start of the round
	
	def get_pygame_arm_points(self,pos):
		degrconv = np.pi / 180.;
		offset = np.array([250,250])
		points = [];
		points.append(offset)
		for x in range(self.num_arms):
			h = self.arm_lengths[0][x]*self.arm_length #get arm length for point
			p = np.array([h*np.sin(pos[0][x]*degrconv),h*np.cos(pos[0][x]*degrconv)]); #use the degree value in c_arm_pos to get the x,y coordinates
			if x == 0:
				points.append(copy.deepcopy(p+offset));
			elif x > 0:
				p = p + np.array(points[x]); #x,y coords adjusted based on which joint in arm
				points.append(copy.deepcopy(p));
		return points
		
		
#####

class Memory:
	def __init__(self):
		self.prestates = []
		self.poststates = []
		self.actions = []
		self.rewards = []
	
	def store(self,prestate,poststate,action,reward):
		self.prestates.append(prestate)
		self.poststates.append(poststate)
		self.actions.append(action)
		self.rewards.append(reward)
	
	def clear(self):
		self.prestates = []
		self.poststates = []
		self.actions = []
		self.rewards = []

class Memory_CTBG:
	def __init__(self):
		self.prestrstates = []
		self.poststrstates = []
		self.prepremstates = []
		self.postpremstates = []
		self.actions = []
		self.rewards = []
	
	def store(self,prestrstate,prepremstate,poststrstate,postpremstate,action,reward):
		self.prestrstates.append(prestrstate)
		self.poststrstates.append(poststrstate)
		self.prepremstates.append(prepremstate)
		self.postpremstates.append(postpremstate)
		self.actions.append(action)
		self.rewards.append(reward)
	
	def clear(self):
		self.prestrstates = []
		self.poststrstates = []
		self.prepremstates = []
		self.postpremstates = []
		self.actions = []
		self.rewards = []

class Actor(keras.Model):
	def __init__(self,state_sz,action_sz):
		super(Actor,self).__init__();
		self.action_sz = action_sz;
		self.state_sz = state_sz;
		self.l1 = layers.Dense(100,activation='relu');
		self.l2 = layers.Dense(50,activation='relu');
		#self.drop1 = layers.Dropout(0.1);
		self.lout = layers.Dense(action_sz,activation='sigmoid');
		#self.gn = layers.GaussianNoise(stddev=2);
		self(tf.convert_to_tensor([np.zeros(state_sz,dtype='float32')]));
		
	def call(self,inputs):
		x = self.l1(inputs);
		#x = self.drop1(x);
		x = self.l2(x);
		action = self.lout(x);
		#print(action)
		#action_noised = self.gn(action)
		#print(action_noised)
		return action
		
	def layerweights(self):
		return self.l1.get_weights(), self.l2.get_weights(), self.lout.get_weights()
	
# OLD	
# class Critic(keras.Model):
	# def __init__(self,state_sz,action_sz):
		# super(Critic,self).__init__();
		# self.state_sz = state_sz;
		# self.action_sz = action_sz;
		# self.l1 = layers.Dense(64,activation='relu');
		# self.l2 = layers.Dense(300,activation='relu');
		# self.lout = layers.Dense(1,activation='relu');
		# self(tf.convert_to_tensor([np.zeros(state_sz,dtype='float32')]),tf.convert_to_tensor([np.zeros(action_sz,dtype='float32')]))
		
	# def call(self,state,action):
		# inputs = layers.concatenate([state,action]);
		# x = self.l1(inputs);
		#x = self.l2(x);
		# q_val = self.lout(x);
		# return q_val
		
class Critic_CTBG(keras.Model):
    def __init__(self,hparams):
        super(Critic_CTBG,self).__init__()
        self.l1i = layers.Dense(hparams[UNIT_2],activation='relu')
        #self.l2i = layers.Dense(hparams[UNIT_2],activation='relu')
        self.l3i = layers.Dense(9,activation='relu')
        self.l1 = layers.Dense(hparams[UNIT_3],activation='relu');
        self.l2 = layers.Dense(hparams[UNIT_3],activation='relu');
        #self.l3 = layers.Dense(hparams[UNIT_3],activation='relu');
        self.bna = tf.keras.layers.BatchNormalization()
        self.bnb = tf.keras.layers.BatchNormalization()
        self.bnai = tf.keras.layers.BatchNormalization()
        #self.bnbi = tf.keras.layers.BatchNormalization()
        self.bnci = tf.keras.layers.BatchNormalization()
        #self.bnc = tf.keras.layers.BatchNormalization()
        self.lout = layers.Dense(1,activation='relu');

    def call(self,state,action,bnorm):
        y = self.l1i(state[:,0:9])
        y = self.bnai(y,training=bnorm)
        #y = self.l2i(y)
        #y = self.bnbi(y,training=bnorm)
        y = self.l3i(y)
        y = self.bnci(y,training=bnorm)
        inputs = layers.concatenate([y,state[:,9:15],action])
        x = self.l1(inputs);
        x = self.bna(x,training=bnorm)
        x = self.l2(x);
        x = self.bnb(x,training=bnorm)
        #x = self.l3(x)
        #x = self.bnc(x,training=bnorm)
        val = self.lout(x);
        #val = tf.clip_by_value(x,0,100)
        return val

class SG(keras.Model):
	def __init__(self,goal_sz):
		super(SG,self).__init__();
		self.l1 = layers.Dense(400,activation='relu')
		self.lout = layers.Dense(goal_sz,activation='relu')
	
	def call(self,state):
		x = self.l1(state)
		goal_logits = self.lout(x)
		return goal_logits
	
#currently only outputs goal arm position, not color. Color stays constant	
class SGS(keras.Model):
	def __init__(self,goal_sz):
		super(SGS,self).__init__();
		self.l1 = layers.Dense(400,activation='relu')
		self.lout = layers.Dense(goal_sz,activation='relu')
	
	def call(self,state,goal):
		input = layers.concatenate([state,goal])
		x = self.l1(input)
		state_logits = self.lout(x)
		return state_logits

class SGR(keras.Model):
	def __init__(self,reward_sz):
		super(SGR,self).__init__();
		self.l1 = layers.Dense(100,activation='relu')
		self.lout = layers.Dense(reward_sz,activation='relu') #vector, n of each reward object
	
	def call(self,state,goal):
		input = layers.concatenate([state,goal])
		x = self.l1(input)
		reward_pred = self.lout(x)
		return reward_pred

class Agent_MB:

	def __init__(self,min_arm_p,max_arm_p,n_arms,reward_sz,summary_writer,hparams):
		tf.keras.backend.set_learning_phase(1)
		self.hparams = hparams
		
		
		self.reward_sz = reward_sz #make sure this is 3
		
		self.memory = Memory()
		
		self.number_to_draw_g = 10; #10 is quite high #with num == 1, you sort of have a model free system, calculating a q function on the fly; as long as valuation doesn't change its fine to use this as model free
		self.number_to_draw_s = 1; #10 is quite high #1 is fine for now because this isn't a multi stage task
		self.discount = 0.3;
		self.max_depth = 1; #10 is too long. 3 takes a while too.
		
		self.min_arm_position = min_arm_p;
		self.max_arm_position = max_arm_p;
		self.num_arms = n_arms;
		self.bin_sz = 10.0;
		self.pos_scale = self.max_arm_position / self.bin_sz
		self.n_c = int(((self.max_arm_position - self.min_arm_position) / self.bin_sz) + 1)
		
		self.state_sz = self.n_c ** n_arms
		self.goal_sz = self.state_sz #here they are equivalent because we are keeping color constant. 19 * 19 * 19
		
		self.values = [0.5,0.8,0.9]
		self.val_scale = float(self.hparams[VAL_SCALE])
		
		self.state_to_goal = SG(self.goal_sz)
		self.reward_pred = SGR(self.reward_sz)
		self.new_state_pred = SGS(self.goal_sz)
		lr_sg = 0.1;
		lr_sgs = 0.1;
		lr_sgr = 0.5;
		self.sg_opt = tf.keras.optimizers.Adam(learning_rate=lr_sg);
		self.sgs_opt = tf.keras.optimizers.Adam(learning_rate=lr_sgs);
		self.sgr_opt = tf.keras.optimizers.Adam(learning_rate=lr_sgr);
		
		self.last_goal = [];
		self.last_reward_value = 0;
		
		self.last_few_goal_len = 10;
		
		self.last_few_goals = deque();
		
		self.met_goal = False
		self.met_goal_count = 0
		self.goal_attempt = 0
		
		self.n_check = 0
		self.summary_writer = summary_writer
	
	def act(self,c_arm_pos,current_color):
		self.met_goal = False
		binned_pos = self.full_pos_to_bins(c_arm_pos,self.bin_sz) / self.pos_scale
		#print(self.bin_pos_to_idx(binned_pos))
		self.current_state = layers.concatenate([current_color,binned_pos])
		self.last_state = copy.deepcopy(self.current_state); #to use for memory storage later
		goal_choice = self.max_goal(self.current_state,self.number_to_draw_g,self.max_depth)
		goal_pos = self.get_pos_from_choice(goal_choice,False) / self.pos_scale;
		#print(full_goal)
		#bin_pos_new = self.full_pos_to_bins(full_goal.numpy()[0],self.bin_sz) / self.pos_scale
		self.last_goal = goal_choice
		#print(self.last_goal)
		#print(self.last_few_goals)
		#full_goal = self.get_pos_from_choice(goal,True)
		return goal_pos
	
	def full_pos_to_bins(self,arm_pos,bin_sz):
		binned_pos = []
		for a in arm_pos:
			p = np.round(a / bin_sz)
			binned_pos.append(p)
		binned_pos = np.array([binned_pos],dtype='float32')
		return binned_pos
	
	def max_goal(self,state,number_to_draw_g,max_depth):
		goal_logits = self.state_to_goal(state) #given the 'binned' state, get a distribution on 'best' goals
		#print(np.max(goal_logits))
		gn = layers.GaussianNoise(stddev=0.01)
		goal_logits_noise = gn(goal_logits)
		#print(np.argmax(goal_logits))
		#print(np.argmax(goal_logits_noise))
		probs = tf.nn.softmax(goal_logits_noise).numpy()[0]
		#print(probs)
		goal_options = []
		reward_predictions = []
		current_depth = 1
		n = 0
		while n < number_to_draw_g:
			g = np.argmax(probs)
			#print(g)
			#print(probs[g])
			#print(goal_logits.numpy()[0][g])
			if g in self.last_few_goals:
				probs[g] = -1.0
				continue
			else:
				goal_options.append(g)
				reward_predictions.append(self.predict(state,g,current_depth,max_depth))
				probs[g] = -1.0
				n += 1
		max_goal = goal_options[np.argmax(reward_predictions)]
		#print('orig goal')
		#print(max_goal)
		# pg = get_pos_from_choice(max_goal,False);
		# if noised_goal > 0 and noised_goal < self.state_sz:
			# max_goal = noised_goal
		#print(np.argmax(reward_predictions))
		return max_goal
	
	#return a scalar, valuated reward
	def predict(self,state,goal,current_depth,max_depth):
		goal_pos = self.get_pos_from_choice(goal,False)
		goal_pos = goal_pos / self.pos_scale
		pred_reward = self.valuation(self.reward_pred(state,goal_pos))
		#could add a novelty reward here ^, but not clear how useful this will be
		if current_depth == max_depth:
			return pred_reward
		else:
			new_state_logits = self.new_state_pred(state,goal_pos) #based on this position, predict a new state (in this case just an arm pos, color is constant)
			probs = tf.nn.softmax(new_state_logits).numpy()[0]
			probs_draw = copy.deepcopy(probs)
			new_states = []
			for n in range(self.number_to_draw_s):
				s = np.argmax(probs_draw)
				new_states.append(s)
				probs_draw[s] = -1.0
			for s in new_states:
				new_state_pos = self.get_pos_from_choice(s,False)
				new_state_pos = new_state_pos / self.pos_scale
				#print(state)
				new_state = layers.concatenate([state[:,0:3],new_state_pos]) #state[:,0:3] = color of that state
				#print(new_state)
				#print(new_state[:,3:])
				goal_logits = self.state_to_goal(new_state) #new state, new goals
				probs_g = tf.nn.softmax(goal_logits).numpy()[0]
				g = np.argmax(probs_g) #assume you take the maximum goal in this new state #may not work
				pred_reward += probs[s]*(self.discount*self.predict(new_state,g,current_depth+1,max_depth))
			return pred_reward
			
	def get_pos_from_choice(self,choice,full):
		arm_pos = [];
		n_c = int(((self.max_arm_position - self.min_arm_position) / self.bin_sz) + 1)
		base = self.min_arm_position / self.bin_sz
		
		noise_std_base = 10.0
		alpha = 0.2
		noise_std = noise_std_base
		if self.last_reward_value > 0:
			noise_std = 0.1 + (noise_std_base - 0.1) / ((self.last_reward_value/self.val_scale)**alpha)
		#print(noise_std)
		a = np.floor(choice / (n_c * n_c))
		arm_pos.append(a+base)
		b = choice % (n_c * n_c)
		c = np.floor(b / n_c)
		arm_pos.append(c+base)
		d = b % n_c
		arm_pos.append(d+base)
		#if full, using some noise, give the full position within the range
		if full:
			for i in range(len(arm_pos)):
				arm_pos[i] *= self.bin_sz
				#arm_pos[i] += np.clip(np.random.normal(loc=0.0,scale=2.5),-5.0,5.0) #add noise, clip at 5 away from mean
				arm_pos[i] += np.random.normal(loc=0.0,scale=noise_std)
				arm_pos[i] = np.clip(arm_pos[i],self.min_arm_position,self.max_arm_position) #clip so not less than min or greater than max
		arm_pos_t = tf.convert_to_tensor(np.array([arm_pos],dtype='float32'))
		return arm_pos_t
	
	def valuation(self,reward):
		sum = 0;
		for i in range(len(reward[0])):
			sum += self.values[i]*reward[0][i]*self.val_scale
		return sum
	
	def bin_pos_to_idx(self,binned_pos):
		base = self.min_arm_position / self.bin_sz
		n_c = int(((self.max_arm_position - self.min_arm_position) / self.bin_sz) + 1)
		binned_pos *= self.pos_scale
		binned_pos -= base
		bp = binned_pos[0]
		idx = bp[0]*(n_c*n_c) + bp[1]*(n_c) + bp[2]
		return int(idx)
	
	def update_memory(self,c_arm_pos,current_color,reward):
		binned_pos = self.full_pos_to_bins(c_arm_pos,self.bin_sz) / self.pos_scale
		current_state = layers.concatenate([current_color,binned_pos])
		self.last_reward_value = self.valuation(reward);
		if self.met_goal and self.last_reward_value <= 0:
			self.last_few_goals.append(self.last_goal);
			if len(self.last_few_goals) > self.last_few_goal_len:
				self.last_few_goals.popleft()
		#print('last_goal')
		#print(self.last_goal)
		#print(self.last_few_goals)
		#print('reward')
		#print(self.last_reward_value)
		self.memory.store(self.last_state,current_state,self.last_goal,reward)
	
	def check_met_goal(self,goal,c_arm_pos):
		goal = goal * self.pos_scale
		binned_pos = self.full_pos_to_bins(c_arm_pos,self.bin_sz)
		self.n_check += 1
		self.goal_attempt += 1
		self.goal_thresh_1 = self.hparams[GOALS_MET_THRESH_1] #needs to be a hyperparameter
		self.goal_thresh_2 = self.hparams[GOALS_MET_THRESH_1] #hyperparam, tough one
		
		dist = np.sum(np.abs(binned_pos-goal)) #sum of the distance between the goal (binned) and the current position (binned)
		
		#print('dist')
		#print(dist)
		if verbose:
			print('in check goal')
			print(goal)
			print('binned pos')
			print(binned_pos)
			print('dist')
			print(dist)
		thresh_1 = self.hparams[GOAL_DIST_1]
		dist_scale = self.hparams[DIST_SCALE]
		neg_dist_scale = self.hparams[NEG_DIST_SCALE]
		thresh_2 = 1
		base = self.hparams[REWARD_BASE]
		reward = 0
		if dist < thresh_1:
			reward = base * np.exp(-dist/dist_scale)
			if dist <= thresh_2:
				self.met_goal_count += 1 #if within the bin, then count this as met goal
				#if it seems like you are meeting goal consistenty (ie 50% of the last 100 checks) then count this as met goal
				if self.met_goal_count >= self.goal_thresh_1:
					self.met_goal = True
					self.goal_attempt = 0
					self.met_goal_count = 0
					if verbose:
						print('met goal!')
		else:
			reward = -np.exp((dist - thresh_1)/(neg_dist_scale))
						
		with self.summary_writer.as_default():
			tf.summary.scalar('Distance to goal',dist,step=self.n_check)
			tf.summary.scalar('Met goal count',self.met_goal_count,step=self.n_check)

		if self.goal_attempt >= self.goal_thresh_2:
			self.goal_attempt = 0
			self.met_goal_count = 0
		
		return reward
	
	def train(self):
		pre_mem = tf.convert_to_tensor(np.vstack(self.memory.prestates),dtype=tf.float32)
		post_mem = tf.convert_to_tensor(np.vstack(self.memory.poststates),dtype=tf.float32)
		mem_goals = tf.convert_to_tensor(np.vstack(self.memory.actions),dtype=tf.float32)
		mem_rewards = tf.convert_to_tensor(np.vstack(self.memory.rewards),dtype=tf.float32)
		
		mgp = []
		for g in self.memory.actions:
			goal_pos = self.get_pos_from_choice(g,False)
			goal_pos = goal_pos / self.pos_scale
			mgp.append(goal_pos)
		mem_goal_pos = tf.convert_to_tensor(np.vstack(mgp),dtype=tf.float32)
		
		pmi = []
		for p in self.memory.poststates:
			b = p[:,3:] #strip color information
			pmi.append(self.bin_pos_to_idx(b))
		post_mem_goal_idx = tf.convert_to_tensor(pmi,dtype=tf.int32)
		
		##
		mse = tf.keras.losses.MeanSquaredError()

		with tf.GradientTape() as tape:
			pred_post_logits = self.new_state_pred(pre_mem,mem_goal_pos)
			sgs_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=post_mem_goal_idx,logits=pred_post_logits))
		sgs_grad = tape.gradient(sgs_loss,self.new_state_pred.trainable_variables)
		self.sgs_opt.apply_gradients(zip(sgs_grad,self.new_state_pred.trainable_variables))
		
		##
		with tf.GradientTape() as tape:
			pred_rewards = self.reward_pred(pre_mem,mem_goal_pos)
			sgr_loss = mse(mem_rewards, pred_rewards)
		sgr_grad = tape.gradient(sgr_loss,self.reward_pred.trainable_variables)
		self.sgr_opt.apply_gradients(zip(sgr_grad,self.reward_pred.trainable_variables))
		
		##
		nmgi = []
		nmgs = []
		n = 0;
		for s in self.memory.prestates:
			if self.valuation(self.memory.rewards[n]) > 0:
				nmgi.append(mem_goals[n][0])
				nmgs.append(copy.copy(s))
				#print(mem_goals[n][0])
				#print(self.valuation(self.memory.rewards[n]))
			n += 1
		
		max_draw_g = 500
		if len(nmgi) > 0:
			if self.number_to_draw_g < max_draw_g:
				self.number_to_draw_g += 5
			new_max_goals_idx = tf.convert_to_tensor(nmgi,dtype=tf.int32)
			new_max_goal_states = tf.convert_to_tensor(np.vstack(nmgs),dtype=tf.float32)
			
			with tf.GradientTape() as tape:
				goal_logits = self.state_to_goal(new_max_goal_states) #logits for each state
				sg_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=new_max_goals_idx,logits=goal_logits))
			sg_grad = tape.gradient(sg_loss,self.state_to_goal.trainable_variables)
			self.sg_opt.apply_gradients(zip(sg_grad,self.state_to_goal.trainable_variables))
		
		self.memory.clear();

class Agent_3:
	def __init__(self,min_arm_p,max_arm_p,n_arms,reward_sz,summary_writer,hparams):
		self.hparams = hparams
		self.memory = Memory_CTBG()
		
		tf.keras.backend.set_learning_phase(1)
	
		self.pfc = Agent_MB(min_arm_p,max_arm_p,n_arms,reward_sz,summary_writer,self.hparams)
		
		#units = [self.hparams[UNIT_1],self.hparams[UNIT_2],self.hparams[UNIT_3]]
		#print(units)
		ctbg_params = {}
		ctbg_params['UNIT_1'] = self.hparams[UNIT_1]
		ctbg_params['UNIT_2'] = self.hparams[UNIT_2]
		ctbg_params['UNIT_3'] = self.hparams[UNIT_3]
		ctbg_params['STD_MC'] = self.hparams[STD_MC]
		ctbg_params['TAU'] = self.hparams[TAU]
		ctbg_params['NOISE_SCALE'] = self.hparams[NOISE_SCALE]
		ctbg_params['NOISE_SCALE_2'] = self.hparams[NOISE_SCALE_2]
		ctbg_params['NOISE_BASE'] = self.hparams[NOISE_BASE]
		ctbg_params['NUM_TO_FIX'] = self.hparams[NUM_TO_FIX]
		
		self.ctbg = CTBG.CTBG(summary_writer,ctbg_params)
		lr_ctbg = self.hparams[LR_CTBG]
		#print(lr_ctbg)
		self.ctbg_opt = tf.keras.optimizers.Adam(learning_rate=lr_ctbg)
		
		self.critic = CTBG.Critic_CTBG_2(summary_writer,ctbg_params)
		lr_critic = self.hparams[LR_CRITIC]
		#print(lr_critic)
		self.critic_opt = tf.keras.optimizers.Adam(learning_rate=lr_critic)

		self.gamma = self.hparams[GAMMA] #discount parameter when calculating Q
		
		self.max_fr = 20.
		self.last_state = []
		self.max_grad = 10 #for huber loss
		
		self.goal = []
		self.last_action = [[0,0,0,0,0,0]]
		self.last_action_tensor = tf.convert_to_tensor(np.array(self.last_action,dtype='float32'))
		
		self.n_step = 0
		self.n_train = 0
		
		self.summary_writer = summary_writer
		self.c_arm_pos_new = []
		
		self.last_actor_loss = 0
		self.last_reward_avg = 0
        
		self.tde = 0
		self.use_reset_noise = True
		self.n_goal = 0
		self.bound = 0
		self.bound_div = 1.0
		
		self.max_critic_loss = 10000
		
		self.current_time = ''
		
		self.use_fix_weights = True	
		
	def act(self,c_arm_pos,current_color):
		#First stage is to just reach PFC goals, so try one goal at a time
		if self.n_step == 0 or self.pfc.met_goal: #question
			self.n_goal += 1
			self.bound = min(np.floor(self.n_goal/self.bound_div) / 9.0,1.0)
			if self.use_reset_noise and self.bound < 1.0:
				#self.ctbg.std_mc_init = self.hparams[STD_MC] * self.bound #doesn't work
				self.ctbg.reset_noise()
			if self.n_step > 0 and self.use_fix_weights:
				self.ctbg.fix_weights()
				self.critic.fix_weights()
			self.goal = self.pfc.act(c_arm_pos,current_color)
			self.goal = tf.clip_by_value(self.goal,-1.0*self.bound,self.bound)
			print(self.goal*self.pfc.pos_scale)
	
		c_arm_pos = np.array([c_arm_pos])
		c_arm_pos = self.expand_vector(c_arm_pos)
		goal = self.expand_vector(self.goal)

		str_inputs = layers.concatenate([goal,current_color,c_arm_pos,self.last_action_tensor])
		prem_inputs = layers.concatenate([goal,current_color,c_arm_pos])
		self.last_state = copy.deepcopy([str_inputs,prem_inputs])
		action = self.ctbg(str_inputs,prem_inputs,self.tde,use_noisy_relaxation=True,bnorm=False)

		self.last_action_tensor = copy.deepcopy(action)
		#action_fr = tf.math.multiply(self.max_fr,action)
		#action_fr = tf.clip_by_value(action,0,self.max_fr)
		#print(action_fr)
		
		self.n_step += 1
		
		if verbose:
			print('goal')
			print(self.goal * self.pfc.pos_scale)
			print('current arm position')
			print(c_arm_pos)
			print('action')
			print(action)
		
		#return action_fr
		return action
	
	def expand_vector(self,v):
		vnew = np.zeros((1,6))
		
		i = 0
		for p in v[0]:
			if p > 0:
				vnew[0][i] = np.abs(p)
			else:
				vnew[0][i+1] = np.abs(p)
			i += 2
			
		return vnew
	
	def update_memory(self,c_arm_pos,current_color,reward):
		total_reward = self.pfc.check_met_goal(self.goal,c_arm_pos)
		valuated_reward = self.pfc.valuation(reward)
		if self.pfc.met_goal and self.bound == 1.0:
			self.pfc.update_memory(c_arm_pos,current_color,reward)
			total_reward += valuated_reward
		# else:
			# if total_reward == 0 and valuated_reward < 0:
				# total_reward = valuated_reward
		#pfc_pred_reward = self.pfc.**
		self.last_reward = total_reward
		self.c_arm_pos_new = c_arm_pos
		goal = self.expand_vector(self.goal)
		self.log(1)
		c_arm_pos = np.array([c_arm_pos])
		c_arm_pos = self.expand_vector(c_arm_pos)

		new_str = layers.concatenate([goal,current_color,c_arm_pos,self.last_action_tensor]); #goal always stays constant from the .act function 
		new_prem = layers.concatenate([goal,current_color,c_arm_pos])
		self.memory.store(self.last_state[0],self.last_state[1],new_str,new_prem,self.last_action_tensor,total_reward); #<s_str, s_prem, s'_str, s'_prem, a, r> -> store
	
	def train(self):
		if self.pfc.met_goal and self.bound == 1.0:
			self.pfc.train()
		#
		decay_rate = -0.00001
		#print('here')
		huber = tf.keras.losses.Huber(delta=self.max_grad);
		pre_str_mem = tf.convert_to_tensor(np.vstack(self.memory.prestrstates),dtype=tf.float32)
		post_str_mem = tf.convert_to_tensor(np.vstack(self.memory.poststrstates),dtype=tf.float32)
		pre_prem_mem = tf.convert_to_tensor(np.vstack(self.memory.prepremstates),dtype=tf.float32)
		post_prem_mem = tf.convert_to_tensor(np.vstack(self.memory.postpremstates),dtype=tf.float32)
		#print(post_prem_mem.shape)
		mem_actions = tf.convert_to_tensor(np.vstack(self.memory.actions),dtype=tf.float32)

		mem_rewards = tf.convert_to_tensor(np.vstack(self.memory.rewards),dtype=tf.float32)
		
		self.last_reward_avg = tf.reduce_mean(mem_rewards)
		#print(self.last_reward_avg)
		
		#This may fail as it is ON TARGET. This system perhaps precludes CTBG containing internal noise because incorrect actions will be assoc w/ reward? The noise can be mild maybe
		target_Q = self.critic(post_prem_mem,self.ctbg(post_str_mem,post_prem_mem,0,False,bnorm=False),bnorm=False) #trying giving prem_state; 6/8/20


		#Critical. In the future, target_Q = weighted average of model free Q and model based Q from your PFC module. Also this can be 'tuned off' when turning down complexity
		target_Q = mem_rewards + tf.math.multiply(target_Q,self.gamma)
		#print('after')
		#print(target_Q)
		with tf.GradientTape() as tape:
			current_Q = self.critic(pre_prem_mem,mem_actions,bnorm=True)
			td_errors = huber(target_Q,current_Q)
			#print(tf.math.reduce_max(current_Q))
			#print(tf.math.reduce_max(target_Q))
			#print(tf.math.reduce_max(td_errors))
			#td_errors = (target_Q - current_Q)**2	
			self.critic_loss = tf.reduce_mean(td_errors)
		
		#self.critic_loss = tf.clip_by_value(self.critic_loss,0,self.max_critic_loss)
		critic_grad = tape.gradient(self.critic_loss,self.critic.trainable_variables)
		#print('critic grad')
		#print(critic_grad)
		#decay_vector = np.ones_like(critic_grad) * decay_rate
		#decay_vector = tf.convert_to_tensor(decay_vector,dtype=tf.float32)
		self.critic_opt.apply_gradients(zip(critic_grad,self.critic.trainable_variables))
		#self.critic_opt.apply_gradients(zip(decay_vector,self.critic.trainable_variables))

		tde_scale = self.hparams[TDE_SCALE]
		self.tde = target_Q*tde_scale - current_Q
		self.tde = tf.reduce_mean(self.tde)
		self.tde = min(self.tde,100) #to prevent catastrophically high TDE
		self.ctbg.update_noise(self.tde)
		
		with tf.GradientTape() as tape:
			next_actions = self.ctbg(pre_str_mem,pre_prem_mem,0,False,bnorm=True);
			#print(self.critic(pre_str_mem,next_actions))
			#gradient ascent, using the critic that was just updated
			#self.actor_loss = -tf.clip_by_value(tf.reduce_mean(self.critic(pre_prem_mem,next_actions,bnorm=False)),0,self.max_critic_loss)
			self.actor_loss = -tf.reduce_mean(self.critic(pre_prem_mem,next_actions,bnorm=False))
		
		#self.actor_loss = tf.clip_by_value(self.actor_loss,-self.max_critic_loss,self.max_critic_loss)
		self.last_actor_loss = self.actor_loss
		#print(self.ctbg.trainable_variables)
		self.actor_grad = tape.gradient(self.actor_loss,self.ctbg.trainable_variables)
		#decay_vector = np.ones_like(self.actor_grad) * decay_rate
		#decay_vector = tf.convert_to_tensor(decay_vector,dtype=tf.float32)		
		#print(tf.math.reduce_max(self.actor_grad[1]))
		self.ctbg_opt.apply_gradients(zip(self.actor_grad,self.ctbg.trainable_variables))
		#self.ctbg_opt.apply_gradients(zip(decay_vector,self.ctbg.trainable_variables))
		self.log(2)
		self.n_train += 1
		self.memory.clear()
		
		# if linux:
			# self.ctbg.save_weights('/scratch/users/gchatt/checkpoints/checkpoint-'+self.current_time+'/ctbg/ctbg_checkpoint')
			# self.critic.save_weights('/scratch/users/gchatt/checkpoints/checkpoint-'+self.current_time+'/critic/critic_checkpoint')
			# with open('/scratch/users/gchatt/checkpoints/checkpoint-'+self.current_time+'/log.txt','w') as logfile:
				# logfile.write('ngoal='+str(self.n_goal))
		# else:
			# self.ctbg.save_weights(os.getcwd()+'\\checkpoints\\checkpoint-'+self.current_time+'\\ctbg\\ctbg_checkpoint')
			# self.critic.save_weights(os.getcwd()+'\\checkpoints\\checkpoint-'+self.current_time+'\\critic\\critic_checkpoint')
			# with open(os.getcwd()+'\\checkpoints\\checkpoint-'+self.current_time+'\\log.txt','w') as logfile:
				# logfile.write('n_goal='+str(self.n_goal))
			
		#if self.use_fix_weights:
			#self.ctbg.fix_weights()
			#self.ctbg.decay()
			#self.critic.decay()
		
		
		
	def log(self,type):
		if type == 2:
			self.ctbg.log(self.n_train)
		
		with self.summary_writer.as_default():
			if type == 1:
				tf.summary.scalar('goal[0]',self.goal.numpy()[0][0]*self.pfc.pos_scale,step=self.n_step)
				tf.summary.scalar('goal[1]',self.goal.numpy()[0][1]*self.pfc.pos_scale,step=self.n_step)
				tf.summary.scalar('goal[2]',self.goal.numpy()[0][2]*self.pfc.pos_scale,step=self.n_step)
				tf.summary.scalar('arm position[0]',self.c_arm_pos_new[0],step=self.n_step)
				tf.summary.scalar('arm position[1]',self.c_arm_pos_new[1],step=self.n_step)
				tf.summary.scalar('arm position[2]',self.c_arm_pos_new[2],step=self.n_step)
				tf.summary.scalar('last reward',self.last_reward,step=self.n_step)
			elif type == 2:
				tf.summary.scalar('actor loss',self.actor_loss,step=self.n_train)
				tf.summary.scalar('last avg reward',self.last_reward_avg,step=self.n_train)
				#tf.summary.histogram('actor gradient',self.actor_grad[19],step=self.n_train)
				tf.summary.scalar('critic loss',self.critic_loss,step=self.n_train)
				tf.summary.scalar('raw TD error',self.tde,step=self.n_train)
				#tf.summary.histogram('critic weights',self.critic.trainable_weights[0],step=self.n_train)

			
		

#Start script here
mv_envs = [];
n = 0;
trials = 1;
if linux:
	with tf.summary.create_file_writer('/scratch/users/gchatt/logs').as_default():
		hp.hparams_config(
			hparams=[FORCE_SCALE,UPDATE_FREQ,TOLERANCE,MAX_STEPS,MAX_SESSIONS,CDIV,VAL_SCALE,LR_CTBG,LR_CRITIC,UNIT_1,UNIT_2,UNIT_3,TAU,STD_MC,GAMMA,GOALS_MET_THRESH_1,GOALS_MET_THRESH_2,GOAL_DIST_1,DIST_SCALE,NEG_DIST_SCALE,REWARD_BASE,NOISE_SCALE,NOISE_SCALE_2,NOISE_BASE,TDE_SCALE,NUM_TO_FIX],
			metrics=[hp.Metric(METRIC_ACCURACY, display_name='Reward')],
		)
else:
	with tf.summary.create_file_writer(os.getcwd()+'\\logs').as_default():
		hp.hparams_config(
			hparams=[FORCE_SCALE,UPDATE_FREQ,TOLERANCE,MAX_STEPS,MAX_SESSIONS,CDIV,VAL_SCALE,LR_CTBG,LR_CRITIC,UNIT_1,UNIT_2,UNIT_3,TAU,STD_MC,GAMMA,GOALS_MET_THRESH_1,GOALS_MET_THRESH_2,GOAL_DIST_1,DIST_SCALE,NEG_DIST_SCALE,REWARD_BASE,NOISE_SCALE,NOISE_SCALE_2,NOISE_BASE,TDE_SCALE,NUM_TO_FIX],
			metrics=[hp.Metric(METRIC_ACCURACY, display_name='Reward')],
		)

hparams = {}
hp_mult = []
hp_mult_num = []
for hyper_param in manual_hparams:
	if len(hyper_param.domain.values) == 1:
		hparams[hyper_param] = hyper_param.domain.values[0]
	elif len(hyper_param.domain.values) > 1:
		hp_mult.append(hyper_param)
		hp_mult_num.append(len(hyper_param.domain.values))

hp_mult_count = np.zeros(len(hp_mult))

if len(hp_mult) > 0:
	done = False
	n = 0
	num_trials = 1
	while not done:
		#check if at max
		#print(len(hp_mult))
		for i in range(len(hp_mult)):
			#print(i)
			#print(hp_mult_count[i])
			if int(hp_mult_count[i]+1) >= hp_mult_num[i]:
				done = True
			else:
				done = False
				break
				
		#continue and initiate threads
		for i in range(len(hp_mult)):
			hparams[hp_mult[i]] = hp_mult[i].domain.values[int(hp_mult_count[i])]
		for num_trial in range(num_trials):
			mv_envs.append(moving_arm_env(hparams))
			if not linux:
				mv_envs[n].daemon = True
			mv_envs[n].start()
			n += 1
			
		#if done from earlier check, break
		if done:
			break
		
		for i in range(len(hp_mult)):
			if i == 0:
				hp_mult_count[i] += 1
			if int(hp_mult_count[i]) >= hp_mult_num[i]:
				hp_mult_count[i] = 0
				if (i+1) != len(hp_mult):
					hp_mult_count[i+1] += 1
else:
	mv_envs.append(moving_arm_env(hparams))
	if not linux:
		mv_envs[0].daemon = True
	mv_envs[0].start()
	
	
if (not linux) and manual_exit:
	print('enter any input to exit')
	exit = input()
