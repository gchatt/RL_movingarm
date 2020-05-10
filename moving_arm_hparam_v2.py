import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
from numpy import random
import copy
import pickle
#import pygame
import datetime
import threading

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
from tensorboard.plugins.hparams import api as hp
#L1UNITS = hp.HParam('layer_1_units',hp.Discrete([50,100,150,200,250,300,350,400]))
L1UNITS = hp.HParam('layer_1_units',hp.Discrete([100]))

#L2UNITS = hp.HParam('layer_2_units',hp.Discrete([25,50,75,100,150,200,250,300]))
L2UNITS = hp.HParam('layer_2_units',hp.Discrete([50,300]))

NLAYER = hp.HParam('number_of_layers',hp.Discrete([1,2]))
#ALPHA = hp.HParam('alpha',hp.RealInterval(0.3,1.5))
ALPHA = hp.HParam('alpha',hp.RealInterval(0.3,0.8))
STD_G = hp.HParam('std_g',hp.Discrete([1,3,5]))

#VAL_SCALE = hp.HParam('value_scale',hp.Discrete([1,5,10]))
VAL_SCALE = hp.HParam('value_scale',hp.Discrete([1]))

#MAXSTEPR = hp.HParam('max_steps_real',hp.Discrete([200,500,1000]))
MAXSTEPR = hp.HParam('max_steps_real',hp.Discrete([1000]))

#UPDATE_FREQ = hp.HParam('update_frequency',hp.Discrete([20,50,100,200,500]))
UPDATE_FREQ = hp.HParam('update_frequency',hp.Discrete([100]))

#TAU_S = hp.HParam('target_critic_update_rate',hp.RealInterval(0.005,0.5))
TAU_S = hp.HParam('target_critic_update_rate',hp.RealInterval(0.05,0.2))

PRE_NOISE = hp.HParam('store_pre_noise',hp.Discrete([0,1]))

METRIC_ACCURACY = 'accuracy'
#maxgrad
#learning rates

class moving_arm_env(threading.Thread):

	def __init__(self,hparams):
		super(moving_arm_env, self).__init__()
		self.hparams = hparams;
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
		self.max_obj_amounts = 20; #max amount you can have for objects
		
		self.n_step = 0;
		self.n_session = 0;
		
		self.max_steps_real = hparams[MAXSTEPR]; #max steps per session once agent out of exploration phase
		self.max_sessions = 1000; #number of sessions before stopping
		self.max_steps_init = 10000; #max number of steps during the initialization / exploration stage
		self.max_steps = self.max_steps_init; #max steps per session
		self.max_explore_sessions = 10000;
		
		self.c_arm_pos = np.zeros((1,self.num_arms));
		#self.arm_lengths = np.ones((1,self.num_arms)); #can make settable in the future
		self.arm_lengths = np.array([[1,0.75,0.5]])
		self.arm_length = 50; #make settable
		
		self.last_action = [[0,0,0]];
		
		self.action_choices = [-1,0,1]; #not active
		self.max_action = 1; #not active
		
		self.tolerance = 100.;
		
		self.backup_reward_array = [];
		
		self.fr_force_scale = 200.;
		
		self.update_freq = hparams[UPDATE_FREQ]; #how often to run the 'train' function
		
		self.neg_val_scale = 10;#how much negative reward touching the 'edge' brings
		self.edge_tolerance = 30; #how close to the edge counts as touching the edge
		
		state_size = 3+self.num_arms+self.num_arms*self.num_choices; #color, current arm position, last action
		
		current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S%f")
		param_str = str({h.name: hparams[h] for h in hparams})
		log_dir = os.getcwd()+'\\logs\\' + current_time
		self.summary_writer = tf.summary.create_file_writer(log_dir)
		
		with open(log_dir+'\\header.txt','w') as header_file:
			header_file.write(param_str)
		
		self.agent = Agent_1(self.num_arms,self.num_choices,state_size,hparams,self.summary_writer)
		
		self.avg_reward = 0;
	
	def run(self):
		hp.hparams(self.hparams)
		self.gen_new_env();
		self.start_round();
		with self.summary_writer.as_default():
			tf.summary.scalar(METRIC_ACCURACY,self.avg_reward,step=self.n_session)
	
	def gen_new_env(self):
		for c in range(self.num_colors):
			color = [random.randint(0,255),random.randint(0,255),random.randint(0,255)]; #get a color
			rewards = [];
			for n in range(self.num_rewards_per_color):
				reward_position = [];
				for x in range(self.num_arms):
					reward_position.append(random.randint(self.min_arm_position,self.max_arm_position)); #choose positions that are rewarded for that color
				print(reward_position)
				reward_object_vector = [];
				max_amount = self.max_obj_amounts; #max amount you can have of an object
				for x in range(self.num_reward_objects):
					reward_object_vector.append(random.randint(0,max_amount)); #assign rewards vectors to the positions rewarded for that color
				rewards.append([reward_position,reward_object_vector]); #add a tuple to the array	
			self.reward_array.append([color,rewards]);
	
	def save_env(self,moving_arm_env):
		with open(moving_arm_env,'wb') as f: #give filename to save as
			pickle.dump(self.reward_array);
			
	def load_env(self,moving_arm_env):
		with open(moving_arm_env,'rb') as f: #give filename to open
			self.reward_array = pickle.load(f);
			
	def start_round(self):
		self.backup_reward_array = copy.deepcopy(self.reward_array); #reward array can be altered if answers are found; store back up
		
		#initialize round
		self.n_step = 0;
		self.n_session = 0;
		self.current_color = self.reward_array[0][0]; #color is first array in set of arrays; next are positions; next are rewards

		
		self.agent.init_state(np.array(self.current_color)/255.,np.array(self.c_arm_pos)/self.pos_range)
		done = False
		while not done:
			#Detect session length; in exploration phase session length is longer
			if self.n_session > self.max_explore_sessions:
				self.max_steps = self.max_steps_real; #go to actual session length
			
			#Take action
			action = self.agent.take_action();
			action = np.divide(action,self.fr_force_scale)
			#Take step in environment based on chosen action
			cp, cc, reward, done, done_2 = self.step(action[0]);
			#Update agent about reward and new state
			self.agent.store_update(np.array(self.current_color)/255.,np.array(self.c_arm_pos)/self.pos_range,reward);
			
			#Update weights
			if self.n_step % self.update_freq == 0: 
				self.avg_reward = self.agent.train(self.n_step + self.max_steps*self.n_session);
			#reset if session over
			if(done_2):
				self.c_arm_pos = np.zeros((1,self.num_arms)); 
				self.agent.c_arm_pos = np.zeros((1,self.num_arms));
				self.agent.last_action = [[0,0,0,0,0,0]];
				self.agent.last_action_tensor = tf.convert_to_tensor(np.array(self.agent.last_action,dtype='float32'));
				self.agent.memory.clear();
	
	def step(self,action):
		done = False
		done_2 = False
		self.n_step += 1;
		self.move_arm(action);
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
	
	def check_reward(self):
		total_reward = np.zeros((1,self.num_reward_objects));
		max_pos = np.array([self.max_arm_position,self.max_arm_position,self.max_arm_position]);
		min_pos = np.array([self.min_arm_position,self.min_arm_position,self.min_arm_position]);
		max_dist = np.sum(np.abs(max_pos-min_pos))
		for r in self.reward_array:
			if self.current_color == r[0]:
				i = 0;
				for p in r[1]: #positions for this color
					pos = np.array(p[0]); #make position a numpy array
					#dist = np.linalg.norm(self.c_arm_pos-pos,ord=1); #l1 norm
					dist = np.sum(np.abs(self.c_arm_pos-pos))
					if dist < self.tolerance:
						total_reward += ((1 - dist/max_dist) ** 3) * np.array(r[1][i][1]); #reward is discounted in relation to distance from optimal position
					# if dist == 0: #if it hits the spot perfectly, this position is removed from reward pool; it is no longe rewarding
						# r[2][i] = [0 for z in range(self.num_reward_objects)]
					i += 1;
		e_dist = np.sum(np.abs(max_pos - np.abs(self.c_arm_pos)));
		tol = self.edge_tolerance
		if e_dist <=tol:
			total_reward = self.neg_val_scale*np.negative(np.ones((1,self.num_reward_objects)));
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

class Actor(keras.Model):
	def __init__(self,state_sz,action_sz,nlayer,l1units,l2units):
		super(Actor,self).__init__();
		
		self.action_sz = action_sz;
		self.state_sz = state_sz;
		self.nlayer = nlayer
		self.l1 = layers.Dense(l1units,activation='relu');
		if self.nlayer == 2:
			self.l2 = layers.Dense(l2units,activation='relu');
		#self.drop1 = layers.Dropout(0.1);
		self.lout = layers.Dense(action_sz,activation='sigmoid');
		#self.gn = layers.GaussianNoise(stddev=2);
		self(tf.convert_to_tensor([np.zeros(state_sz,dtype='float32')]));
		
	def call(self,inputs):
		x = self.l1(inputs);
		#x = self.drop1(x);
		if self.nlayer == 2:
			x = self.l2(x);
		action = self.lout(x);
		#action_noised = self.gn(action)
		return action
		
	def layerweights(self):
		if self.nlayer == 2:
			return self.l1.get_weights(), self.l2.get_weights(), self.lout.get_weights()
		else:
			return self.l1.get_weights(), self.lout.get_weights()
			
		
class Critic(keras.Model):
	def __init__(self,state_sz,action_sz):
		super(Critic,self).__init__();
		self.state_sz = state_sz;
		self.action_sz = action_sz;
		self.l1 = layers.Dense(64,activation='relu');
		self.l2 = layers.Dense(300,activation='relu');
		self.lout = layers.Dense(1,activation='relu');
		self(tf.convert_to_tensor([np.zeros(state_sz,dtype='float32')]),tf.convert_to_tensor([np.zeros(action_sz,dtype='float32')]))
		
	def call(self,state,action):
		inputs = layers.concatenate([state,action]);
		x = self.l1(inputs);
		#x = self.l2(x);
		q_val = self.lout(x);
		return q_val
		
#model free agent using DDPG		
class Agent_1:

	def __init__(self,n_arms,n_choices,state_sz,hparams,summary_writer):
		self.n_arms = n_arms;
		self.n_choices = n_choices;
		self.state_sz = state_sz;
		self.action_sz = n_arms*n_choices;
		
		self.current_color = [];
		self.c_arm_pos = [];
		self.last_action = [[0,0,0,0,0,0]];
		self.last_action_tensor = tf.convert_to_tensor(np.array(self.last_action,dtype='float32'));
		
		self.total_reward = 0;
		self.memory = Memory()
		self.gamma = 0.99; #cumulative reward discount rate
		self.total_loss = 0;
		self.avg_reward = 0;
		
		#
		self.nlayer = hparams[NLAYER];
		self.l1units = hparams[L1UNITS];
		self.l2units = hparams[L2UNITS];
		tf.keras.backend.set_learning_phase(1) #sets 'trainable' mode on all models
		self.actor = Actor(self.state_sz,self.action_sz,self.nlayer,self.l1units,self.l2units)
		self.actor_target = Actor(self.state_sz,self.action_sz,self.nlayer,self.l1units,self.l2units)
		lr_actor = 0.001; #hyperparameter
		self.actor_opt = tf.keras.optimizers.Adam(learning_rate=lr_actor);
		#print(self.actor_target.weights)
		self.update_target_variables(self.actor_target.weights, self.actor.weights,1.0);
		
		#
		self.critic = Critic(self.state_sz,self.action_sz);
		self.critic_target = Critic(self.state_sz,self.action_sz);
		lr_critic = 0.001; #hyperparameter
		self.critic_opt = tf.keras.optimizers.Adam(learning_rate=lr_actor);
		self.update_target_variables(self.critic_target.weights, self.critic.weights,1.0);
		self.max_grad = 10; #used by huber loss function. hyperparameter
		#self.tau_s = 0.005 #original value
		self.tau_s = hparams[TAU_S] #hyper parameter. We need to try higher. #this is how much the target critic updates from the primary critic
		
		self.max_fr = 200.;
		
		self.last_state = [];
		
		self.n_train = 0;
		self.n_action = 0;
		self.store_pre_noise = hparams[PRE_NOISE] #If true (1), then the 'last action tensor' passed to the actor is the action pre gaussian noise. Sees to perform poorly when True
		self.actor_loss = 0;
		self.alpha = hparams[ALPHA]; #hyperparameter for how the gaussian std_dev is chosen. High alpha means that variability will drop fast as the actor gets better. 1.5 is too high
		self.std_g = hparams[STD_G]; #std dev for gaussian noise on action. hyperparameter
		
		self.value_scale = hparams[VAL_SCALE]; #how much to scale rewards by. Hyperparameter.
		
		#initialize log
		# current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S%f")
		# print({h.name: hparams[h] for h in hparams})
		# log_dir = os.getcwd()+'\\logs\\' + current_time
		# self.summary_writer = tf.summary.create_file_writer(log_dir)
		self.summary_writer = summary_writer;
	
	def init_state(self,current_color,c_arm_pos):
		self.current_color = tf.convert_to_tensor(np.array([current_color],dtype='float32')); #bracket current color to get right shape, then convert to float via numpy, then to tensor
		self.c_arm_pos = tf.convert_to_tensor(c_arm_pos);
		
		
	def take_action(self):
		self.n_action += 1;
		inputs = layers.concatenate([self.current_color,self.c_arm_pos,self.last_action_tensor])
		self.last_state = copy.deepcopy(inputs); #to use for memory storage later
		action = self.actor.call(inputs)
		if self.store_pre_noise == 1:
			self.last_action_tensor = copy.deepcopy(action); #store here if store_pre_noise is true. This theoretically may allow the actor to get richer information; intended action and the actual position the arm went to (c_arm_pos)
		if self.actor_loss > 0:
			self.std_g = 0.1 + (self.std_g - 0.1) / (np.abs(self.actor_loss)**self.alpha + 1);
		gn = layers.GaussianNoise(stddev=self.std_g)
		action = gn(action);
		if self.store_pre_noise == 0:
			self.last_action_tensor = copy.deepcopy(action); #store this for generating the next input state
		action_fr = tf.math.multiply(self.max_fr,action); #action in firing rates
		with self.summary_writer.as_default():
			tf.summary.scalar('action1',self.last_action_tensor[0][0],step=self.n_action)
			tf.summary.scalar('action2',self.last_action_tensor[0][1],step=self.n_action)
			tf.summary.scalar('action3',self.last_action_tensor[0][2],step=self.n_action)
			tf.summary.scalar('action4',self.last_action_tensor[0][3],step=self.n_action)
			tf.summary.scalar('action5',self.last_action_tensor[0][4],step=self.n_action)
			tf.summary.scalar('action6',self.last_action_tensor[0][5],step=self.n_action)
		return action_fr
	
	def store_update(self,current_color,c_arm_pos,reward):
		self.total_reward += reward;
		self.current_color = tf.convert_to_tensor(np.array([current_color],dtype='float32'));
		self.c_arm_pos = tf.convert_to_tensor(c_arm_pos);
		new_state = layers.concatenate([self.current_color,self.c_arm_pos,self.last_action_tensor]);
		self.memory.store(self.last_state,new_state,self.last_action_tensor,reward); #s, s', a, r; a in s -> s' and r
		with self.summary_writer.as_default():
			tf.summary.scalar('last reward',self.valuation(reward),step=self.n_action)
	
	def valuation(self,reward):
		sum = 0;
		values = [0.5,0.8,0.9]
		for i in range(len(reward)):
			sum += values[i]*reward[0][i]*self.value_scale
		return sum
		
	def train(self,steps):
		self.n_train += 1;
		self.avg_reward = self.valuation(self.total_reward) / steps;
		
		huber = tf.keras.losses.Huber(delta=self.max_grad);
		post_mem = tf.convert_to_tensor(np.vstack(self.memory.poststates),dtype=tf.float32)
		pre_mem = tf.convert_to_tensor(np.vstack(self.memory.prestates),dtype=tf.float32)
		mem_actions = tf.convert_to_tensor(np.vstack(self.memory.actions),dtype=tf.float32)
		v = [];
		for r in self.memory.rewards:
			v.append(self.valuation(r));
		mem_rewards = tf.convert_to_tensor(np.vstack(v),dtype=tf.float32)
		#seems to be working so far
		with tf.GradientTape() as tape:
			
			target_Q = self.critic_target(post_mem,self.actor_target(post_mem)); #for each post_state, and each action on the post state using the target policy, find the Q value
			target_Q = mem_rewards + tf.math.multiply(target_Q,self.gamma);
			target_Q = tf.stop_gradient(target_Q) #don't run through the target Q weights, just use these values without gradient tape looking at how they came about
			current_Q = self.critic(pre_mem,mem_actions);
			#current Q values minus the target Q values; target Q values are closer to real because they include the reward received (and then the predicted target cumulative reward)
			#td_errors = (target_Q - current_Q)**2; #can try squaring
			td_errors = huber(target_Q,current_Q)**2;
			#print(td_errors)
			critic_loss = tf.reduce_mean(td_errors); #mean
		#update the critic weights (not the critic target weights)
		#print(td_errors);
		#print(critic_loss);
		#print(tape.gradient(y,x))
		critic_grad = tape.gradient(critic_loss,self.critic.trainable_variables)
		#print(tf.shape(critic_grad))
		self.critic_opt.apply_gradients(zip(critic_grad,self.critic.trainable_variables));
		
		with tf.GradientTape() as tape:
			next_actions = self.actor(pre_mem);
			#print(next_actions);
			#gradient ascent, using the critic that was just updated
			self.actor_loss = -tf.reduce_mean(self.critic(pre_mem,next_actions));
		
		actor_grad = tape.gradient(self.actor_loss,self.actor.trainable_variables);
		self.log(self.actor_loss)
		#print(actor_loss)
		#print(tf.math.reduce_max(actor_grad[0]))
		#print(tf.math.reduce_max(actor_grad[1]))
		#input('wait')
		self.actor_opt.apply_gradients(zip(actor_grad,self.actor.trainable_variables));
		
		
		self.update_target_variables(self.critic_target.weights,self.critic.weights,self.tau_s);
		self.update_target_variables(self.actor_target.weights,self.actor.weights,self.tau_s);
		
		self.memory.clear();
		return self.avg_reward
		
	def update_target_variables(self,target_variables,source_variables,tau,use_locking=False,name="update_target_variables"):
		def update_op(target_variable, source_variable, tau):
			if tau == 1.0:
				return target_variable.assign(source_variable, use_locking)
			else:
				return target_variable.assign(
					tau * source_variable + (1.0 - tau) * target_variable, use_locking)

		# with tf.name_scope(name, values=target_variables + source_variables):
		#print(target_variables)
		update_ops = [update_op(target_var, source_var, tau)
					  for target_var, source_var
					  in zip(target_variables, source_variables)]
		return tf.group(name="update_all_variables", *update_ops)
	
	def log(self,actor_loss):
		
		if self.nlayer == 2:
			l1w,l2w,l3w = self.actor.layerweights();
		else:
			l1w,l3w = self.actor.layerweights();
			
		
		with self.summary_writer.as_default():
			tf.summary.scalar('actor loss',actor_loss,step=self.n_train)
			#tf.summary.scalar(METRIC_ACCURACY,self.avg_reward,step=self.n_train)
			tf.summary.histogram('actor layer1 weights',l1w[0],step=self.n_train)
			if self.nlayer == 2:
				tf.summary.histogram('actor layer2 weights',l2w[0],step=self.n_train)
			tf.summary.histogram('actor layer3-out weights',l3w[0],step=self.n_train)




		
		

#Start script here
mv_envs = [];
#env_threads = [];
n = 0;
trials = 1;
with tf.summary.create_file_writer(os.getcwd()+'\\logs').as_default():
	hp.hparams_config(
		hparams=[L1UNITS,L2UNITS,NLAYER,ALPHA,STD_G,VAL_SCALE,MAXSTEPR,UPDATE_FREQ,TAU_S,PRE_NOISE],
		metrics=[hp.Metric(METRIC_ACCURACY, display_name='Reward')],
	)
for a in L1UNITS.domain.values:
	for b in NLAYER.domain.values:
		for c in L2UNITS.domain.values:
			for d in (ALPHA.domain.min_value,ALPHA.domain.max_value):
				for e in STD_G.domain.values:
					for f in VAL_SCALE.domain.values:
						for g in MAXSTEPR.domain.values:
							for h in UPDATE_FREQ.domain.values:
								for i in (TAU_S.domain.min_value,TAU_S.domain.max_value):
									for j in PRE_NOISE.domain.values:
										for t in range(trials):
											hparams = {
														L1UNITS: a,
														L2UNITS: c,
														NLAYER: b,
														ALPHA: d,
														STD_G: e,
														VAL_SCALE: f,
														MAXSTEPR: g,
														UPDATE_FREQ: h,
														TAU_S: i,
														PRE_NOISE: j,
													}
											mv_envs.append(moving_arm_env(hparams));
											mv_envs[n].start();
											n += 1;
											print(n)

