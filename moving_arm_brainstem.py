##Step 1: Have 'brainstem' get some fixed weights so it can learn some cue / behavior pairs
##
##Goal, objectively quantifying the advantage that inhibitor / indirect / direct pathways provide to the motor system.

##To Do next:
#Debug, make sure it's working
#Set up tensorboard
#Save and load weights

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
import threading
from tensorboard.plugins.hparams import api as hp

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

verbose = True
GUI = False
linux = False
if GUI:
	import pygame
manual_exit = True
if linux:
	manual_exit = False

class cust_layer(layers.Layer):
	
    def __init__(self,input_dim,units):
        super(cust_layer,self).__init__()
        self.input_dim = input_dim
        self.units = units
        w_init = tf.random_uniform_initializer(minval=-0.001, maxval=0.001)
        b_init = tf.random_uniform_initializer(minval=-0.0, maxval=0.0)
        #decay_rate_init = tf.random_uniform_initializer(minval=decay_init[0], maxval=decay_init[1])
        self.w = self.add_weight(shape=(self.input_dim,self.units),initializer=w_init,trainable=True)
        self.b = self.add_weight(shape=(self.units,), initializer=b_init, trainable=True)
        #self.decay_rates = self.add_weight(shape=(self.input_dim,self.units),initializer=decay_rate_init,trainable=False)
        self.initial_weights = copy.deepcopy(self.w)
        #print(self.initial_weights)
        self.mask = np.ones_like(self.w.numpy())
        self.fixed_values = np.zeros_like(self.w.numpy())

    def call(self,inputs):
        masked_weights = tf.multiply(self.mask,self.w)
        w = masked_weights + self.fixed_values
        out = tf.tensordot(inputs,w,[[1],[0]]) + self.b
        return out

    #def decay(self): #NOT WORKING
        #vector in direction pointint from fixed_weights to initial_weights; take an uneven step in this direction based on decay rates
        #self.w = self.w - tf.multiply((self.w - self.initial_weights),self.decay_rates)
        #self.w = self.w - tf.zeros_like(self.w)
		
class Actor(keras.Model):
    def __init__(self):
        super(Actor,self).__init__()
        #hyperparams
        self.bs_units = 1000 #brainstem layer units
        self.use_batch_norm = True
        self.number_to_fix = 10 #weights to fix on each 'success'
        self.std_d = 90
        self.std_d_init = self.std_d
        self.tau = 0.1
        self.noise_scale = 0.009
        self.noise_scale_2 = 0.008
        self.noise_base = 3.0
        self.max_fr = 200
        self.use_noise = True
        #end hyperparams

        self.network = []
        self.network.append(cust_layer(9,self.bs_units))
        self.network.append(cust_layer(self.bs_units,self.bs_units))
        self.network.append(cust_layer(self.bs_units,6)) #6 = nlimbs * njoints * 2

        self.bna = tf.keras.layers.BatchNormalization()
        self.bnb = tf.keras.layers.BatchNormalization()
        self.n_step = 0
		
    def update_noise(self,tde):
        if self.std_d <= self.noise_base and tde > 0:
            self.std_d = max(self.std_d - self.noise_scale_2*(tde - self.tau),1.0)
        else:
            self.std_d = max(min(self.std_d - self.noise_scale*(tde - self.tau),self.std_d_init),self.noise_base)

    def reset_noise(self):
        self.std_d = self.std_d_init
	
    def fix_weights(self):
        gv = []
        #get the distance vectors for each layer
        for i in range(len(self.network)):
            gv.append(np.absolute((self.network[i].w - self.network[i].initial_weights).numpy()))

        for i in range(self.number_to_fix):
            choice = random.randint(len(self.network))
            gv_choice = gv[choice]
            max_ind = np.unravel_index(np.argmax(gv_choice,axis=None),gv_choice.shape)
            #print(gv_choice.shape)
            #print(max_ind)
            # max_ind is a tuple. Can use [] to query; can also go in gv[max_ind] like this to get the max value
            already_fixed = (self.network[choice].mask[max_ind] == 0) #Boolean. Is the mask already 0?
            while already_fixed:
                gv_choice[max_ind] = 0 #set the gv value to 0 so it is not picked again
                choice = random.randint(len(self.network))
                gv_choice = gv[choice]
                max_ind = np.unravel_index(np.argmax(gv_choice,axis=None),gv_choice.shape)
                already_fixed = (self.network[choice].mask[max_ind] == 0)

        # set mask to 0 (initialized as ones)
        self.network[choice].mask[max_ind] = 0
        # set the fixed value to the current weight
        self.network[choice].fixed_values[max_ind] = self.network[choice].w[max_ind]

    def call(self,inputs,use_noise,bnorm):
        self.gn = layers.GaussianNoise(stddev=self.std_d)
        out = self.network[0].call(inputs)
        out = tf.nn.relu(out)
        if self.use_batch_norm:
            out = self.bna(out,training=bnorm)

        out = self.network[1].call(out)
        out = tf.nn.relu(out)
        if self.use_batch_norm:
            out = self.bnb(out,training=bnorm)

        out = self.network[2].call(out)
        out = tf.nn.relu(out)
        if use_noise:
            out = self.gn(out)
        out = tf.clip_by_value(out,0,self.max_fr)

        self.n_step += 1

        return out

class Critic(keras.Model):
    def __init__(self):
        super(Critic,self).__init__()
        #hyperparams
        self.units = 1000
        self.number_to_fix = 10
        self.input_sz = 9
        self.action_sz = 6
        self.use_batch_norm = True

        self.critic_layers = []
        #color,c_pos,action
        self.critic_layers.append(cust_layer(self.input_sz+self.action_sz,self.units))
        self.critic_layers.append(cust_layer(self.units,self.units))
        self.critic_layers.append(cust_layer(self.units,1))
        self.bna = tf.keras.layers.BatchNormalization()
        self.bnb = tf.keras.layers.BatchNormalization()		

    def call(self,state,action,bnorm):
        inputs = layers.concatenate([state,action])
        out = self.critic_layers[0].call(inputs)
        out = tf.nn.relu(out)
        if self.use_batch_norm:
            out = self.bna(out,training=bnorm)
        
        out = self.critic_layers[1].call(out)
        out = tf.nn.relu(out)
        if self.use_batch_norm:
            out = self.bnb(out,training=bnorm)
        
        out = self.critic_layers[2].call(out)
        val = tf.nn.relu(out)
        
        return val
        
    def fix_weights(self):
        gv = []
        #get the distance vectors for each layer
        for i in range(len(self.critic_layers)):
            gv.append(np.absolute((self.critic_layers[i].w - self.critic_layers[i].initial_weights).numpy()))

        for i in range(self.number_to_fix):
            choice = random.randint(len(self.critic_layers))
            gv_choice = gv[choice]
            max_ind = np.unravel_index(np.argmax(gv_choice,axis=None),gv_choice.shape)
            already_fixed = (self.critic_layers[choice].mask[max_ind] == 0) #Boolean. Is the mask already 0?
            while already_fixed:
                gv_choice[max_ind] = 0 #set the gv value to 0 so it is not picked again
                choice = random.randint(len(self.critic_layers))
                gv_choice = gv[choice]
                max_ind = np.unravel_index(np.argmax(gv_choice,axis=None),gv_choice.shape)
                already_fixed = (self.critic_layers[choice].mask[max_ind] == 0)

        # set mask to 0 (initialized as ones)
        self.critic_layers[choice].mask[max_ind] = 0
        # set the fixed value to the current weight
        self.critic_layers[choice].fixed_values[max_ind] = self.critic_layers[choice].w[max_ind]

	
class Object():
    def __init__(self,utility=[],location=[],name=''):
        self.utility = utility
        self.location = location #x,y
        self.name = name

class Context():
    def __init__(self,color=[],objects=[],rfp=[]):
        self.color = color
        self.objects = objects
        #reward function parameters
        self.rfp = rfp #[threshold1,threshold2,scale]
        self.met_goal = 0
        self.tries = 0
        
        #hyperparam
        self.met_goal_criteria = 50
        self.max_tries = 200

    def query_reward(self,c_pos,nlimb,njoint):
        reward = 0
        threshold1 = self.rfp[0]
        threshold2 = self.rfp[1]
        scale = self.rfp[2]
        for o in self.objects:
            for l in range(nlimb):
                pos = c_pos[l][njoint] #needs to be converted to x,y of distal portion of limb
                dist = np.linalg.norm(o.location - pos)
                if dist <= threshold1:
                    reward += o.utility*np.exp(-dist/scale)
                    if dist < threshold2 and o.utility > 0:
                        self.met_goal += 1
                elif dist > threshold1 and o.utility < 0:
                    self.met_goal += 1
        self.tries += 1
        return reward
    
    def check_met_goal(self):
        met = False
        if self.tries <= self.max_tries:
            if self.met_goal >= self.met_goal_criteria:
                met = True
        else:
            self.met_goal = 0
            self.tries = 0
        return met
        
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
        
class Agent():
    def __init__(self,summary_writer,hparams=[]):
        self.hparams = hparams
        self.summary_writer = summary_writer
        #hyperparams
        self.n_limbs = 1
        self.n_joints = 3
        self.min_position = -90
        self.max_position = 90
        self.limb_ratios = np.array([[1,1,0.3]]) #set for just one limb
        self.limb_lengths = np.array([[50]]) #again, just one limb here, but can add more
        self.limb_offsets = np.array([[0,0]]) #[[x_limb1,y_limb1],..] (fixed point of a limb)
        self.cdiv = 255 #for normalizing the color vector
        self.max_grad = 10 #for huber loss
        self.lr_actor = 0.001
        self.lr_critic = 0.001
        self.gamma = 0.01 #discount parameter
        self.max_critic_loss = 10000
        self.tde_scale = 1.02
        self.max_tde = 100
        #end hyperparams
        self.actor = Actor()
        self.actor_opt = tf.keras.optimizers.Adam(learning_rate=self.lr_actor)
        self.critic = Critic()
        self.critic_opt = tf.keras.optimizers.Adam(learning_rate=self.lr_critic)
        
        #needed for using noise
        tf.keras.backend.set_learning_phase(1)

        self.memory = Memory()
        
        self.pre_state = []
        self.post_state = []
        self.action = []
        
        self.actor_loss = 0
        self.critic_loss = 0
        self.tde = 0
        self.last_reward_avg = 0
        self.n_train = 0
        self.n_step = 0

        self.c_pos = np.zeros((self.n_limbs,self.n_joints)) #current position of all limbs, joints		
        self.total_limb_length = []
        for i in range(self.n_limbs):
            for x in range(self.n_joints):
                l = self.limb_ratios[i][0]*self.limb_lengths[i]+self.limb_ratios[i][1]*self.limb_lengths[i]+self.limb_ratios[i][2]*self.limb_lengths[i]
                self.total_limb_length.append([l])
        self.total_limb_length = np.array(self.total_limb_length)
	
    def act(self,context):
        self.n_step += 1
        color = np.array([context.color])/self.cdiv
        pos = self.expand_vector(self.c_pos)
        self.pre_state = layers.concatenate([color,pos])
        #feed forward pass through the actor (in this case 'brainstem') with contextual input
        self.action = self.actor(self.pre_state,use_noise=True,bnorm=True)
        self.move_arm(self.action) #updates c_pos
        new_pos = self.expand_vector(self.c_pos)
        self.post_state = layers.concatenate([color,new_pos])
        self.log_act()
        
    def move_arm(self,action):
        action = action[0]
        pos_action = np.array([action[0],action[2],action[4]]);
        neg_action = np.array([action[1],action[3],action[5]]);
        #fix
        self.c_pos = self.c_pos + pos_action; #add numpy arrays #position in unit degrees; degree representing the position on a circle
        self.c_pos = self.c_pos - neg_action;
        self.c_pos = np.clip(self.c_pos,self.min_position,self.max_position); #array, min, max. Clip the values so the arm doesn't swing all the way around

    def update_memory(self,reward):
        self.memory.store(self.pre_state,self.post_state,self.action,reward)
        
    def neutral(self):
        self.c_pos = np.zeros((self.n_limbs,self.n_joints))
        
    #A vector is doubled in size where element pairs (1,2) (3,4) etc are positive/negative    
    def expand_vector(self,v):
        vnew = np.zeros((1,len(v[0])*2))
        i = 0
        for p in v[0]:
            if p > 0:
                vnew[0][i] = np.abs(p)
            else:
                vnew[0][i+1] = np.abs(p)
            i += 2
        return vnew
    
    def train(self):
        huber = tf.keras.losses.Huber(delta=self.max_grad)
        pre_mem = tf.convert_to_tensor(np.vstack(self.memory.prestates),dtype=tf.float32)
        post_mem = tf.convert_to_tensor(np.vstack(self.memory.poststates),dtype=tf.float32)
        mem_actions = tf.convert_to_tensor(np.vstack(self.memory.actions),dtype=tf.float32)
        mem_rewards = tf.convert_to_tensor(np.vstack(self.memory.rewards),dtype=tf.float32)
        self.last_reward_avg = tf.reduce_mean(mem_rewards)

        target_Q = self.critic(post_mem,self.actor(post_mem,use_noise=False,bnorm=False),bnorm=False)
        target_Q = mem_rewards + tf.math.multiply(target_Q,self.gamma)
        
        with tf.GradientTape() as tape:
            current_Q = self.critic(pre_mem,mem_actions,bnorm=True) #bnorm only true here because these were real pairs that occurred
            td_errors = huber(target_Q,current_Q)
            self.critic_loss = tf.clip_by_value(tf.reduce_mean(td_errors),0,self.max_critic_loss)	
        critic_grad = tape.gradient(self.critic_loss,self.critic.trainable_variables)
        self.critic_opt.apply_gradients(zip(critic_grad,self.critic.trainable_variables))
        self.tde = target_Q*self.tde_scale - current_Q
        self.tde = tf.reduce_mean(self.tde)
        self.tde = min(self.tde,self.max_tde)
        self.actor.update_noise(self.tde)
        
        with tf.GradientTape() as tape:
            next_actions = self.actor(pre_mem,use_noise=False,bnorm=False)
            self.actor_loss = -tf.reduce_mean(self.critic(pre_mem,next_actions,bnorm=False))
        
        self.actor_grad = tape.gradient(self.actor_loss,self.actor.trainable_variables)
        self.actor_opt.apply_gradients(zip(self.actor_grad,self.actor.trainable_variables))

        self.n_train += 1
        self.log_train()
        self.memory.clear()
    
    def log_act(self):
        with self.summary_writer.as_default():
            tf.summary.scalar('arm position[0]',self.c_pos[0][0],step=self.n_step)
            tf.summary.scalar('arm position[1]',self.c_pos[0][1],step=self.n_step)
            tf.summary.scalar('arm position[2]',self.c_pos[0][2],step=self.n_step)
    
    def log_train(self):
        with self.summary_writer.as_default():
            tf.summary.scalar('actor loss',self.actor_loss,step=self.n_train)
            tf.summary.scalar('last avg reward',self.last_reward_avg,step=self.n_train)
            tf.summary.scalar('critic loss',self.critic_loss,step=self.n_train)
            tf.summary.scalar('raw TD error',self.tde,step=self.n_train)
            tf.summary.scalar('noise',self.actor.std_d,step=self.n_train)

class Environment():
    def __init__(self,hparams=[]):
        #super(moving_arm_env, self).__init__()
        self.hparams = hparams
        #hyperparams
        self.n_contexts = 10
        self.n_objects = 1 #n_objects per context
        self.env_h = 500
        self.env_w = 500
        #self.max_obj_dist = 100
        self.min_utility = 20
        self.max_utility = 30
        self.scale_ratio_max = 20
        self.scale_ratio_min = 5
        self.threshold_ratio_max = 3
        self.threshold_ratio_min = 2
        self.threshold_2 = 10 #distance for meeting criteria of meeting goal
        self.cdiv = 255
        self.update_freq = 10
        self.max_steps = 5
        self.max_sessions = 1000000
        #end hyperparams

        self.current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S%f")
        #param_str = str({h.name: hparams[h] for h in hparams})
        if linux:
            log_dir = '/scratch/users/gchatt/logs/' + self.current_time
            self.summary_writer = tf.summary.create_file_writer(log_dir)
            #with open(log_dir+'/header.txt','w') as header_file:
                #header_file.write(param_str)
        else:
            log_dir = os.getcwd()+'\\logs\\' + self.current_time
            self.summary_writer = tf.summary.create_file_writer(log_dir)
            #with open(log_dir+'\\header.txt','w') as header_file:
                #header_file.write(param_str)

        self.agent = Agent(self.summary_writer)
        self.contexts = []
        #current context index
        self.cc_idx = 0
        self.first_round = True #if haven't completed all contexts yet


	
    def start(self):
        self.n_step = 0
        self.n_step_total = 0
        self.n_session = 0 #session += 1 whenever the limbs reset to neutral pos
        context = self.contexts[self.cc_idx] #current context
        done = False
        
        if GUI:
            border = 0
            pygame.init()
            screen = pygame.display.set_mode((self.env_w+(2*border), self.env_h+(2*border)))
            running = True

            black = (0,0,0)
            gray = (120,120,120)

            screen.fill(context.color)
            p = self.get_euclidian_points(self.agent.c_pos)[0] #slicing at 0 here
            pygame.draw.lines(screen,black,False,p,5)
            pygame.draw.circle(screen,gray,context.objects[0].location,5)
            pygame.display.update()


            while running:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
                if not done:
                    pos, context, done = self.main(context)
                    screen.fill(context.color)
                    pygame.draw.lines(screen,black,False,pos[0],5)#pos sliced at 0 since only one limb
                    pygame.draw.circle(screen,gray,context.objects[0].location,5)
                    pygame.display.update()
                    
        else:
            while not done:
                pos, context, done = self.main(context)
    
    def main(self,context):
        done = False
        self.agent.act(context)
        pos = self.get_euclidian_points(self.agent.c_pos,self.agent.n_limbs) #slice at 0
        reward = context.query_reward(pos,self.agent.n_limbs,self.agent.n_joints)
        self.agent.update_memory(reward)#
        #check on resets
        self.n_step += 1
        self.n_step_total += 1
        if self.n_step >= self.max_steps:
            self.n_step = 1
            self.n_session += 1
            self.agent.neutral()
        
        if self.n_session >= self.max_sessions:
            done = True
        if self.n_step_total % self.update_freq == 0:
            self.agent.train()
        
        if context.check_met_goal() == True:
            self.cc_idx += 1
            agent.actor.fix_weights()
            agent.critic.fix_weights()
            agent.neutral()
            agent.memory.clear()
            if(self.cc_idx >= len(self.contexts)):
                self.cc_idx = 0
                self.first_round = False
            if self.first_round:
                agent.actor.reset_noise()
            #update context    
            context = self.contexts[self.cc_idx]
        return pos, context, done
    
    def gen_env(self):
        minp = self.agent.min_position
        maxp = self.agent.max_position
        maxdist = 2*max(self.agent.total_limb_length)[0]
        for n in range(self.n_contexts):
            color = [random.randint(0,255),random.randint(0,255),random.randint(0,255)] #get a color
            for o in range(self.n_objects):
                location = self.get_euclidian_points([[random.randint(minp,maxp),random.randint(minp,maxp),random.randint(minp,maxp)]])
                #print(location)
                location = location[0][self.agent.n_joints] #last euclidian point; not subtracting 1 becasue there is an extra base point at spot 0
                location_int = [int(np.round(location[0])),int(np.round(location[1]))]

                utility = random.randint(self.min_utility,self.max_utility)
                scale = random.uniform(maxdist/self.scale_ratio_max,maxdist/self.scale_ratio_min)
                threshold = random.uniform(maxdist/self.threshold_ratio_max,maxdist/self.threshold_ratio_min) #ratio of the maximum distance. future proofing.
            self.contexts.append(Context(color,[Object(utility,location_int)],[threshold,self.threshold_2,scale]))
		
	
    def get_euclidian_points(self,pos,n_limbs=1):
        degrconv = np.pi / 180.
        all_points = []
        for i in range(n_limbs):
            points = []
            offset = np.array([self.env_h/2,self.env_w/2])
            offset += self.agent.limb_offsets[i]
            points.append(offset)
            for x in range(self.agent.n_joints):
                h = self.agent.limb_ratios[i][x]*self.agent.limb_lengths[i][0] #get arm length for point
                #print(pos)
                #print(pos[i][x])
                p = np.array([h*np.sin(pos[i][x]*degrconv),h*np.cos(pos[i][x]*degrconv)]) #use the degree value in c_arm_pos to get the x,y coordinates
                #print(p)
                if x == 0:
                    points.append(copy.deepcopy(p+offset))
                elif x > 0:
                    p = p + np.array(points[x]) #x,y coords adjusted based on which joint in arm
                    points.append(copy.deepcopy(p))
            all_points.append(copy.deepcopy(points))
        all_points = np.array(all_points)
        return all_points
    
e = Environment()
e.gen_env()
e.start()
#print(e.contexts[0].objects[0].location)



