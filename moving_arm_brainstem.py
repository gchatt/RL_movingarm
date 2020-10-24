##Step 1: Have 'brainstem' get some fixed weights so it can learn some cue / behavior pairs
##
##Goal, objectively quantifying the advantage that inhibitor / indirect / direct pathways provide to the motor system.

import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.constraints import non_neg
import numpy as np
from numpy import random
import copy
import pickle
import datetime
from collections import deque
import threading
from tensorboard.plugins.hparams import api as hp
import time
from scipy.spatial import distance_matrix

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

verbose = True
GUI = False
use_sleep = False
linux = False
if GUI:
    import pygame
manual_exit = True
if linux:
	manual_exit = False

#HPARAMS
manual_hparams = []

###Environment###
N_CONTEXTS = hp.HParam('n_contexts',hp.Discrete([4]))
manual_hparams.append(N_CONTEXTS)
N_OBJECTS = hp.HParam('n_objects',hp.Discrete([1]))
manual_hparams.append(N_OBJECTS)
MIN_UTILITY = hp.HParam('min_utility',hp.Discrete([100]))
manual_hparams.append(MIN_UTILITY)
MAX_UTILITY = hp.HParam('max_utility',hp.Discrete([110]))
manual_hparams.append(MAX_UTILITY)
SCALE_RATIO_MIN = hp.HParam('scale_ration_min',hp.Discrete([19]))
manual_hparams.append(SCALE_RATIO_MIN)
SCALE_RATIO_MAX = hp.HParam('scale_ration_max',hp.Discrete([20]))
manual_hparams.append(SCALE_RATIO_MAX)
THRESHOLD_RATIO_MIN = hp.HParam('threshold_ratio_min',hp.Discrete([2]))
manual_hparams.append(THRESHOLD_RATIO_MIN)
THRESHOLD_RATIO_MAX = hp.HParam('threshold_ratio_max',hp.Discrete([3]))
manual_hparams.append(THRESHOLD_RATIO_MAX)
#distance threshold for meeting reward vs punishment (cost)
THRESHOLD_2 = hp.HParam('threshold_2',hp.Discrete([5]))
manual_hparams.append(THRESHOLD_2)
UPDATE_FREQ = hp.HParam('update_freq',hp.Discrete([100]))
manual_hparams.append(UPDATE_FREQ)

#Steps before returning to neutral
MAX_STEPS = hp.HParam('max_steps',hp.Discrete([5])) #CURRENTLY THIS CAN'T BE CHANGED
manual_hparams.append(MAX_STEPS)
MAX_SESSIONS = hp.HParam('max_sessions',hp.Discrete([10000000]))
manual_hparams.append(MAX_SESSIONS)

BS_COMPLETED_LR = hp.HParam('bs_completed_lr',hp.Discrete([0.0001]))
manual_hparams.append(BS_COMPLETED_LR)


###Agent###
N_LIMBS = hp.HParam('n_limbs',hp.Discrete([1])) #Not ready for more than 1
manual_hparams.append(N_LIMBS)
N_JOINTS = hp.HParam('n_joints',hp.Discrete([3])) #Not ready for more yet
manual_hparams.append(N_JOINTS)
MIN_POSITION = hp.HParam('min_position',hp.Discrete([-90]))
manual_hparams.append(MIN_POSITION)
MAX_POSITION = hp.HParam('max_position',hp.Discrete([90]))
manual_hparams.append(MAX_POSITION)
#Lower means that the color vector is weighted more
C_DIV = hp.HParam('c_div',hp.Discrete([255.0/16.0]))
manual_hparams.append(C_DIV)
MAX_GRAD = hp.HParam('max_grad',hp.Discrete([10.0])) #for Huber loss
manual_hparams.append(MAX_GRAD)
LR_ACTOR = hp.HParam('lr_actor',hp.Discrete([0.001]))
manual_hparams.append(LR_ACTOR)
LR_CRITIC = hp.HParam('lr_critic',hp.Discrete([0.001]))
manual_hparams.append(LR_CRITIC)
GAMMA = hp.HParam('gamma',hp.Discrete([0.01]))
manual_hparams.append(GAMMA)
MAX_CRITIC_LOSS = hp.HParam('max_critic_loss',hp.Discrete([10000]))
manual_hparams.append(MAX_CRITIC_LOSS)
#TDE scale. Scales the target_Q vs. current_Q. TDE = target_Q * tde_scale - current_Q
#Positive values give some bias to reward vs. prediction
TDE_SCALE = hp.HParam('tde_scale',hp.Discrete([1.02]))
manual_hparams.append(TDE_SCALE)
MAX_TDE = hp.HParam('max_tde',hp.Discrete([100]))
manual_hparams.append(MAX_TDE)
ACTOR_DECAY_RATE = hp.HParam('actor_decay_rate',hp.Discrete([0.00001]))
manual_hparams.append(ACTOR_DECAY_RATE)
CRITIC_DECAY_RATE = hp.HParam('critic_decay_rate',hp.Discrete([0.00001]))
manual_hparams.append(CRITIC_DECAY_RATE)

###Context###
MET_GOAL_CRITERIA = hp.HParam('met_goal_criteria',hp.Discrete([600]))
manual_hparams.append(MET_GOAL_CRITERIA)
MAX_TRIES = hp.HParam('max_tries',hp.Discrete([1000]))
manual_hparams.append(MAX_TRIES)
MET_GOAL_DECAY = hp.HParam('met_goal_decay',hp.Discrete([0.25]))
manual_hparams.append(MET_GOAL_DECAY)
MIN_GOAL_SCALE = hp.HParam('min_goal_scale',hp.Discrete([0.1]))
manual_hparams.append(MIN_GOAL_SCALE)

###Critic###
CRITIC_UNITS = hp.HParam('critic_units',hp.Discrete([1000]))
manual_hparams.append(CRITIC_UNITS)
#number to fix
CRITIC_NFIX = hp.HParam('critic_nfix',hp.Discrete([1000]))
manual_hparams.append(CRITIC_NFIX)

###Actor###
#brainstem layer units
BS_UNITS = hp.HParam('bs_units',hp.Discrete([1000]))
manual_hparams.append(BS_UNITS)

#striatal layer units
STR_UNITS = hp.HParam('str_units',hp.Discrete([1000]))
manual_hparams.append(STR_UNITS)
GPI_UNITS = hp.HParam('gpi_units',hp.Discrete([100]))
manual_hparams.append(GPI_UNITS)

ACTOR_NFIX = hp.HParam('actor_nfix',hp.Discrete([1000]))
manual_hparams.append(ACTOR_NFIX)
STR_NFIX = hp.HParam('str_nfix',hp.Discrete([100]))
manual_hparams.append(STR_NFIX)
MAX_FR = hp.HParam('max_fr',hp.Discrete([200]))
manual_hparams.append(MAX_FR)
#Noise parameters
#noise = max(min(std_d - self.noise_scale*(loss - self.tau),self.std_d_init),noise_base)
#min of noise 1, max noise of the initial noise term (e.g. 90)
#loss is TDE (raw) (or now as of 10/19/20, * a measure of max_reward)
#I can't remember why I have a tau
STD_D = hp.HParam('std_d',hp.Discrete([90]))
manual_hparams.append(STD_D)
TAU = hp.HParam('tau',hp.Discrete([0.1]))
manual_hparams.append(TAU)
NOISE_SCALE = hp.HParam('noise_scale',hp.Discrete([0.02]))
manual_hparams.append(NOISE_SCALE)
NOISE_SCALE_2 = hp.HParam('noise_scale_2',hp.Discrete([0.008]))
manual_hparams.append(NOISE_SCALE_2)
NOISE_BASE = hp.HParam('noise_base',hp.Discrete([3.0]))
manual_hparams.append(NOISE_BASE)
USE_BATCH_NORM = hp.HParam('use_batch_norm',hp.Discrete([True]))
manual_hparams.append(USE_BATCH_NORM)
#if loading weights, what should be the noise (gaussian std)
LOAD_NOISE = hp.HParam('load_noise',hp.Discrete([2.0]))
manual_hparams.append(LOAD_NOISE)

USE_CUST = hp.HParam('use_cust',hp.Discrete([True]))
manual_hparams.append(USE_CUST)

LOADING_STATE = hp.HParam('loading_state',hp.Discrete([False]))
manual_hparams.append(LOADING_STATE)

USE_BG = hp.HParam('use_bg',hp.Discrete([False]))
manual_hparams.append(USE_BG)
USE_BSL1RNN = hp.HParam('use_bsl1rnn',hp.Discrete([True]))
manual_hparams.append(USE_BSL1RNN)

GPI_BIAS = hp.HParam('gpi_bias',hp.Discrete([30.0]))
manual_hparams.append(GPI_BIAS)
GPI_TRAINABLE = hp.HParam('gpi_trainable',hp.Discrete([True]))
manual_hparams.append(GPI_TRAINABLE)

#expand the points in memory (by linear interpolation) to allow for more points for critic to learn from
EXPAND_N = hp.HParam('expand_n',hp.Discrete([100]))
manual_hparams.append(EXPAND_N)
#How close do points need to be to be interpolated. Cannot be less than 0.1.
QUANT_PERCENT = hp.HParam('quant_percent',hp.Discrete([0.1]))
manual_hparams.append(QUANT_PERCENT)
#how much the noise can decline is based on how much the last session was close to the peak reward. If you're getting closer then reward can decline faster.
MAX_R_RATIO = hp.HParam('max_r_ratio',hp.Discrete([0.8]))
manual_hparams.append(MAX_R_RATIO)


METRIC_ACCURACY = 'accuracy'


class cust_layer(layers.Layer):
	
    def __init__(self,input_dim,units,w_init=[-0.001,0.001],b_init=[0.0,0.0],nonneg=False,use_trainable=True):
        super(cust_layer,self).__init__()
        self.input_dim = input_dim
        self.units = units
        self.w_init = tf.random_uniform_initializer(minval=w_init[0], maxval=w_init[1])
        self.b_init = tf.random_uniform_initializer(minval=b_init[0], maxval=b_init[1])
        #decay_rate_init = tf.random_uniform_initializer(minval=decay_init[0], maxval=decay_init[1])
        if nonneg:
            self.w = self.add_weight(shape=(self.input_dim,self.units),initializer=self.w_init,trainable=use_trainable,constraint=non_neg())
        else:
            self.w = self.add_weight(shape=(self.input_dim,self.units),initializer=self.w_init,trainable=use_trainable)  
        self.b = self.add_weight(shape=(self.units,), initializer=self.b_init, trainable=use_trainable)
        
        #self.decay_rates = np.ones_like(self.w.numpy(),dtype=float) * 0.0001
        #self.decay_rate = 0.0001
        #print(self.decay_rates.shape)
        #print(self.initial_weights)
        self.mask = np.ones_like(self.w.numpy())
        self.fixed_values = np.zeros_like(self.w.numpy())
        
        self.added_inputs = False
        self.aw = []
        self.aw_stored_init = []
        self.am = []
        self.afv = []

    def call(self,inputs,new_in=[]):
        masked_weights = tf.multiply(self.mask,self.w)
        w = masked_weights + self.fixed_values
        out = tf.tensordot(inputs,w,[[1],[0]]) + self.b
        if len(new_in) > 0:
            for i in range(len(new_in)):
                #print(inputs)
                
                #print(new_in[0])
                #print('in custom layer')
                #print(len(self.am))
                #print(len(self.aw))
                #print(len(self.afv))
                #print(len(new_in))
                #print(i)
                #print(new_in[i])
                #print(self.aw[i])
                #print(self.am[i])
                #print(self.afv[i])
                mw = tf.multiply(self.am[i],self.aw[i])
                
                w = mw + self.afv[i]
                #print(tf.tensordot(new_in[i],w,[[1],[0]]))
                out += tf.tensordot(new_in[i],w,[[1],[0]])
            #print('here')
            #print(tf.tensordot(new_in[i],w,[[1],[0]]))
            #print(out)
        return out

    def save_init_wts(self):
        self.initial_weights = self.w.numpy()
        for i in range(len(self.aw)):
            self.aw_stored_init.append(self.aw[i].numpy())
        
    #def get_config(self):
        #config = super(cust_layer,self).get_config()
        #config.update({'units':self.units})
        #return config
        
    #probably redo
    def get_wts(self):
        #all in numpy format
        return self.w.numpy(),self.b.numpy(),self.mask,self.fixed_values, self.aw, self.am, self.afv
    
    def load_wts(self,w,b,m,fv,aw_in=[]):
        #dependent on number of weight variables set during initiation
        self.mask = m
        self.fixed_valeus = fv
        if len(aw_in) == 0:
            self.set_weights([tf.convert_to_tensor(w,dtype=tf.float32),tf.convert_to_tensor(b,dtype=tf.float32)])
        elif len(aw_in) == 1:
            self.set_weights([tf.convert_to_tensor(w,dtype=tf.float32),tf.convert_to_tensor(b,dtype=tf.float32),tf.convert_to_tensor(aw_in[0],dtype=tf.float32)])
        elif len(aw_in) == 2:
            self.set_weights([tf.convert_to_tensor(w,dtype=tf.float32),tf.convert_to_tensor(b,dtype=tf.float32),tf.convert_to_tensor(aw_in[0],dtype=tf.float32),tf.convert_to_tensor(aw_in[1],dtype=tf.float32)])
        

    #need to redo for addnl weights
    def decay(self,decay_rate):
        #take a small step back towards init_weights
        w = self.get_weights()[0]
        dw = w - (w - self.initial_weights)*decay_rate
        #decay rates should eventually be trainable variables as should time of fixation
        dw = tf.convert_to_tensor(dw,dtype=tf.float32)
        self.set_weights([dw,self.get_weights()[1]])
    
    def add_inputs(self,input_dim,aw_init=[-0.001,0.001],nonneg=False,use_trainable=True,am_in=[],afv_in=[]):
        #print('here')
        self.added_inputs = True
        self.aw_init = tf.random_uniform_initializer(minval=aw_init[0], maxval=aw_init[1])
        if nonneg:
            self.aw.append(self.add_weight(shape=(input_dim,self.units),initializer=self.aw_init,trainable=use_trainable,constraint=non_neg()))
        else:
            self.aw.append(self.add_weight(shape=(input_dim,self.units),initializer=self.aw_init,trainable=use_trainable))
        
        if len(am_in) > 0:
            for i in range(len(am_in)):
                self.am.append(am_in[i])
        else:
            self.am.append(np.ones_like(self.aw[len(self.aw)-1].numpy()))
            
        if len(afv_in) > 0:
            for i in range(len(afv_in)):
                self.afv.append(afv_in[i])
        else:
            self.afv.append(np.zeros_like(self.aw[len(self.aw)-1].numpy()))    
        
class Actor(keras.Model):
    def __init__(self,params):
        super(Actor,self).__init__()
        #hyperparams
        self.bs_units = params['BS_UNITS'] #brainstem layer units
        self.use_batch_norm = params['USE_BATCH_NORM']
        self.number_to_fix = params['ACTOR_NFIX'] #weights to fix on each 'success'
        self.std_d = params['STD_D']
        self.std_d_init = self.std_d
        self.tau = params['TAU']
        self.noise_scale = params['NOISE_SCALE']
        #self.noise_scale_2 = params['NOISE_SCALE_2']
        self.noise_scale_2 = max(self.noise_scale - 0.001,0.001)
        self.noise_base = params['NOISE_BASE']
        self.max_fr = params['MAX_FR']
        self.use_noise = True
        #end hyperparams
        self.testing = False
        self.use_cust = params['USE_CUST']
        self.use_trainable = params['USE_TRAINABLE']
        self.use_bg = params['USE_BG']
        self.use_bsl1rnn = params['USE_BSL1RNN']
        self.gpi_bias = params['GPI_BIAS']
        self.str_nfix = params['STR_NFIX']
        self.gpi_trainable = params['GPI_TRAINABLE']
        
        ##
        self.str_inputsz = 12
        self.str_units = params['STR_UNITS']
        self.gpi_units = params['GPI_UNITS']
        if self.use_bg:
            self.bg = []
            self.bgbn = [] #batch_norm
            self.last_str_out = []
            self.last_gpi_out = []
            
            #input, output
            #both weights are trainable
            self.bg.append(cust_layer(self.str_inputsz,self.str_units))
            self.bg.append(cust_layer(self.str_units,self.gpi_units,w_init=[-0.001,0.001],b_init=[self.gpi_bias,self.gpi_bias],nonneg=False,use_trainable=self.gpi_trainable)) #positive bias, test both nonneg true and false
            
            #batch_norm
            #Only one for now, but decide if need one on the output since it's going to another layer
            self.bgbn.append(tf.keras.layers.BatchNormalization())
            self.set_bgbn = []
        ######
      
        
        self.network = []
        if self.use_cust:
            #making the input layer trainable; test this on and off
            self.network.append(cust_layer(9,self.bs_units,use_trainable=self.use_trainable))
            self.network.append(cust_layer(self.bs_units,self.bs_units,use_trainable=self.use_trainable))
            self.network.append(cust_layer(self.bs_units,6,use_trainable=self.use_trainable)) #6 = nlimbs * njoints * 2
        else:
            self.l1 = layers.Dense(self.bs_units,activation='relu')
            self.l2 = layers.Dense(self.bs_units,activation='relu')
            self.l3 = layers.Dense(6,activation='relu')


        self.bna = tf.keras.layers.BatchNormalization()
        self.bnb = tf.keras.layers.BatchNormalization()
        
        self.load = False
        
        self.l1_last_out = []
        
        self.n_step = 0
		
    def update_noise(self,tde):
        if self.std_d <= self.noise_base and tde > 0:
            self.std_d = max(self.std_d - self.noise_scale_2*(tde + self.tau),1.0)
        else:
            self.std_d = max(min(self.std_d - self.noise_scale*(tde + self.tau),self.std_d_init),self.noise_base)

    def reset_noise(self):
        self.std_d = self.std_d_init
	
    def fix_weights(self):
        #fix the weights that have the largest distance vector from initiation
        gv = []
        gv_bg = []
        #get the distance vectors for each layer
        
        if self.use_trainable:
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
        
        if self.use_bg:
            gv_bg = np.absolute((self.bg[0].w - self.bg[0].initial_weights).numpy())
            for i in range(self.str_nfix):    
                max_bg_ind = np.unravel_index(np.argmax(gv_bg,axis=None),gv_bg.shape)
                already_fixed_bg = (self.bg[0].mask[max_bg_ind] == 0) #boolean
                while already_fixed_bg:
                    if verbose:
                        print('bg weight was already fixed. trying another')
                    gv_bg[max_bg_ind] = 0
                    max_bg_ind = np.unravel_index(np.argmax(gv_bg,axis=None),gv_bg.shape)
                    already_fixed_bg = (self.bg[0].mask[max_bg_ind] == 0) #boolean
            # set mask to 0 (initialized as ones)
            self.bg[0].mask[max_bg_ind] = 0
            # set the fixed value to the current weight
            self.bg[0].fixed_values[max_bg_ind] = self.bg[0].w[max_bg_ind]                


    
    def decay(self,decay_rate):
        for i in range(len(self.network)):
            self.network[i].decay(decay_rate)
    
    def save_init_weights(self):
        for i in range(len(self.network)):
            self.network[i].save_init_wts()
        if self.use_bg:
            for i in range(len(self.bg)):
                self.bg[i].save_init_wts()
    
    def call(self,bs_input=[],bg_input=[],use_noise=True,bnorm=False,new_in=[]):
        #print('in call '+str(len(new_in)))
        #print(new_in)
        self.gn = layers.GaussianNoise(stddev=self.std_d)
        
        if(self.use_bg):
            #striatum
            out = self.bg[0].call(bg_input)
            out = tf.nn.relu(out)
            if self.use_noise:
                out = self.gn(out)
            #out = -1 * out #forced inhibition; 'out' itself can never be negative due to the relu activation function
            self.last_str_out = out
            #gpi
            #batchnorm
            if self.use_batch_norm:
                out = self.bgbn[0](out,training=bnorm)
            out = self.bg[1].call(out)
            out = tf.nn.relu(out)
            bgout = -1 * out #forced inhibition
            self.last_gpi_out = bgout
            new_in.append(bgout)
            #print(bgout)
        
        
        #Brainstem
        
        #consider trying putting noise here (ie why SMCx to BG and CM/Pf to BG?)
        
        #Layer 1 Brainstem (BS)
        if self.use_cust:
            out = self.network[0].call(bs_input,new_in)
            out = tf.nn.relu(out)
        else:
            out = self.l1(bs_input)
        if self.use_batch_norm:
            out = self.bna(out,training=bnorm)

        #Store L1 out, can be used to create RNN. Currently accessed by the 'agent' class and then fed in via actor.call, new_in   
        #this is post batch_norm
        self.l1_last_out = copy.deepcopy(out)
        
        #Layer 2 Brainstem
        if self.use_cust:
            out = self.network[1].call(out)
            out = tf.nn.relu(out)
        else:
            out = self.l2(out)
        if self.use_batch_norm:
            out = self.bnb(out,training=bnorm)

        #Layer 3 Brainstem
        if self.use_cust:
            out = self.network[2].call(out)
            out = tf.nn.relu(out)
        else:
            out = self.l3(out)
        if use_noise:
            out = self.gn(out)
        #Clip to max firing rate
        out = tf.clip_by_value(out,0,self.max_fr)

        self.n_step += 1
        if(self.n_step == 1 and self.load):
            self.bna.set_weights(self.set_bna)
            self.bnb.set_weights(self.set_bnb)
            if self.use_bg and len(self.set_bgbn) > 0:
                for i in range(len(self.bgbn)):
                    self.bgbn[i].set_weights(self.set_bgbn[i])
                
        
        new_in.clear()
        return out
    
    def save_wts(self,label):
        bs_label = label
        bg_label = label
        if linux:
            bs_label = '/scratch/users/gchatt/logs/' + label + '/bs_layer_'
            bg_label = '/scratch/users/gchatt/logs/' + label + '/bg_layer_'
        else:
            bs_label = os.getcwd()+'\\logs\\'+ label +'\\bs_layer_'
            bg_label = os.getcwd()+'\\logs\\'+ label +'\\bg_layer_'
        for i in range(len(self.network)):
            w,b,m,fv,aw,am,afv = self.network[i].get_wts()
            np.save(bs_label+str(i)+'_weights',w,allow_pickle=False)
            np.save(bs_label+str(i)+'_biases',b,allow_pickle=False)
            np.save(bs_label+str(i)+'_mask',m,allow_pickle=False)
            np.save(bs_label+str(i)+'_fixed_values',fv,allow_pickle=False)
            if self.network[i].added_inputs:
                for x in range(len(aw)):
                    np.save(bs_label+str(i)+'_added_weight_'+str(x),aw[x].numpy(),allow_pickle=False)
                    np.save(bs_label+str(i)+'_added_mask_'+str(x),am[x],allow_pickle=False)
                    np.save(bs_label+str(i)+'_added_fixed_value_'+str(x),afv[x],allow_pickle=False)
        if self.use_bg:
            for i in range(len(self.bg)):
                w,b,m,fv,aw,am,afv = self.bg[i].get_wts()
                np.save(bg_label+str(i)+'_weights',w,allow_pickle=False)
                np.save(bg_label+str(i)+'_biases',b,allow_pickle=False)
                np.save(bg_label+str(i)+'_mask',m,allow_pickle=False)
                np.save(bg_label+str(i)+'_fixed_values',fv,allow_pickle=False)
                if self.bg[i].added_inputs:
                    for x in range(len(aw)):
                        np.save(bg_label+str(i)+'_added_weight_'+str(x),aw[x].numpy(),allow_pickle=False)
                        np.save(bg_label+str(i)+'_added_mask_'+str(x),am[x],allow_pickle=False)
                        np.save(bg_label+str(i)+'_added_fixed_value_'+str(x),afv[x],allow_pickle=False)                
        if self.use_batch_norm:
            np.save(bs_label+'batch_norm_a_0',self.bna.get_weights()[0],allow_pickle=False)
            np.save(bs_label+'batch_norm_a_1',self.bna.get_weights()[1],allow_pickle=False)
            np.save(bs_label+'batch_norm_a_2',self.bna.get_weights()[2],allow_pickle=False)
            np.save(bs_label+'batch_norm_a_3',self.bna.get_weights()[3],allow_pickle=False)            
            np.save(bs_label+'batch_norm_b_0',self.bnb.get_weights()[0],allow_pickle=False)
            np.save(bs_label+'batch_norm_b_1',self.bnb.get_weights()[1],allow_pickle=False)
            np.save(bs_label+'batch_norm_b_2',self.bnb.get_weights()[2],allow_pickle=False)
            np.save(bs_label+'batch_norm_b_3',self.bnb.get_weights()[3],allow_pickle=False)
            if self.use_bg:
                for i in range(len(self.bgbn)):
                    #0 to 3
                    for x in range(4):
                        np.save(bg_label+'batch_norm_'+str(i)+'_'+str(x),self.bgbn[i].get_weights()[x],allow_pickle=False)
 
    def load_wts(self,dir_label):
        self.load = True
        added = 0
        aw = []
        am = []
        afv = []
        make_rnn = False
        make_bg = False
        for i in range(len(self.network)):
            w = np.load(dir_label+'bs_layer_'+str(i)+'_weights.npy')
            b = np.load(dir_label+'bs_layer_'+str(i)+'_biases.npy')
            m = np.load(dir_label+'bs_layer_'+str(i)+'_mask.npy')
            fv = np.load(dir_label+'bs_layer_'+str(i)+'_fixed_values.npy')
            
            if i == 0:
                if self.use_bsl1rnn:
                    try:
                        if verbose:
                            print('trying to load RNN weights')
                            print(len(self.network[0].aw))
                        aw.append(np.load(dir_label+'bs_layer_0_added_weight_0.npy'))
                        am.append(np.load(dir_label+'bs_layer_0_added_mask_0.npy'))
                        afv.append(np.load(dir_label+'bs_layer_0_added_fixed_value_0.npy'))
                        self.network[0].add_inputs(self.bs_units,use_trainable=self.use_trainable,am_in=am,afv_in=afv)
                        added += 1
                        if verbose:
                            print('loaded RNN weights')
                            print(len(self.network[0].aw))
                    except:
                        make_rnn = True
                        if verbose:
                            print('no BSL1 RNN weights to load')
                if self.use_bg:
                    am.clear()
                    afv.clear()
                    if verbose:
                        print('added bias to BS L1 neurons')
                    #print(added)
                    try:
                        if verbose:
                            print('trying to load BG->BS weights')
                        aw.append(np.load(dir_label+'bs_layer_0_added_weight_'+str(added)+'.npy'))
                        am.append(np.load(dir_label+'bs_layer_0_added_mask_'+str(added)+'.npy'))
                        afv.append(np.load(dir_label+'bs_layer_0_added_fixed_value_'+str(added)+'.npy'))
                        self.network[0].add_inputs(self.gpi_units,use_trainable=self.use_trainable)
                        if verbose:
                            print('loaded BG->BS weights')
                    except:
                        make_bg = True
                        #add a bias of +20 to counteract the negative input from the GPi
                        b += 20.0 * np.ones_like(b.shape)
                        if verbose:
                            print('no BG->BS weights to load')
                        #print(make_bg)
                        #print('here')
                self.network[i].load_wts(w,b,m,fv,aw_in=aw)
            else:
                self.network[i].load_wts(w,b,m,fv)
        if self.use_batch_norm:
            a0 = np.load(dir_label+'bs_layer_'+'batch_norm_a_0.npy')
            a1 = np.load(dir_label+'bs_layer_'+'batch_norm_a_1.npy')
            a2 = np.load(dir_label+'bs_layer_'+'batch_norm_a_2.npy')
            a3 = np.load(dir_label+'bs_layer_'+'batch_norm_a_3.npy')
            #print(len(self.bna.get_weights()))
            self.set_bna = [tf.convert_to_tensor(a0,dtype=tf.float32),tf.convert_to_tensor(a1,dtype=tf.float32),tf.convert_to_tensor(a2,dtype=tf.float32),tf.convert_to_tensor(a3,dtype=tf.float32)]

            
            b0 = np.load(dir_label+'bs_layer_'+'batch_norm_b_0.npy')
            b1 = np.load(dir_label+'bs_layer_'+'batch_norm_b_1.npy')
            b2 = np.load(dir_label+'bs_layer_'+'batch_norm_b_2.npy')
            b3 = np.load(dir_label+'bs_layer_'+'batch_norm_b_3.npy')
            self.set_bnb = [tf.convert_to_tensor(b0,dtype=tf.float32),tf.convert_to_tensor(b1,dtype=tf.float32),tf.convert_to_tensor(b2,dtype=tf.float32),tf.convert_to_tensor(b3,dtype=tf.float32)]
        if (not make_bg) and self.use_bg:
            for i in range(len(self.bg)):
                w = np.load(dir_label+'bg_layer_'+str(i)+'_weights.npy')
                b = np.load(dir_label+'bg_layer_'+str(i)+'_biases.npy')
                m = np.load(dir_label+'bg_layer_'+str(i)+'_mask.npy')
                fv = np.load(dir_label+'bg_layer_'+str(i)+'_fixed_values.npy')
                self.bg[i].load_wts(w,b,m,fv)
            for i in range(len(self.bgbn)):
                self.set_bgbn.append([tf.convert_to_tensor(np.load(dir_label+'bg_layer_batch_norm_'+str(i)+'_'+str(x)+'.npy'),dtype=tf.float32) for x in range(4)])
                
        #w = np.array([self.network[i].get_wts() for i in range(len(self.network))])
        #w.save(os.getcwd()+'\\logs\\'+'array')
        return make_rnn, make_bg

class Critic(keras.Model):
    def __init__(self,params):
        super(Critic,self).__init__()
        #hyperparams
        self.units = params['CRITIC_UNITS']
        self.number_to_fix = params['CRITIC_NFIX']
        self.input_sz = params['INPUT_SZ']
        self.action_sz = params['ACTION_SZ']
        self.use_batch_norm = params['USE_BATCH_NORM']
        self.use_cust = params['USE_CUST']

        self.critic_layers = []
        #color,c_pos,action
        if self.use_cust:
            self.critic_layers.append(cust_layer(self.input_sz+self.action_sz,self.units))
            self.critic_layers.append(cust_layer(self.units,self.units))
            self.critic_layers.append(cust_layer(self.units,1))
        else:
            self.l1 = layers.Dense(self.units,activation='relu')
            self.l2 = layers.Dense(self.units,activation='relu')
            self.l3 = layers.Dense(1,activation='relu')

        self.bna = tf.keras.layers.BatchNormalization()
        self.bnb = tf.keras.layers.BatchNormalization()	

        self.n_step = 0
        self.load = False

    def call(self,state,action,bnorm):
        inputs = layers.concatenate([state,action])
        if self.use_cust:
            out = self.critic_layers[0].call(inputs)
            out = tf.nn.relu(out)
        else:
            out = self.l1(inputs)
        if self.use_batch_norm:
            #print(out)
            out = self.bna(out,training=bnorm)
            #print(out - out_b)
            #out = out_b
            
        if self.use_cust:
            out = self.critic_layers[1].call(out)
            out = tf.nn.relu(out)
        else:
            out = self.l2(out)
        if self.use_batch_norm:
            out = self.bnb(out,training=bnorm)
        
        if self.use_cust:
            val = self.critic_layers[2].call(out)
            val = tf.nn.relu(val)
        else:
            val = self.l3(out)
        
        self.n_step += 1
        if(self.n_step == 1 and self.load):
            self.bna.set_weights(self.set_bna)
            self.bnb.set_weights(self.set_bnb)
        
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
    
    def decay(self,decay_rate):
        for i in range(len(self.critic_layers)):
            self.critic_layers[i].decay(decay_rate)
    
    def save_init_weights(self):
        for i in range(len(self.critic_layers)):
            self.critic_layers[i].save_init_wts()
            
    def save_wts(self,label):
        save_label = label
        if linux:
            save_label = '/scratch/users/gchatt/logs/' + label + '/critic_layer_'
        else:
            save_label = os.getcwd()+'\\logs\\'+ label +'\\critic_layer_'
        for i in range(len(self.critic_layers)):
            w,b,m,fv,aw,am,afv = self.critic_layers[i].get_wts()
            np.save(save_label+str(i)+'_weights',w,allow_pickle=False)
            np.save(save_label+str(i)+'_biases',b,allow_pickle=False)
            np.save(save_label+str(i)+'_mask',m,allow_pickle=False)
            np.save(save_label+str(i)+'_fixed_values',fv,allow_pickle=False)
        if self.use_batch_norm:
            np.save(save_label+'batch_norm_a_0',self.bna.get_weights()[0],allow_pickle=False)
            np.save(save_label+'batch_norm_a_1',self.bna.get_weights()[1],allow_pickle=False)
            np.save(save_label+'batch_norm_a_2',self.bna.get_weights()[2],allow_pickle=False)
            np.save(save_label+'batch_norm_a_3',self.bna.get_weights()[3],allow_pickle=False)            
            np.save(save_label+'batch_norm_b_0',self.bnb.get_weights()[0],allow_pickle=False)
            np.save(save_label+'batch_norm_b_1',self.bnb.get_weights()[1],allow_pickle=False)
            np.save(save_label+'batch_norm_b_2',self.bnb.get_weights()[2],allow_pickle=False)
            np.save(save_label+'batch_norm_b_3',self.bnb.get_weights()[3],allow_pickle=False)
            
    def load_wts(self,dir_label):
        self.load = True
        for i in range(len(self.critic_layers)):
            w = np.load(dir_label+str(i)+'_weights.npy')
            b = np.load(dir_label+str(i)+'_biases.npy')
            m = np.load(dir_label+str(i)+'_mask.npy')
            fv = np.load(dir_label+str(i)+'_fixed_values.npy')
            self.critic_layers[i].load_wts(w,b,m,fv)
        if self.use_batch_norm:
            a0 = np.load(dir_label+'batch_norm_a_0.npy')
            a1 = np.load(dir_label+'batch_norm_a_1.npy')
            a2 = np.load(dir_label+'batch_norm_a_2.npy')
            a3 = np.load(dir_label+'batch_norm_a_3.npy')
            self.set_bna = [tf.convert_to_tensor(a0,dtype=tf.float32),tf.convert_to_tensor(a1,dtype=tf.float32),tf.convert_to_tensor(a2,dtype=tf.float32),tf.convert_to_tensor(a3,dtype=tf.float32)]

            b0 = np.load(dir_label+'batch_norm_b_0.npy')
            b1 = np.load(dir_label+'batch_norm_b_1.npy')
            b2 = np.load(dir_label+'batch_norm_b_2.npy')
            b3 = np.load(dir_label+'batch_norm_b_3.npy')
            self.set_bnb = [tf.convert_to_tensor(b0,dtype=tf.float32),tf.convert_to_tensor(b1,dtype=tf.float32),tf.convert_to_tensor(b2,dtype=tf.float32),tf.convert_to_tensor(b3,dtype=tf.float32)]

	
class Object():
    def __init__(self,utility=[],location=[],name=''):
        self.utility = utility
        self.location = location #x,y
        self.name = name

class Context():
    def __init__(self,color=[],objects=[],rfp=[],hparams=[]):
        self.color = color
        self.color2 = []
        self.hparams = hparams
        self.objects = objects
        #reward function parameters
        self.rfp = rfp #[threshold1,threshold2,scale]
        self.met_goal = 0
        self.tries = 0
        
        #hyperparam
        self.met_goal_criteria = self.hparams[MET_GOAL_CRITERIA]
        self.max_tries = self.hparams[MAX_TRIES]
        self.met_goal_scale = 1.0
        self.min_goal_scale = self.hparams[MIN_GOAL_SCALE]
        self.met_goal_decay = self.hparams[MET_GOAL_DECAY]

    def query_reward(self,c_pos,nlimb,njoint):
        reward = np.array([0.0])
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
        reward = reward * self.met_goal_scale
        return reward
    
    def check_met_goal(self):
        met = False
        if self.tries <= self.max_tries:
            if self.met_goal >= self.met_goal_criteria:
                met = True
                self.met_goal = 0
                self.tries = 0
        else:
            self.met_goal = 0
            self.tries = 0
        return met
    
    def decay_reward(self):
        self.met_goal_scale = max(self.met_goal_scale - self.met_goal_decay,self.min_goal_scale)
        
class Memory:
    def __init__(self):
        self.prestates = []
        self.poststates = []
        self.prestates2 = []
        self.poststates2 = []
        self.actions = []
        self.rewards = []
        self.l1_last_outs = []
        self.max_reward = 0
        
    def store(self,prestate,poststate,action,reward):
        self.prestates.append(prestate)
        self.poststates.append(poststate)
        self.actions.append(action)
        self.rewards.append(reward)
        if reward > self.max_reward:
            self.max_reward = reward

    def store2(self,prestate,prestate2,poststate,poststate2,action,reward):
        self.prestates.append(prestate)
        self.poststates.append(poststate)
        self.prestates2.append(prestate2)
        self.poststates2.append(poststate2)
        self.actions.append(action)
        self.rewards.append(reward)
        if reward > self.max_reward:
            self.max_reward = reward
    
    def store_l1(self,l1_out):
        self.l1_last_outs.append(l1_out)
        #print(l1_out)
            

    def clear(self):
        self.prestates = []
        self.poststates = []
        self.prestates2 = []
        self.poststates2 = []
        self.actions = []
        self.rewards = []
        self.l1_last_outs = []
        
        
class Agent():
    def __init__(self,summary_writer,log_save_label,hparams=[]):
        self.hparams = hparams
        self.summary_writer = summary_writer
        #hyperparams
        self.n_limbs = self.hparams[N_LIMBS]
        self.n_joints = self.hparams[N_JOINTS]
        self.min_position = self.hparams[MIN_POSITION]
        self.max_position = self.hparams[MAX_POSITION]
        self.limb_ratios = np.array([[1,1,0.3]]) #set for just one limb
        self.limb_lengths = np.array([[50]]) #again, just one limb here, but can add more
        self.limb_offsets = np.array([[0,0]]) #[[x_limb1,y_limb1],..] (fixed point of a limb)
        self.cdiv = self.hparams[C_DIV] #for normalizing the color vector
        self.max_grad = self.hparams[MAX_GRAD] #for huber loss
        self.lr_actor = self.hparams[LR_ACTOR]
        self.lr_critic = self.hparams[LR_CRITIC]
        self.gamma = self.hparams[GAMMA] #discount parameter
        self.max_critic_loss = self.hparams[MAX_CRITIC_LOSS]
        self.tde_scale = self.hparams[TDE_SCALE]
        self.max_tde = self.hparams[MAX_TDE]
        self.actor_decay_rate = self.hparams[ACTOR_DECAY_RATE]
        self.critic_decay_rate = self.hparams[CRITIC_DECAY_RATE]
        self.use_bg = self.hparams[USE_BG]
        self.use_bsl1rnn = self.hparams[USE_BSL1RNN]
        self.expand_n = self.hparams[EXPAND_N] #expanding the 'memory' for critic training
        self.max_r_ratio = self.hparams[MAX_R_RATIO]
        self.quant_percent = self.hparams[QUANT_PERCENT]
        #end hyperparams
        
        self.loading_state = self.hparams[LOADING_STATE]
        
        actor_params = {}
        actor_params['BS_UNITS'] = self.hparams[BS_UNITS]
        actor_params['STR_UNITS'] = self.hparams[STR_UNITS]
        actor_params['GPI_UNITS'] = self.hparams[GPI_UNITS]
        actor_params['USE_BATCH_NORM'] = self.hparams[USE_BATCH_NORM]
        actor_params['ACTOR_NFIX'] = self.hparams[ACTOR_NFIX]
        actor_params['STD_D'] = self.hparams[STD_D]
        actor_params['TAU'] = self.hparams[TAU]
        actor_params['NOISE_SCALE'] = self.hparams[NOISE_SCALE]
        actor_params['NOISE_SCALE_2'] = self.hparams[NOISE_SCALE_2]
        actor_params['NOISE_BASE'] = self.hparams[NOISE_BASE]
        actor_params['MAX_FR'] = self.hparams[MAX_FR]
        actor_params['USE_CUST'] = self.hparams[USE_CUST]
        actor_params['USE_BG'] = self.hparams[USE_BG]
        actor_params['USE_BSL1RNN'] = self.hparams[USE_BSL1RNN]
        actor_params['GPI_BIAS'] = self.hparams[GPI_BIAS]
        actor_params['STR_NFIX'] = self.hparams[STR_NFIX]
        actor_params['GPI_TRAINABLE'] = self.hparams[GPI_TRAINABLE]
        
        if self.loading_state:
            actor_params['USE_TRAINABLE'] = False
        else:
            actor_params['USE_TRAINABLE'] = True
        
        critic_params = {}
        critic_params['CRITIC_UNITS'] = self.hparams[CRITIC_UNITS]
        critic_params['CRITIC_NFIX'] = self.hparams[CRITIC_NFIX]
        critic_params['USE_CUST'] = self.hparams[USE_CUST]
        critic_params['USE_BATCH_NORM'] = self.hparams[USE_BATCH_NORM]
        if self.use_bg:
            critic_params['INPUT_SZ'] = 12
            critic_params['ACTION_SZ'] = 6        
        else:
            critic_params['INPUT_SZ'] = 9
            critic_params['ACTION_SZ'] = 6
        
        
        self.actor = Actor(actor_params)
        self.actor_opt = tf.keras.optimizers.Adam(learning_rate=self.lr_actor)
        self.critic = Critic(critic_params)
        self.critic_opt = tf.keras.optimizers.Adam(learning_rate=self.lr_critic)
        
        self.bg_out = []
        
        self.log_save_label = log_save_label
        #needed for using noise
        tf.keras.backend.set_learning_phase(1)

        self.memory = Memory()
        
        self.pre_state = []
        self.pre_state2 = []
        self.post_state = []
        self.post_state2 = []
        self.action = []
        self.last_bs_l1_out = []
        
        self.actor_loss = 0
        self.critic_loss = 0
        self.tde = 0
        self.last_reward_avg = 0
        self.n_train = 0
        self.n_step = 0
        self.last_reward = [0.0]
        self.location = [0.0,0.0]
        self.euclidian_pos = [0.0,0.0]
        
        self.is_neutral = True
        
        self.cost = 0
        #make hyperparam
        self.cost_scale = 0.001
        
        self.use_decay = False
        self.bnorm_train = True

        self.c_pos = np.zeros((self.n_limbs,self.n_joints)) #current position of all limbs, joints		
        self.total_limb_length = []
        for i in range(self.n_limbs):
            for x in range(self.n_joints):
                l = self.limb_ratios[i][0]*self.limb_lengths[i]+self.limb_ratios[i][1]*self.limb_lengths[i]+self.limb_ratios[i][2]*self.limb_lengths[i]
                self.total_limb_length.append([l])
        self.total_limb_length = np.array(self.total_limb_length)
	
    def act(self,context):
        self.location = context.objects[0].location
        self.n_step += 1
        color = np.array([context.color])/self.cdiv
        pos = self.expand_vector(self.c_pos)
        if self.use_bg:
            color2 = np.array([context.color2])/self.cdiv
            self.pre_state = layers.concatenate([color,pos])
            #a new context vector with a second color (more contextual complexity)
            self.pre_state2 = layers.concatenate([color,color2,pos])
            
            #get BS L1 out info, it is written out; don't have to use it though
            if self.is_neutral:
                self.last_bs_l1_out = tf.zeros([1,1000])
            else:
                self.last_bs_l1_out = self.actor.l1_last_out 
            
            #Here you decide if you use the BS L1 info or not
            if self.use_bsl1rnn:
                #print('here')
                self.memory.store_l1(self.last_bs_l1_out)
                self.action = self.actor(self.pre_state,bg_input=self.pre_state2,use_noise=True,bnorm=False,new_in=[self.last_bs_l1_out])
            else:
                self.action = self.actor(self.pre_state,bg_input=self.pre_state2,use_noise=True,bnorm=False)
            self.cost = self.cost_scale * tf.norm(self.action).numpy()
            if(self.n_step == 1):
                self.actor.save_init_weights()
                self.critic.save_init_weights()
            self.move_arm(self.action) #updates c_pos
            new_pos = self.expand_vector(self.c_pos)
            self.post_state = layers.concatenate([color,new_pos])
            self.post_state2 = layers.concatenate([color,color2,new_pos])
            self.log_act()
        else:
            #print('here in else')
            if self.is_neutral:
                self.last_bs_l1_out = tf.zeros([1,1000])
            else:
                self.last_bs_l1_out = self.actor.l1_last_out 
            
            self.memory.store_l1(self.last_bs_l1_out)
            
            self.pre_state = layers.concatenate([color,pos])
            #feed forward pass through the actor (in this case 'brainstem') with contextual input
            if self.use_bsl1rnn:
                self.action = self.actor(self.pre_state,use_noise=True,bnorm=False,new_in=[self.last_bs_l1_out])
            else:
                self.action = self.actor(self.pre_state,use_noise=True,bnorm=False)
            
            
            self.cost = self.cost_scale * tf.norm(self.action).numpy()
            #max norm is 350 for max_fr 200 for 3 limbs
            if(self.n_step == 1):
                self.actor.save_init_weights()
                self.critic.save_init_weights()
            self.move_arm(self.action) #updates c_pos
            new_pos = self.expand_vector(self.c_pos)
            self.post_state = layers.concatenate([color,new_pos])
            self.log_act()
        self.is_neutral = False
        
    def move_arm(self,action):
        action = action[0]
        pos_action = np.array([action[0],action[2],action[4]]);
        neg_action = np.array([action[1],action[3],action[5]]);
        #fix
        self.c_pos = self.c_pos + pos_action; #add numpy arrays #position in unit degrees; degree representing the position on a circle
        self.c_pos = self.c_pos - neg_action;
        self.c_pos = np.clip(self.c_pos,self.min_position,self.max_position); #array, min, max. Clip the values so the arm doesn't swing all the way around

    def update_memory(self,reward):
        reward = reward - self.cost
        self.last_reward = reward
        if self.loading_state:
            self.memory.store2(self.pre_state,self.pre_state2,self.post_state,self.post_state2,self.action,reward)
        else:
            self.memory.store(self.pre_state,self.post_state,self.action,reward)
    
    def neutral(self):
        self.c_pos = np.zeros((self.n_limbs,self.n_joints))
        self.is_neutral = True
        
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
    
    def load_existing_ac(self,dir_label):
        if linux:
            dir_label_a = dir_label+'/'
            dir_label_c = dir_label+'/critic_layer_'
        else:
            dir_label_a = dir_label+'\\'
            dir_label_c = dir_label+'\\critic_layer_'       
        make_rnn,make_bg = self.actor.load_wts(dir_label_a)
        #Load critic here, fix
        return make_rnn,make_bg

    def create_new_weights(self,make_rnn=True,make_bg=True):
        #Add inputs
        #fix, make robust w/ hyperparams
        #Order may matter
        if self.use_bsl1rnn and make_rnn:
            if verbose:
                print('adding RNN weights to BS L1')
                print(len(self.actor.network[0].aw))
            self.actor.network[0].add_inputs(self.hparams[BS_UNITS]) #Recurrent inputs
            if verbose:
                print('added RNN weights to BS L1')
                print(len(self.actor.network[0].aw))        
        
        #BG inputs to BS. Fix the weights and make them non_trainable. BS biases are also fixed
        if self.use_bg and make_bg:
            if verbose:
                print('adding BG weights to BS L1')
                print(len(self.actor.network[0].aw))
            self.actor.network[0].add_inputs(self.hparams[GPI_UNITS],aw_init=[0.001,0.001],nonneg=True,use_trainable=False)        
            if verbose:
                print('added BG weights to BS L1')
                print(len(self.actor.network[0].aw))                
            
    def expand_q_function(self,pre_mem,mem_actions,target_Q,color_size):
        expand_n = self.expand_n
        if expand_n == 0:
            return pre_mem,mem_actions,target_Q
        csz = color_size #3 normally, 6 if more info, etc.
        x = np.concatenate([pre_mem.numpy(),mem_actions.numpy()],1)
        x_r = x.shape[0] #100; elements stored in memory
        x_c = x.shape[1] #15; 3 color, 6 position, 6 action
        #x = np.concatenate([pre_mem[:,x_c_start:x_c],mem_actions],1) #this was to slice out color info
        #print(x.shape)
        #print(x)
        x_dm = distance_matrix(x,x,p=2) #pair wise distance between all vectors in x. Shape will be x_r, x_r
        quant_percent = 0.1 #This is important to play around w/
        thresh = np.quantile(x_dm,quant_percent)
        #print(thresh)
        idx1,idx2 = np.asarray(x_dm<=thresh).nonzero()
        #print(len(idx1))
        
            
        for i in range(len(idx1)):
            a,b = idx1[i],idx2[i]
            if a < b and np.linalg.norm(x[a]-x[b])>0 and all(x[a][0:csz]==x[b][0:csz]):
                #must be same color, must not be the same point, a<b prevents repeats
                #x[a][0]==x[b][0] is a weak check though; it should be 1:3 or more for BG
                
                #print(pre_mem[a])
                expand_pre_mem = np.linspace(pre_mem[a],pre_mem[b],expand_n,endpoint=False)
                #print(expand_pre_mem)
                expand_pre_mem = expand_pre_mem[1:expand_n,:]
                epm = tf.convert_to_tensor(expand_pre_mem,dtype=tf.float32)
                pre_mem = tf.concat([pre_mem,epm],0)
                
                #print(expand_pre_mem)
                #print(pre_mem[b])
                expand_mem_actions = np.linspace(mem_actions[a],mem_actions[b],expand_n,endpoint=False)
                expand_mem_actions = expand_mem_actions[1:expand_n,:]
                ema = tf.convert_to_tensor(expand_mem_actions,dtype=tf.float32)
                mem_actions = tf.concat([mem_actions,ema],0)
                
                expand_target_q = np.linspace(target_Q[a],target_Q[b],expand_n,endpoint=False)
                expand_target_q = expand_target_q[1:expand_n,:]
                etq = tf.convert_to_tensor(expand_target_q,dtype=tf.float32)
                target_Q = tf.concat([target_Q,etq],0)

                #pops off the top because that's just a repeat of the first input in the range
        #print(target_Q.shape)
        return pre_mem,mem_actions,target_Q
        
    
    
    
    def train(self):
        huber = tf.keras.losses.Huber(delta=self.max_grad)
        pre_mem = tf.convert_to_tensor(np.vstack(self.memory.prestates),dtype=tf.float32)
        post_mem = tf.convert_to_tensor(np.vstack(self.memory.poststates),dtype=tf.float32)
        if self.use_bg:
            pre_mem2 = tf.convert_to_tensor(np.vstack(self.memory.prestates2),dtype=tf.float32)
            post_mem2 = tf.convert_to_tensor(np.vstack(self.memory.poststates2),dtype=tf.float32)
        if self.use_bsl1rnn:
            l1_outs = tf.convert_to_tensor(np.vstack(self.memory.l1_last_outs),dtype=tf.float32)
        mem_actions = tf.convert_to_tensor(np.vstack(self.memory.actions),dtype=tf.float32)
        mem_rewards = tf.convert_to_tensor(np.vstack(self.memory.rewards),dtype=tf.float32)
        self.last_reward_avg = tf.reduce_mean(mem_rewards)
        msr_of_max_r = (1-self.max_r_ratio) + self.max_r_ratio*(self.last_reward_avg/self.memory.max_reward)
        #allows the decline in noise to scale up as you get closer to max_reward
        
        if self.use_bg:
            if self.use_bsl1rnn:
                target_Q = self.critic(post_mem2,self.actor(post_mem,bg_input=post_mem2,use_noise=False,bnorm=self.bnorm_train,new_in=[l1_outs]),bnorm=False)
            else:
                target_Q = self.critic(post_mem2,self.actor(post_mem,bg_input=post_mem2,use_noise=False,bnorm=self.bnorm_train),bnorm=False)                          
            target_Q = mem_rewards + tf.math.multiply(target_Q,self.gamma)
            
            with tf.GradientTape() as tape:
                current_Q = self.critic(pre_mem2,mem_actions,bnorm=self.bnorm_train) #bnorm only true here because these were real pairs that occurred
                td_errors = huber(target_Q,current_Q)
                self.critic_loss = tf.clip_by_value(tf.reduce_mean(td_errors),0,self.max_critic_loss)	
            self.critic_grad = tape.gradient(self.critic_loss,self.critic.trainable_variables)
            self.critic_opt.apply_gradients(zip(self.critic_grad,self.critic.trainable_variables))
            
            #adjust actor noise
            self.tde = target_Q*self.tde_scale - current_Q
            self.tde = tf.reduce_mean(self.tde)
            self.tde = self.tde * msr_of_max_r #new 10/19/20
            self.tde = min(self.tde,self.max_tde)
            self.actor.update_noise(self.tde)
        else:
            if self.use_bsl1rnn:
                target_Q = self.critic(post_mem,self.actor(post_mem,use_noise=False,bnorm=self.bnorm_train,new_in=[l1_outs]),bnorm=False)
            else:
                target_Q = self.critic(post_mem,self.actor(post_mem,use_noise=False,bnorm=self.bnorm_train),bnorm=False)            
            target_Q = mem_rewards + tf.math.multiply(target_Q,self.gamma)
            
            #trial
            e_pre_mem,e_mem_actions,e_target_Q = self.expand_q_function(pre_mem,mem_actions,target_Q,3)
            
            with tf.GradientTape() as tape:
                current_Q = self.critic(e_pre_mem,e_mem_actions,bnorm=self.bnorm_train) #bnorm only true here because these were real pairs that occurred
                td_errors = huber(e_target_Q,current_Q)
                self.critic_loss = tf.clip_by_value(tf.reduce_mean(td_errors),0,self.max_critic_loss)	
            self.critic_grad = tape.gradient(self.critic_loss,self.critic.trainable_variables)
            self.critic_opt.apply_gradients(zip(self.critic_grad,self.critic.trainable_variables))
            
            #adjust actor noise
            self.tde = e_target_Q*self.tde_scale - current_Q
            self.tde = tf.reduce_mean(self.tde)
            self.tde = min(self.tde,self.max_tde)        
            self.actor.update_noise(self.tde)
        
        if self.use_bg:
            if self.use_bsl1rnn:
                with tf.GradientTape() as tape:
                    self.next_actions = self.actor(pre_mem,bg_input=pre_mem2,use_noise=False,bnorm=self.bnorm_train,new_in=[l1_outs])
                    self.actor_loss = -tf.reduce_mean(self.critic(pre_mem2,self.next_actions,bnorm=False))
            else:
                with tf.GradientTape() as tape:
                    self.next_actions = self.actor(pre_mem,bg_input=pre_mem2,use_noise=False,bnorm=self.bnorm_train)
                    self.actor_loss = -tf.reduce_mean(self.critic(pre_mem2,self.next_actions,bnorm=False))

            self.actor_grad = tape.gradient(self.actor_loss,self.actor.trainable_variables)
            self.actor_opt.apply_gradients(zip(self.actor_grad,self.actor.trainable_variables))
            
        else:
            if self.use_bsl1rnn:
                with tf.GradientTape() as tape:
                    self.next_actions = self.actor(pre_mem,use_noise=False,bnorm=self.bnorm_train,new_in=[l1_outs])
                    #print(next_actions)
                    self.actor_loss = -tf.reduce_mean(self.critic(pre_mem,self.next_actions,bnorm=False))
            else:
                with tf.GradientTape() as tape:
                    self.next_actions = self.actor(pre_mem,use_noise=False,bnorm=self.bnorm_train)
                    #print(next_actions)
                    self.actor_loss = -tf.reduce_mean(self.critic(pre_mem,self.next_actions,bnorm=False))            
            
            self.actor_grad = tape.gradient(self.actor_loss,self.actor.trainable_variables)
            self.actor_opt.apply_gradients(zip(self.actor_grad,self.actor.trainable_variables))
            if self.use_decay:
                self.actor.decay(self.actor_decay_rate)
                self.critic.decay(self.critic_decay_rate)
            ###go back and fix use_cust
        self.actor.save_wts(self.log_save_label)
        self.critic.save_wts(self.log_save_label)
        
        self.n_train += 1
        self.log_train()
        self.memory.clear()
    
    def log_act(self):
        with self.summary_writer.as_default():
            tf.summary.scalar('arm position[0]',self.c_pos[0][0],step=self.n_step)
            tf.summary.scalar('arm position[1]',self.c_pos[0][1],step=self.n_step)
            tf.summary.scalar('arm position[2]',self.c_pos[0][2],step=self.n_step)
            tf.summary.scalar('arm x',self.euclidian_pos[0],step=self.n_step)
            tf.summary.scalar('arm y',self.euclidian_pos[1],step=self.n_step)
            tf.summary.scalar('reward',self.last_reward[0],step=self.n_step)
            tf.summary.scalar('obj x',self.location[0],step=self.n_step)
            tf.summary.scalar('obj y',self.location[1],step=self.n_step)
            tf.summary.scalar('cost',self.cost,step=self.n_step)
            tf.summary.histogram('action',self.action,step=self.n_step)
            tf.summary.histogram('bs_l1_out',self.last_bs_l1_out,step=self.n_step)
            if self.use_bg:
                tf.summary.histogram('str_out',self.actor.last_str_out,step=self.n_step)    
                tf.summary.histogram('gpi_out',self.actor.last_gpi_out,step=self.n_step)
    
    def log_train(self):
        with self.summary_writer.as_default():
            tf.summary.scalar('actor loss',self.actor_loss,step=self.n_train)
            tf.summary.scalar('last avg reward',self.last_reward_avg,step=self.n_train)
            tf.summary.scalar('critic loss',self.critic_loss,step=self.n_train)
            tf.summary.scalar('raw TD error',self.tde,step=self.n_train)
            tf.summary.scalar('noise',self.actor.std_d,step=self.n_train)
            #tf.summary.histogram('critic grad',self.critic_grad[0],step=self.n_train)
            #tf.summary.histogram('actor grad',self.actor_grad[0],step=self.n_train)
            tf.summary.histogram('next actions',self.next_actions[0],step=self.n_train)


            #fix, use_cust
            tf.summary.histogram('actor bsl1 weights',self.actor.network[0].weights[0],step=self.n_train)
            tf.summary.histogram('actor bsl1 biases',self.actor.network[0].weights[1],step=self.n_train)
            # tf.summary.histogram('actor weights 2',self.actor.network[1].trainable_weights[0],step=self.n_train)
            # tf.summary.histogram('actor weights 3',self.actor.network[2].trainable_weights[0],step=self.n_train)
            # tf.summary.histogram('critic weights 1',self.critic.critic_layers[0].trainable_weights[0],step=self.n_train)
            # tf.summary.histogram('critic weights 2',self.critic.critic_layers[1].trainable_weights[0],step=self.n_train)
            # tf.summary.histogram('critic weights 3',self.critic.critic_layers[2].trainable_weights[0],step=self.n_train)
            # if self.hparams[USE_BATCH_NORM]:
                # tf.summary.histogram('bnorma0',self.actor.bna.get_weights()[0],step=self.n_train)
                # tf.summary.histogram('bnorma1',self.actor.bna.get_weights()[1],step=self.n_train)
                # tf.summary.histogram('bnorma2',self.actor.bna.get_weights()[2],step=self.n_train)
                # tf.summary.histogram('bnorma3',self.actor.bna.get_weights()[3],step=self.n_train)
                # tf.summary.histogram('bnormb0',self.actor.bnb.get_weights()[0],step=self.n_train)
                # tf.summary.histogram('bnormb1',self.actor.bnb.get_weights()[1],step=self.n_train)
                # tf.summary.histogram('bnormb2',self.actor.bnb.get_weights()[2],step=self.n_train)
                # tf.summary.histogram('bnormb3',self.actor.bnb.get_weights()[3],step=self.n_train)



class Environment(threading.Thread):
    def __init__(self,hparams):
        super(Environment, self).__init__()
        self.hparams = hparams
        #hyperparams
        self.n_contexts = self.hparams[N_CONTEXTS]
        self.n_objects = self.hparams[N_OBJECTS] #n_objects per context #not ready to increase this
        self.env_h = 500#pygame
        self.env_w = 500#pygame
        #self.max_obj_dist = 100
        self.min_utility = self.hparams[MIN_UTILITY] #not ready to make negative yet
        self.max_utility = self.hparams[MAX_UTILITY]
        self.scale_ratio_max = self.hparams[SCALE_RATIO_MAX]
        self.scale_ratio_min = self.hparams[SCALE_RATIO_MIN]
        self.threshold_ratio_max = self.hparams[THRESHOLD_RATIO_MAX]
        self.threshold_ratio_min = self.hparams[THRESHOLD_RATIO_MIN]
        self.threshold_2 = self.hparams[THRESHOLD_2] #distance for meeting criteria of meeting goal
        #self.cdiv = 255
        self.update_freq = self.hparams[UPDATE_FREQ]
        self.max_steps = self.hparams[MAX_STEPS]
        self.max_sessions = self.hparams[MAX_SESSIONS]
        self.bs_completed_lr = self.hparams[BS_COMPLETED_LR]
        self.use_bg = self.hparams[USE_BG]
        #end hyperparams
        self.log_dir = ''
        self.current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S%f")
        param_str = str({h.name: self.hparams[h] for h in hparams})
        if linux:
            self.log_dir = '/scratch/users/gchatt/logs/' + self.current_time
            self.summary_writer = tf.summary.create_file_writer(self.log_dir)
            with open(self.log_dir+'/header.txt','w') as header_file:
                header_file.write(param_str)
        else:
            self.log_dir = os.getcwd()+'\\logs\\' + self.current_time
            self.summary_writer = tf.summary.create_file_writer(self.log_dir)
            with open(self.log_dir+'\\header.txt','w') as header_file:
                header_file.write(param_str)

        self.agent = Agent(self.summary_writer,self.current_time,hparams=self.hparams)
        self.contexts = []
        #current context index
        self.cc_idx = 0
        self.first_round = True #if haven't completed all contexts yet
        self.loading_state = self.hparams[LOADING_STATE]
        self.load_dir_label = ''
        ###go back and fix use_cust
        self.fix_weights = False
        if self.hparams[USE_CUST]:
            self.fix_weights = True
        
        self.solved_context = []
            
        self.rounds = 0

    def run(self):

        if self.loading_state:
            self.load_env(os.getcwd()+'\\load\\solved-rnn-1')
            #self.load_env(os.getcwd()+'\\load\\20200909-212541996035')
        else:
            self.gen_env()

        self.start_round()
        with self.summary_writer.as_default():
            hp.hparams(self.hparams)
            tf.summary.scalar(METRIC_ACCURACY,self.agent.last_reward[0],step=1)

	
    def start_round(self):
        self.n_step = 0
        self.n_step_total = 0
        self.n_session = 0 #session += 1 whenever the limbs reset to neutral pos
        context = self.contexts[self.cc_idx] #current context
        make_rnn = True
        make_bg = True
        
        if self.loading_state:
            self.agent.actor.std_d = self.hparams[LOAD_NOISE]
            make_rnn,make_bg = self.agent.load_existing_ac(self.load_dir_label)
            #self.fix_weights = False
        
        if self.use_bg and make_bg:
            if verbose:
                print('altering contexts')
            self.add_new_col()
        


        
        #Needs to come after the above loading step
        #print(make_bg)
        self.agent.create_new_weights(make_rnn,make_bg)
        
        if verbose:
            print(context.color)
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
                if use_sleep:
                    time.sleep(1.0)
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
                if not done:
                    pos, context, done = self.main(context)
                    screen.fill(context.color)
                    pygame.draw.lines(screen,black,False,pos[0],5)#pos sliced at 0 since only one limb
                    if self.use_bg:
                        pygame.draw.circle(screen,context.color2,context.objects[0].location,5)
                    else:
                        pygame.draw.circle(screen,gray,context.objects[0].location,5)
                    pygame.display.update()
                    
        else:
            while not done:
                pos, context, done = self.main(context)
    
    def main(self,context):
        done = False
        round_complete = False
            
        self.agent.act(context)
        pos = self.get_euclidian_points(self.agent.c_pos,self.agent.n_limbs) #slice at 0
        self.agent.euclidian_pos = pos[0][self.agent.n_joints]
        reward = context.query_reward(pos,self.agent.n_limbs,self.agent.n_joints)
        self.agent.update_memory(reward)
        #check on resets
        self.n_step += 1
        self.n_step_total += 1
        if self.n_step >= self.max_steps:
            self.n_step = 0
            self.n_session += 1
            
            self.cc_idx += 1
            if(self.cc_idx >= len(self.contexts)):
                self.cc_idx = 0
            context = self.contexts[self.cc_idx]
            self.agent.neutral()
        
        if self.n_session >= self.max_sessions:
            done = True
            
        if self.n_step_total % self.update_freq == 0:
            #print('training')
            start = time.time()
            self.agent.train()
            end = time.time()
            #print('time to train...'+str(end-start))
        
        if context.check_met_goal() == True:
            if verbose:
                print(context.color)
            if not self.cc_idx in self.solved_context:
                self.solved_context.append(self.cc_idx)
                if self.fix_weights:
                    if verbose:
                        print('fixing weights for context: '+str(self.cc_idx))
                    self.agent.actor.fix_weights()
                    self.agent.critic.fix_weights()                                    
            if len(self.contexts) == len(self.solved_context):
                done = True
            #reduce the reward of a solved context to avoid repeatedly solving one context
            context.decay_reward()
        return pos, context, done
    
    def gen_env(self):
        minp = self.agent.min_position
        maxp = self.agent.max_position
        maxdist = 2*max(self.agent.total_limb_length)[0]
        for n in range(self.n_contexts):
            color = [random.randint(0,255),random.randint(0,255),random.randint(0,255)] #get a color
            for o in range(self.n_objects):
                location = self.get_euclidian_points([[random.randint(minp,maxp),random.randint(minp,maxp),random.randint(minp,maxp)]])
                location = location[0][self.agent.n_joints] #last euclidian point; not subtracting 1 becasue there is an extra base point at spot 0
                location_int = [int(np.round(location[0])),int(np.round(location[1]))]

                utility = random.randint(self.min_utility,self.max_utility)
                scale = random.uniform(maxdist/self.scale_ratio_max,maxdist/self.scale_ratio_min)
                threshold = random.uniform(maxdist/self.threshold_ratio_max,maxdist/self.threshold_ratio_min) #ratio of the maximum distance. future proofing.
            self.contexts.append(Context(color,[Object(utility,location_int)],[threshold,self.threshold_2,scale],self.hparams))
        if linux:
            with open(self.log_dir+'/contexts.pickle','wb') as f:
                pickle.dump(self.contexts,f)
        else:
            with open(self.log_dir+'\\contexts.pickle','wb') as f:
                pickle.dump(self.contexts,f)

    def load_env(self,dir_label):
        self.load_dir_label = dir_label
        self.loading_state = True
        self.agent.loading_state = True
        if linux:
            with open(dir_label+'/contexts.pickle','rb') as f:
                self.contexts = pickle.load(f)
        else:
            with open(dir_label+'\\contexts.pickle','rb') as f:
                self.contexts = pickle.load(f)
    #new
    def add_new_col(self):
        #add a second color and change object locations
        locations = []
        for c in self.contexts:
            color = [random.randint(0,255),random.randint(0,255),random.randint(0,255)] #get a color
            c.color2 = color
            for o in c.objects:
                locations.append(o.location)
        reverse = True
        displace = True
        disp_range = 30
        #exchange positions. Can also displace (commented out)
        #make this better, reverse the list
        for c in self.contexts:
            for o in c.objects:
                ##r = random.randint(len(locations))
                if reverse:
                    r = len(locations)-1
                    o.location = locations[r]
                    del locations[r]

                if displace:
                    disp = [random.randint(-1*disp_range,disp_range),random.randint(-1*disp_range,disp_range)]
                    o.location = [o.location[0]+disp[0],o.location[1]+disp[1]]
                #print(o.location)
        if linux:
            with open(self.log_dir+'/contexts.pickle','wb') as f:
                pickle.dump(self.contexts,f)
        else:
            with open(self.log_dir+'\\contexts.pickle','wb') as f:
                pickle.dump(self.contexts,f)
        if verbose:
            print('new context created and saved')
            print('displace: '+str(displace))
            print('reverse: '+str(reverse))
    
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
    
#e = Environment()
#e.gen_env()
#e.start()
#print(e.contexts[0].objects[0].location)

mv_envs = []
n = 0
trials = 1
if linux:
    with tf.summary.create_file_writer('/scratch/users/gchatt/logs').as_default():
        hp.hparams_config(
            hparams=[N_CONTEXTS,\
            N_OBJECTS,\
            MIN_UTILITY,\
            MAX_UTILITY,\
            SCALE_RATIO_MIN,\
            SCALE_RATIO_MAX,\
            THRESHOLD_RATIO_MIN,\
            THRESHOLD_RATIO_MAX,\
            THRESHOLD_2,\
            UPDATE_FREQ,\
            MAX_STEPS,\
            MAX_SESSIONS,\
            N_LIMBS,\
            N_JOINTS,\
            MIN_POSITION,\
            MAX_POSITION,\
            C_DIV,\
            MAX_GRAD,\
            LR_ACTOR,\
            LR_CRITIC,\
            GAMMA,\
            MAX_CRITIC_LOSS,\
            TDE_SCALE,\
            MAX_TDE,\
            ACTOR_DECAY_RATE,\
            CRITIC_DECAY_RATE,\
            MET_GOAL_CRITERIA,\
            MAX_TRIES,\
            CRITIC_UNITS,\
            CRITIC_NFIX,\
            BS_UNITS,\
            STR_UNITS,\
            GPI_UNITS, \
            ACTOR_NFIX,\
            MAX_FR,\
            STD_D,\
            TAU,\
            NOISE_SCALE,\
            NOISE_SCALE_2,\
            NOISE_BASE,\
            USE_BATCH_NORM, \
            LOAD_NOISE, \
            USE_CUST, \
            MIN_GOAL_SCALE, \
            MET_GOAL_DECAY, \
            LOADING_STATE, \
            USE_BG, \
            USE_BSL1RNN, \
            GPI_BIAS, \
            STR_NFIX, \
            GPI_TRAINABLE, \
            EXPAND_N, \
            QUANT_PERCENT, \
            MAX_R_RATIO, \
            BS_COMPLETED_LR],\
            metrics=[hp.Metric(METRIC_ACCURACY, display_name='Reward')]
        )
else:
    with tf.summary.create_file_writer(os.getcwd()+'\\logs').as_default():
        hp.hparams_config(
            hparams=[N_CONTEXTS,\
            N_OBJECTS,\
            MIN_UTILITY,\
            MAX_UTILITY,\
            SCALE_RATIO_MIN,\
            SCALE_RATIO_MAX,\
            THRESHOLD_RATIO_MIN,\
            THRESHOLD_RATIO_MAX,\
            THRESHOLD_2,\
            UPDATE_FREQ,\
            MAX_STEPS,\
            MAX_SESSIONS,\
            N_LIMBS,\
            N_JOINTS,\
            MIN_POSITION,\
            MAX_POSITION,\
            C_DIV,\
            MAX_GRAD,\
            LR_ACTOR,\
            LR_CRITIC,\
            GAMMA,\
            MAX_CRITIC_LOSS,\
            TDE_SCALE,\
            MAX_TDE,\
            ACTOR_DECAY_RATE,\
            CRITIC_DECAY_RATE,\
            MET_GOAL_CRITERIA,\
            MAX_TRIES,\
            CRITIC_UNITS,\
            CRITIC_NFIX,\
            BS_UNITS,\
            STR_UNITS,\
            GPI_UNITS, \
            ACTOR_NFIX,\
            MAX_FR,\
            STD_D,\
            TAU,\
            NOISE_SCALE,\
            NOISE_SCALE_2,\
            NOISE_BASE,\
            USE_BATCH_NORM, 
            LOAD_NOISE, \
            USE_CUST, \
            MIN_GOAL_SCALE, \
            MET_GOAL_DECAY, \
            LOADING_STATE, \
            USE_BG, \
            USE_BSL1RNN, \
            GPI_BIAS, \
            STR_NFIX, \
            GPI_TRAINABLE, \
            EXPAND_N, \
            QUANT_PERCENT, \
            MAX_R_RATIO, \
            BS_COMPLETED_LR],\
            metrics=[hp.Metric(METRIC_ACCURACY, display_name='Reward')]
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
        #print('here')

hp_mult_count = np.zeros(len(hp_mult))

if len(hp_mult) > 0:
    #print('here')
    done = False
    n = 0
    num_trials = 1
    while not done:
        #check if at max
        for i in range(len(hp_mult)):
            if int(hp_mult_count[i]+1) >= hp_mult_num[i]:
                done = True
            else:
                done = False
                break
        #continue and initiate threads
        for i in range(len(hp_mult)):
            hparams[hp_mult[i]] = hp_mult[i].domain.values[int(hp_mult_count[i])]
        for num_trial in range(num_trials):
            mv_envs.append(Environment(hparams))
            if not linux:
                mv_envs[n].daemon = True
            #mv_envs[n].gen_env()
            #print('here')
            #mv_envs[0].load_env(os.getcwd()+'\\logs\\20200729-211841357894')
            mv_envs[n].start()
            #print('here')
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
    mv_envs.append(Environment(hparams))
    if not linux:
        mv_envs[0].daemon = True
    #mv_envs[0].load_env(os.getcwd()+'\\logs\\20200729-211841357894')
    mv_envs[0].start()

if (not linux) and manual_exit:
	print('enter any input to exit')
	exit = input()
