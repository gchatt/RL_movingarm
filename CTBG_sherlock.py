#Custom Layer Test

import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.constraints import non_neg
import numpy as np
from numpy import random
import time
import copy
import scipy
from scipy import sparse

#unit1 = 200 #striatum division size (multiply by 3 for total size)
#unit2 = 100 #other nuclei division size (multiply by 3 for total size)
#unit3 = 300 #premotor and motor cortex layer size
use_constraints = True


class Striatum(layers.Layer):
	def __init__(self, units,input_dim=15):
		super(Striatum, self).__init__()
		unit1 = units[0]
		unit2 = units[1]
		unit3 = units[2]
		self.units = unit1*3
		self.input_dim = input_dim
		self.training = True
		w_init = tf.random_uniform_initializer(minval=0., maxval=0.01)
		z_init = tf.zeros_initializer()
		str_bias = [0,0]
		b_init = tf.random_uniform_initializer(minval=str_bias[0],maxval=str_bias[1])
		#self.w = []
		limit_1 = unit1
		limit_2 = unit1*2
		limit_3 = unit1*3
		goal_len = 3
		color_len = 6
		pos_len = 9
		act_len = 15
		self.dval=[];
		self.unit_diags = []
		for unit in range(self.units):
			#for each unit, rand int, 50%, make it D1 or D2
			if random.randint(2) > 0:
				self.dval.append(1)
			else:
				self.dval.append(2)
			unit_diag = []
			for input in range(self.input_dim):
				#D1
				if unit in range(limit_1):
					if input in range(goal_len):
						#100% goal in D1
						unit_diag.append(1)
					elif input in range(goal_len,color_len):
						#40% color in D1
						if random.randint(5) > 2:
							unit_diag.append(1)
						else:
							unit_diag.append(0)	
					elif input in range(color_len,pos_len):
						#10% pos in D1
						if random.randint(10) > 8:
							unit_diag.append(1)
						else:
							unit_diag.append(0)	
					elif input in range(pos_len,act_len):
						#0% last action in D1
						unit_diag.append(0)	
				#D2
				elif unit in range(limit_1,limit_2):
					if input in range(goal_len):
						#%50 goal in D2
						if random.randint(10) > 4:
							unit_diag.append(1)
						else:
							unit_diag.append(0)	
					elif input in range(goal_len,color_len):
						#80% color in D2
						if random.randint(5) > 0:
							unit_diag.append(1)
						else:
							unit_diag.append(0)	
					elif input in range(color_len,pos_len):
						#60% pos in D2
						if random.randint(5) > 1:
							unit_diag.append(1)
						else:
							unit_diag.append(0)	
					elif input in range(pos_len,act_len):
						#20% last action to D1
						if random.randint(10) > 7:
							unit_diag.append(1)
						else:
							unit_diag.append(0)	
				#D3
				elif unit in range(limit_2,limit_3):
					if input in range(goal_len):
						#%0 goal in D2
						unit_diag.append(0)
					elif input in range(goal_len,color_len):
						#20% color in D2
						if random.randint(5) > 3:
							unit_diag.append(1)
						else:
							unit_diag.append(0)	
					elif input in range(color_len,pos_len):
						#100% pos in D2
						unit_diag.append(1)
					elif input in range(pos_len,act_len):
						#100% last action to D1
						unit_diag.append(1)
			self.unit_diags.append(np.array(unit_diag))
		self.unit_diags = np.array(self.unit_diags)
		#biases -> Consider adding a negative bias
		if use_constraints:
			self.w = self.add_weight(shape=(self.input_dim,self.units),initializer=w_init,trainable=True,constraint=non_neg())
		else:
			self.w = self.add_weight(shape=(self.input_dim,self.units),initializer=w_init,trainable=True)
		self.b = self.add_weight(shape=(self.units,), initializer=b_init, trainable=True)
		self.s = tf.convert_to_tensor(np.array([sparse.diags(self.unit_diags[i],0).toarray() for i in range(len(self.unit_diags))]),dtype=tf.float32)
	
	def call(self,inputs):
		#concat step is o(n), not sustainable, not usable
		#sparse mat mult is >10 times faster than 'masking' directly
		p = tf.transpose(tf.linalg.matmul(self.s,tf.transpose(inputs)),perm=[2,0,1]) #mask the inputs
		f = tf.linalg.diag_part(tf.linalg.matmul(p,self.w)) + self.b
		return f

class GPe(layers.Layer):
	def __init__(self,units,str_dval=[]):
		super(GPe,self).__init__()
		unit1 = units[0]
		unit2 = units[1]
		unit3 = units[2]
		self.units = unit2*3
		self.input_dim = unit1*3
		self.training = True
		w_init = tf.random_uniform_initializer(minval=0., maxval=0.01)
		z_init = tf.zeros_initializer()
		gpe_bias = [0,0]
		b_init = tf.random_uniform_initializer(minval=gpe_bias[0],maxval=gpe_bias[1]) #give them a positive bias; they will 'output' at baseline #will be a tough hyperparam to tune
		#split GPe into thirds. The middle layer is really a transition layer is all
		limit_1 = unit2
		limit_2 = unit2*2
		limit_3 = unit2*3
		#dorsal striatum input
		ds1_len = unit1
		ds2_len = unit1*2
		ds3_len = unit1*3
		lateral_len = self.units #will use in future; will have to sum input_dim too, input_dim += units
		self.str_dval = str_dval;
		self.unit_diags = []
		for unit in range(self.units):
			unit_diag = []
			for input in range(self.input_dim):
				if self.str_dval[input] == 1 and random.randint(2) > 0: #if D1, only 50% pass through
					unit_diag.append(0)
					continue
				#GPe_1
				if unit in range(limit_1):
					if input in range(ds1_len):
						#100% of dorsal striatum_1 -> GPe_1
						unit_diag.append(1)
					elif input in range(ds1_len,ds2_len):
						#50% ds_2 -> GPe_1
						if random.randint(2) > 0:
							unit_diag.append(1)
						else:
							unit_diag.append(0)
					elif input in range(ds2_len,ds3_len):
						#0% of ds_3 -> GPe_1
						unit_diag.append(0)
				#GPe_2
				elif unit in range(limit_1,limit_2):
					if input in range(ds1_len):
						#50% ds_1 -> GPe_2
						if random.randint(2) > 0:
							unit_diag.append(1)
						else:
							unit_diag.append(0)
					elif input in range(ds1_len,ds2_len):
						#75% ds_2 -> GPe_2
						if random.randint(4) > 0:
							unit_diag.append(1)
						else:
							unit_diag.append(0)
					elif input in range(ds2_len,ds3_len):
						#50% ds_3 -> GPe_2
						if random.randint(2) > 0:
							unit_diag.append(1)
						else:
							unit_diag.append(0)
				#GPe_3
				elif unit in range(limit_2,limit_3):
					if input in range(ds1_len):
						#0% ds_1 -> GPe_3
						unit_diag.append(0)
					elif input in range(ds1_len,ds2_len):
						#50% ds_2 -> GPe_3
						if random.randint(2) > 0:
							unit_diag.append(1)
						else:
							unit_diag.append(0)
					elif input in range(ds2_len,ds3_len):
						#100% ds_3 -> GPe_3
						unit_diag.append(1)
			self.unit_diags.append(np.array(unit_diag))
		self.unit_diags = np.array(self.unit_diags)
		if use_constraints:
			self.w = self.add_weight(shape=(self.input_dim,self.units),initializer=w_init,trainable=True,constraint=non_neg())
		else:
			self.w = self.add_weight(shape=(self.input_dim,self.units),initializer=w_init,trainable=True)
		#positive biases				
		self.b = self.add_weight(shape=(self.units,), initializer=b_init, trainable=True)
		self.s = tf.convert_to_tensor(np.array([sparse.diags(self.unit_diags[i],0).toarray() for i in range(len(self.unit_diags))]),dtype=tf.float32)
				
	def call(self,inputs):
		#self.w = tf.math.negative(self.w)
		p = tf.transpose(tf.linalg.matmul(self.s,tf.transpose(inputs)),perm=[2,0,1]) #mask the inputs
		f = tf.linalg.diag_part(tf.linalg.matmul(p,self.w)) + self.b
		return f			
					
class STN(layers.Layer):
	def __init__(self,units):
		super(STN,self).__init__()
		unit1 = units[0]
		unit2 = units[1]
		unit3 = units[2]
		self.units = unit2*3
		self.input_dim = unit2*3
		self.training = True
		w_init = tf.random_uniform_initializer(minval=0., maxval=0.01)
		z_init = tf.zeros_initializer()
		stn_bias = [0,0]
		b_init = tf.random_uniform_initializer(minval=stn_bias[0],maxval=stn_bias[1]) #give them a positive bias; they will 'output' at baseline #will be a tough hyperparam to tune
		#split STN into thirds. The middle layer is a transition layer
		limit_1 = unit2
		limit_2 = unit2*2
		limit_3 = unit2*3
		#GPe input
		gpe_1_len = unit2
		gpe_2_len = unit2*2
		gpe_3_len = unit2*3
		self.unit_diags = []
		for unit in range(self.units):
			unit_diag = []
			for input in range(self.input_dim):
				#STN_1
				if unit in range(limit_1):
					if input in range(gpe_1_len):
						#100% of gpe_1 -> STN_1
						unit_diag.append(1)
					elif input in range(gpe_1_len,gpe_2_len):
						#50% gpe_2 -> STN_1
						if random.randint(2) > 0:
							unit_diag.append(1)
						else:
							unit_diag.append(0)
					elif input in range(gpe_2_len,gpe_3_len):
						#0% of gpe_3 -> STN_1
						unit_diag.append(0)
				#STN_2
				elif unit in range(limit_1,limit_2):
					if input in range(gpe_1_len):
						#50% gpe_2 -> STN_2
						if random.randint(2) > 0:
							unit_diag.append(1)
						else:
							unit_diag.append(0)
					elif input in range(gpe_1_len,gpe_2_len):
						#75% gpe_2 -> STN_2
						if random.randint(4) > 0:
							unit_diag.append(1)
						else:
							unit_diag.append(0)
					elif input in range(gpe_2_len,gpe_3_len):
						#50% gpe_3 -> STN_2
						if random.randint(2) > 0:
							unit_diag.append(1)
						else:
							unit_diag.append(0)
				#STN_3
				elif unit in range(limit_2,limit_3):
					if input in range(gpe_1_len):
						#0% gpe_1 -> STN_3
						unit_diag.append(0)
					elif input in range(gpe_1_len,gpe_2_len):
						#50% gpe_2 -> STN_3
						if random.randint(2) > 0:
							unit_diag.append(1)
						else:
							unit_diag.append(0)
					elif input in range(gpe_2_len,gpe_3_len):
						#100% gpe_3 -> STN_3
						unit_diag.append(1)
			self.unit_diags.append(np.array(unit_diag))
		self.unit_diags = np.array(self.unit_diags)
		if use_constraints:
			self.w = self.add_weight(shape=(self.input_dim,self.units),initializer=w_init,trainable=True,constraint=non_neg())
		else:
			self.w = self.add_weight(shape=(self.input_dim,self.units),initializer=w_init,trainable=True)
		#biases				
		self.b = self.add_weight(shape=(self.units,), initializer=b_init, trainable=True)
		self.s = tf.convert_to_tensor(np.array([sparse.diags(self.unit_diags[i],0).toarray() for i in range(len(self.unit_diags))]),dtype=tf.float32)
				
	def call(self,inputs):
		p = tf.transpose(tf.linalg.matmul(self.s,tf.transpose(inputs)),perm=[2,0,1]) #mask the inputs
		f = tf.linalg.diag_part(tf.linalg.matmul(p,self.w)) + self.b
		return f

class GPi(layers.Layer):
	#input from dorsal striatum and STN
	def __init__(self,units,str_dval=[]):
		super(GPi,self).__init__()
		unit1 = units[0]
		unit2 = units[1]
		unit3 = units[2]
		self.units = unit2*3
		self.input_dim = (unit1+unit2)*3
		self.training = True
		w_init = tf.random_uniform_initializer(minval=0., maxval=0.01)
		z_init = tf.zeros_initializer()
		gpi_bias = [0,0]
		b_init = tf.random_uniform_initializer(minval=gpi_bias[0],maxval=gpi_bias[1]) #give them a positive bias; they will 'output' at baseline #will be a tough hyperparam to tune
		#split GPi into thirds.
		limit_1 = unit2
		limit_2 = unit2*2
		limit_3 = unit2*3
		#dorsal striatum input
		self.ds1_len = unit1
		self.ds2_len = unit1*2
		self.ds3_len = unit1*3
		self.stn1_len = unit1*3 + unit2
		self.stn2_len = unit1*3 + unit2*2
		self.stn3_len = unit1*3 + unit2*3
		self.str_dval = str_dval;
		self.unit_diags = []
		for unit in range(self.units):
			unit_diag = []
			for input in range(self.input_dim):
				if input < len(self.str_dval):
					if self.str_dval[input] == 2: #if D2, don't pass through
						unit_diag.append(0)
						continue
				#GPi_1
				if unit in range(limit_1):
					if input in range(self.ds1_len):
						#100% dorsal striatum_1 -> GPi_1
						unit_diag.append(1)
					elif input in range(self.ds1_len,self.ds2_len):
						#50% self.ds_2 -> GPi_1
						if random.randint(2) > 0:
							unit_diag.append(1)
						else:
							unit_diag.append(0)
					elif input in range(self.ds2_len,self.ds3_len):
						#0% self.ds_3 -> GPi_1
						unit_diag.append(0)
					elif input in range(self.ds3_len,self.stn1_len):
						#100% self.stn_1 -> GPi_1
						unit_diag.append(1)
					elif input in range(self.stn1_len,self.stn2_len):
						#50% self.stn_2 -> GPi_1
						if random.randint(2) > 0:
							unit_diag.append(1)
						else:
							unit_diag.append(0)
					elif input in range(self.stn2_len,self.stn3_len):
						#0% of self.stn_3 -> GPi_1
						unit_diag.append(0)
				#GPi_2
				elif unit in range(limit_1,limit_2):
					if input in range(self.ds1_len):
						#50% self.ds_1 -> GPi_2
						if random.randint(2) > 0:
							unit_diag.append(1)
						else:
							unit_diag.append(0)
					elif input in range(self.ds1_len,self.ds2_len):
						#75% self.ds_2 -> GPi_2
						if random.randint(4) > 0:
							unit_diag.append(1)
						else:
							unit_diag.append(0)
					elif input in range(self.ds2_len,self.ds3_len):
						#50% self.ds_3 -> GPi_2
						if random.randint(2) > 0:
							unit_diag.append(1)
						else:
							unit_diag.append(0)
					elif input in range(self.stn1_len):
						#50% self.stn_1 -> GPi_2
						if random.randint(2) > 0:
							unit_diag.append(1)
						else:
							unit_diag.append(0)
					elif input in range(self.stn1_len,self.stn2_len):
						#75% self.stn_2 -> GPi_2
						if random.randint(4) > 0:
							unit_diag.append(1)
						else:
							unit_diag.append(0)
					elif input in range(self.stn2_len,self.stn3_len):
						#50% self.stn_3 -> GPi_2
						if random.randint(2) > 0:
							unit_diag.append(1)
						else:
							unit_diag.append(0)
				#GPi_3
				elif unit in range(limit_2,limit_3):
					if input in range(self.ds1_len):
						#0% self.ds_1 -> GPi_3
						unit_diag.append(0)
					elif input in range(self.ds1_len,self.ds2_len):
						#50% self.ds_2 -> GPi_3
						if random.randint(2) > 0:
							unit_diag.append(1)
						else:
							unit_diag.append(0)
					elif input in range(self.ds2_len,self.ds3_len):
						#100% self.ds_3 -> GPi_3
						unit_diag.append(1)
					elif input in range(self.stn1_len):
						#0% self.stn_1 -> GPi_3
						unit_diag.append(0)
					elif input in range(self.stn1_len,self.stn2_len):
						#50% self.stn_2 -> GPi_3
						if random.randint(2) > 0:
							unit_diag.append(1)
						else:
							unit_diag.append(0)
					elif input in range(self.stn2_len,self.stn3_len):
						#100% self.stn_3 -> GPi_3
						unit_diag.append(1)
		
			self.unit_diags.append(np.array(unit_diag))
		self.unit_diags = np.array(self.unit_diags)
		if use_constraints:
			self.w = self.add_weight(shape=(self.input_dim,self.units),initializer=w_init,trainable=True,constraint=non_neg())
		else:
			self.w = self.add_weight(shape=(self.input_dim,self.units),initializer=w_init,trainable=True)
		#positive biases
		self.b = self.add_weight(shape=(self.units,), initializer=b_init, trainable=True)
		self.s = tf.convert_to_tensor(np.array([sparse.diags(self.unit_diags[i],0).toarray() for i in range(len(self.unit_diags))]),dtype=tf.float32)
				
	def call(self,inputs):
		p = tf.transpose(tf.linalg.matmul(self.s,tf.transpose(inputs)),perm=[2,0,1]) #mask the inputs
		f = tf.linalg.diag_part(tf.linalg.matmul(p,self.w)) + self.b
		return f

class Motor_thal(layers.Layer):
	def __init__(self,units):
		super(Motor_thal,self).__init__()
		unit1 = units[0]
		unit2 = units[1]
		unit3 = units[2]
		self.units = unit2*3
		self.input_dim = (unit2*3)+unit3
		self.training = True
		w_init = tf.random_uniform_initializer(minval=0., maxval=0.01)
		z_init = tf.zeros_initializer()
		mthal_bias = [0,0]
		b_init = tf.random_uniform_initializer(minval=mthal_bias[0],maxval=mthal_bias[1])
		#split mthal into thirds. The middle layer is a transition layer
		limit_1 = unit2
		limit_2 = unit2*2
		limit_3 = unit2*3
		#GPi input
		self.gpi_1_len = unit2
		self.gpi_2_len = unit2*2
		self.gpi_3_len = unit2*3
		#premotor_input
		self.premotor_1_len = (unit2*3) + unit3
		self.unit_diags = []
		for unit in range(self.units):
			unit_diag = []
			for input in range(self.input_dim):
				#mthal_1
				if unit in range(limit_1):
					if input in range(self.gpi_1_len):
						#100% of self.gpi_1 -> mthal_1
						unit_diag.append(1)
					elif input in range(self.gpi_1_len,self.gpi_2_len):
						#50% self.gpi_2 -> mthal_1
						if random.randint(2) > 0:
							unit_diag.append(1)
						else:
							unit_diag.append(0)
					elif input in range(self.gpi_2_len,self.gpi_3_len):
						#0% of self.gpi_3 -> mthal_1
						unit_diag.append(0)
					elif input in range(self.gpi_3_len,self.premotor_1_len):
						#100% self.premotor1 -> mthal_1
						unit_diag.append(1)
				#mthal_2
				elif unit in range(limit_1,limit_2):
					if input in range(self.gpi_1_len):
						#50% self.gpi_2 -> mthal_2
						if random.randint(2) > 0:
							unit_diag.append(1)
						else:
							unit_diag.append(0)
					elif input in range(self.gpi_1_len,self.gpi_2_len):
						#75% self.gpi_2 -> mthal_2
						if random.randint(4) > 0:
							unit_diag.append(1)
						else:
							unit_diag.append(0)
					elif input in range(self.gpi_2_len,self.gpi_3_len):
						#50% self.gpi_3 -> mthal_2
						if random.randint(2) > 0:
							unit_diag.append(1)
						else:
							unit_diag.append(0)
					elif input in range(self.gpi_3_len,self.premotor_1_len):
						#90% self.premotor1 -> mthal_2
						if random.randint(10) > 0:
							unit_diag.append(1)
						else:
							unit_diag.append(0)
				#mthal_3
				elif unit in range(limit_2,limit_3):
					if input in range(self.gpi_1_len):
						#0% self.gpi_1 -> mthal_3
						unit_diag.append(0)
					elif input in range(self.gpi_1_len,self.gpi_2_len):
						#50% self.gpi_2 -> mthal_3
						if random.randint(2) > 0:
							unit_diag.append(1)
						else:
							unit_diag.append(0)
					elif input in range(self.gpi_2_len,self.gpi_3_len):
						#100% self.gpi_3 -> mthal_3
						unit_diag.append(1)
					elif input in range(self.gpi_3_len,self.premotor_1_len):
						#80% self.premotor1 -> mthal_3
						if random.randint(10) > 1:
							unit_diag.append(1)
						else:
							unit_diag.append(0)
			self.unit_diags.append(np.array(unit_diag))
		self.unit_diags = np.array(self.unit_diags)
		if use_constraints:
			self.w = self.add_weight(shape=(self.input_dim,self.units),initializer=w_init,trainable=True,constraint=non_neg())
		else:
			self.w = self.add_weight(shape=(self.input_dim,self.units),initializer=w_init,trainable=True)
		#biases				
		self.b = self.add_weight(shape=(self.units,), initializer=b_init, trainable=True)
		self.s = tf.convert_to_tensor(np.array([sparse.diags(self.unit_diags[i],0).toarray() for i in range(len(self.unit_diags))]),dtype=tf.float32)
				
	def call(self,inputs):
		p = tf.transpose(tf.linalg.matmul(self.s,tf.transpose(inputs)),perm=[2,0,1]) #mask the inputs
		f = tf.linalg.diag_part(tf.linalg.matmul(p,self.w)) + self.b
		return f
		
class Premotor(layers.Layer):
	def __init__(self,units,input_dim=9,batch_norm=False):
		super(Premotor,self).__init__()
		self.input_dim = input_dim
		self.batch_norm = batch_norm
		unit1 = units[0]
		unit2 = units[1]
		unit3 = units[2]
		l1_sz = unit3
		l2_sz = unit3
		self.l1 = layers.Dense(l1_sz,activation='sigmoid')
		if batch_norm:
			self.bna = tf.keras.layers.BatchNormalization()
		self.l2 = layers.Dense(l2_sz,activation='sigmoid')
	
	def call(self,inputs):
		#takes goal (3), color(3), c_arm_pos(3) -> shape (1,9)
		x = self.l1(inputs)
		if self.batch_norm:
			x = self.bna(x,training=True)
		x = self.l2(x)
		return x
		
class Motor_cortex(layers.Layer):
	def __init__(self,units,action_sz=6,batch_norm=False):
		super(Motor_cortex,self).__init__()
		self.batch_norm = batch_norm
		unit1 = units[0]
		unit2 = units[1]
		unit3 = units[2]
		l1_sz = unit3
		l2_sz = unit3
		self.l1 = layers.Dense(l1_sz,activation='relu')
		if batch_norm:
			self.bna = tf.keras.layers.BatchNormalization()
			self.bnb = tf.keras.layers.BatchNormalization()
		self.l2 = layers.Dense(l2_sz,activation='relu')
		self.lout = layers.Dense(action_sz,activation='relu');
		#self(tf.convert_to_tensor([np.zeros(state_sz,dtype='float32')]));
	
	def call(self,inputs,bnorm):
		#takes input from mthal. In future, should get some input from premotor directly as well
		x = self.l1(inputs);
		if self.batch_norm:
			x = self.bna(x,training=bnorm)
		x = self.l2(x);
		if self.batch_norm:
			x = self.bnb(x,training=bnorm)
		action = self.lout(x)
		return action
		
#I need to somehow add a random number of lateral inputs.....makes it very complex
class CTBG(keras.Model):
	def __init__(self,summary_writer,units,tau,std_mc):
		super(CTBG,self).__init__()
		#self.hparams = hparams
		self.units = units
		
		self.use_all = False
		self.use_batch_norm = True
		
		if self.use_all:
			self.dorsal_striatum = Striatum(units)
			self.gpe = GPe(units,str_dval=self.dorsal_striatum.dval)
			self.stn = STN(units)
			self.gpi = GPi(units,str_dval=self.dorsal_striatum.dval)
			self.mthal = Motor_thal(units)
			self.premotor = Premotor(units)
			self.mctx = Motor_cortex(units,batch_norm=self.use_batch_norm)
		else:
			self.mctx = Motor_cortex(units,batch_norm=self.use_batch_norm)
		
		self.std_all = 0.1
		self.std_str = 0.5
		self.std_mc = float(std_mc)
		self.std_mc_init = self.std_mc
		self.tau = float(tau)
		self.beta = 0.001
		
		# self.bna = tf.keras.layers.BatchNormalization()
		# self.bnb = tf.keras.layers.BatchNormalization()
		# self.bnc = tf.keras.layers.BatchNormalization()
		# self.bnd = tf.keras.layers.BatchNormalization()
		# self.bne = tf.keras.layers.BatchNormalization()
		# self.bnf = tf.keras.layers.BatchNormalization()
		
		self.summary_writer = summary_writer
		self.n_step = 0
		self.max_fr = 200
		
		self.inhibitory = -1 #set to 1 if you don't want to use it
	
	def call(self,str_inputs,prem_inputs,loss,use_noisy_relaxation,bnorm):
	
		std_all = self.std_all
		std_str = self.std_str
		std_mc = self.std_mc
		
		if use_noisy_relaxation:
			#std_all = std_all * np.exp(loss/self.tau)
			#std_str = std_str * np.exp(loss/self.tau)
			std_mc = max(min(std_mc - self.beta*(loss - self.tau),self.std_mc_init),1.)
			self.std_mc = std_mc
	
		
		self.gn = layers.GaussianNoise(stddev=std_all)
		self.gn_str = layers.GaussianNoise(stddev=std_str)
		self.gn_mc = layers.GaussianNoise(stddev=std_mc)
	
		if self.use_all:
			
			str_out = self.dorsal_striatum(str_inputs)
			if use_noisy_relaxation:
				str_out = self.gn_str(str_out)
			#str_out = self.bna(str_out,training=True)
			str_out = tf.nn.sigmoid(str_out)
			#str_out = tf.nn.relu(str_out)
			#str_out = tf.clip_by_value(str_out,0,self.max_fr)
			str_out = str_out * self.inhibitory
			#print(str_out)
			
			gpe_out = self.gpe(str_out)
			if use_noisy_relaxation:
				gpe_out = self.gn(gpe_out)
			#gpe_out = self.bnb(gpe_out,training=True)
			gpe_out = tf.nn.sigmoid(gpe_out)
			#gpe_out = tf.nn.relu(gpe_out)
			#gpe_out = tf.clip_by_value(gpe_out,0,self.max_fr)
			gpe_out = gpe_out * self.inhibitory
			
			stn_out = self.stn(gpe_out)
			if use_noisy_relaxation:
				stn_out = self.gn(stn_out)
			#stn_out = self.bnc(stn_out,training=True)
			stn_out = tf.nn.sigmoid(stn_out)
			#stn_out = tf.nn.relu(stn_out)
			#stn_out = tf.clip_by_value(stn_out,0,self.max_fr)
			
			gpi_out = self.gpi(tf.concat([str_out,stn_out],axis=1))
			if use_noisy_relaxation:
				gpi_out = self.gn(gpi_out)
			#gpi_out = self.bnd(gpi_out,training=True)
			gpi_out = tf.nn.sigmoid(gpi_out)		
			#gpi_out = tf.nn.relu(gpi_out)
			#gpi_out = tf.clip_by_value(gpi_out,0,self.max_fr)
			gpi_out = gpi_out * self.inhibitory
			
			prem_out = self.premotor(prem_inputs)
			if use_noisy_relaxation:
				prem_out = self.gn(prem_out)
			prem_out = tf.clip_by_value(prem_out,0,self.max_fr)
			#prem_out = tf.clip_by_value(prem_out,0,self.max_fr)
			
			mthal_out = self.mthal(tf.concat([gpi_out,prem_out],axis=1))
			if use_noisy_relaxation:
				mthal_out = self.gn(mthal_out)
			#mthal_out = self.bne(mthal_out,training=True)
			mthal_out = tf.nn.sigmoid(mthal_out)
			#mthal_out = tf.nn.relu(mthal_out)
			#mthal_out = tf.clip_by_value(mthal_out,0,self.max_fr)
			#print(tf.math.reduce_min(mthal_out).numpy())
			
			mctx_out = self.mctx(mthal_out,bnorm)
			if use_noisy_relaxation:
				mctx_out = self.gn_mc(mctx_out)
			mctx_out = tf.clip_by_value(mctx_out,0,self.max_fr)
		else:
			mctx_out = self.mctx(prem_inputs,bnorm)
			if use_noisy_relaxation:
				mctx_out = self.gn_mc(mctx_out)
			mctx_out = tf.clip_by_value(mctx_out,0,self.max_fr)
		
		
		#str_out = self.gn_str(self.bna(tf.nn.sigmoid(self.dorsal_striatum(str_inputs)),training=True))*(self.inhibitory) #input size = 15
		#tf.nn.sigmoid(self.bna(self.gn_str(self.dorsal_striatum(str_inputs)),training=True))*(self.inhibitory)
		#gpe_out = self.gn(self.bnb(tf.nn.sigmoid(self.gpe(str_out)),training=True))*(self.inhibitory)
		#gpe_out = tf.nn.sigmoid(self.bnb(self.gn(self.gpe(str_out),training=True)))*(self.inhibitory)
		#stn_out = self.gn(self.bnc(tf.nn.sigmoid(self.stn(gpe_out)),training=True))
		#stn_out = tf.nn.sigmoid(self.bnc(self.gn(self.stn(gpe_out)),training=True))
		
		#gpi_out = self.gn(self.bnd(tf.nn.sigmoid(self.gpi(tf.concat([str_out,stn_out],axis=1))),training=True))*(self.inhibitory)
		#gpi_out = tf.nn.sigmoid(self.bnd(self.gn(self.gpi(tf.concat([str_out,stn_out],axis=1))),training=True))*(self.inhibitory)
		
		#prem_out = self.bnf(self.gn(self.premotor(prem_inputs)),training=True)
		#mthal_out = self.gn(self.bne(tf.nn.sigmoid(self.mthal(tf.concat([gpi_out,prem_out],axis=1))),training=True))
		#mthal_out = tf.nn.sigmoid(self.bne(self.gn(self.mthal(tf.concat([gpi_out,prem_out],axis=1))),training=True))
		
		if self.use_all:
			with self.summary_writer.as_default():
				tf.summary.histogram('Dorsal Striatum Out',str_out,step=self.n_step)
				tf.summary.histogram('GPe Out',gpe_out,step=self.n_step)
				tf.summary.histogram('GPi Out',gpi_out,step=self.n_step)
				tf.summary.histogram('STN Out',stn_out,step=self.n_step)
				tf.summary.histogram('Motor Thalamus Out',mthal_out,step=self.n_step)
				tf.summary.histogram('Premotor Cortex Out',prem_out,step=self.n_step)
				tf.summary.histogram('Motor Cortex Out',mctx_out,step=self.n_step)
		else:
			with self.summary_writer.as_default():
				tf.summary.histogram('Motor Cortex Out',mctx_out,step=self.n_step)
		
		self.n_step += 1
		
		
		return mctx_out
	
	def log(self,n_train):
		if self.use_all:
			with self.summary_writer.as_default():
				tf.summary.histogram('Dorsal Striatum Weights',self.dorsal_striatum.trainable_weights[0],step=n_train)
				tf.summary.histogram('GPe Weights',self.gpe.trainable_weights[0],step=n_train)
				tf.summary.histogram('GPi Weights',self.gpi.trainable_weights[0],step=n_train)
				tf.summary.histogram('STN Weights',self.stn.trainable_weights[0],step=n_train)
				tf.summary.histogram('Motor Thalamus Weights',self.mthal.trainable_weights[0],step=n_train)
				tf.summary.histogram('Premotor Cortex Weights',self.premotor.trainable_weights[0],step=n_train)
				tf.summary.histogram('Motor Cortex Weights',self.mctx.trainable_weights[0],step=n_train)
				tf.summary.histogram('Dorsal Striatum Biases',tf.keras.backend.flatten(self.dorsal_striatum.trainable_weights[1]),step=n_train)
				tf.summary.histogram('GPe Biases',self.gpe.trainable_weights[1],step=n_train)
				tf.summary.histogram('GPi Biases',self.gpi.trainable_weights[1],step=n_train)
				tf.summary.histogram('STN Biases',self.stn.trainable_weights[1],step=n_train)
				tf.summary.histogram('Motor Thalamus Biases',self.mthal.trainable_weights[1],step=n_train)
				tf.summary.histogram('Premotor Cortex Biases',self.premotor.trainable_weights[1],step=n_train)
				tf.summary.histogram('Motor Cortex Biases',self.mctx.trainable_weights[1],step=n_train)
		else:
			with self.summary_writer.as_default():
				tf.summary.histogram('Motor Cortex Weights',self.mctx.trainable_weights[0],step=n_train)
				tf.summary.histogram('Motor Cortex Biases',self.mctx.trainable_weights[1],step=n_train)
				tf.summary.scalar('Noise',self.std_mc,step=n_train)
