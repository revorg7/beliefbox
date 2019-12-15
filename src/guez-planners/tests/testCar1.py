import numpy as np
import sys
epsilon = 1e-7
#Only for Mountain car
class car_encoder:
	def __init__(self,bins=11):
		self.x_min = -1.2
		self.x_max = 0.6 + epsilon
		self.y_min = -0.07
		self.y_max = 0.07 + epsilon
		self.bins = bins
		self.binsX = np.linspace(self.x_min,self.x_max,self.bins)
		self.binsY = np.linspace(self.y_min,self.y_max,self.bins)
#		print(self.binsY)

	def encode(self,arr):
		x = arr[0]
		y = arr[1]
		if (x<self.x_min or x>self.x_max or y<self.y_min or y>self.y_max):
			raise ValueError('out of range inputs')
		s = np.digitize(x,self.binsX) - 1
		a = np.digitize(y,self.binsY) - 1
#		print(s,a)
		return s + a*(self.bins-1) #(bins-1) is due to the binning logic, where we take epsilon extra in the max-value of each dimension of input vector

discrteization = int(sys.argv[1])
encoder = car_encoder(discrteization+1) #Default is 10 values (bins - 1), check mountain-car.ipynb for more details
#print(encoder.encode(0.6,0.07))

#----------------------------------------------
from numpy import random
import gym
from wrap2 import derived as qlearn


env = gym.make('MountainCar-v0')
#env = gym.wrappers.Monitor(env, './videos/' + "myAgent" + '/')
ql = qlearn(discrteization**2,3,0.99,int(sys.argv[2]),int(sys.argv[3]),int(sys.argv[4]))
total_episodes = 10

num_trials = 1
counter = 0
episodic_reward = []
n_episodes = 0
for i in range(num_trials):
	total_reward = 0
	prev_act = 0
	prev_state = -1

	act = random.randint(3)
	cond = True
	reset_state = encoder.encode(env.reset())
	ql.Reset(reset_state)
	while cond:
#		env.render()
		vals = env.step(act)
		counter+=1
		norm_reward = vals[1] + 2.0
		curr_state = encoder.encode(vals[0])

		if n_episodes >= total_episodes:
			cond=False
		if vals[-2] == True:
#			cond=False
			n_episodes+=1
			print('steps-taken ',len(episodic_reward),'episode no.',n_episodes,' 100-step reward. ',sum(episodic_reward[-99:])+vals[1])
			episodic_reward.clear()

			if vals[0][0] >= 0.5:
				print("reached")
				ql.Observe(prev_state,prev_act,100,encoder.encode(vals[0]),act)
#				ql.Act(100,reset_state) #<< This shouldnt be, its giving +ve reward even to negative states
			else:
				ql.Observe(prev_state,prev_act,norm_reward,encoder.encode(vals[0]),act)

			curr_state = encoder.encode(env.reset())
			ql.Reset(curr_state)	#So that next Act() doesn't update the belief


		#epsilon-greedy
		if (random.uniform() < -0.2):
			act = random.randint(3)
			if vals[-2] != True:
				ql.Observe(prev_state,prev_act,norm_reward,curr_state,act)
			else:
				pass #no need to observe transition to reset_state, note that curr_state is anyways updated to reset_state due to "ql.Reset(curr_state)" above
		else:
			if not (counter % 4):
				act = ql.Act(norm_reward,curr_state) #This tuple will not update belief when reset is called above, it will only update current_state in tree,
													# which is anyways a repetition from "ql.Reset(curr_state)" above
			else:
				ql.Observe(prev_state,prev_act,norm_reward,curr_state,act)
				act = ql.getAction(curr_state)

		prev_state = curr_state
		prev_act = act

#		episodic_reward.append(total_reward)
		episodic_reward.append(vals[1])
#		if sum(episodic_reward[-101:-1]) > 0.97:
#			break
#		print('curr-reward,n-state,n-action',vals[1],vals[0],act)
#	ql.Reset()
#	episodic_reward.append(total_reward)

env.close()

#Note order of printing maybe wrong because I need to use fflush in c++
