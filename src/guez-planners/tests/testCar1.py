import numpy as np

#Only for Mountain car
class car_encoder:
	def __init__(self):
		self.x_min = -1.2
		self.x_max = 0.6
		self.y_min = -0.07
		self.y_max = 0.07
		self.bins = 16
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
		return s + a*self.bins


encoder = car_encoder()
#print(encoder.encode(0.6,0.07))

#----------------------------------------------
from numpy import random
import gym
from wrap2 import derived as qlearn


env = gym.make('MountainCar-v0')
#env = gym.wrappers.Monitor(env, './videos/' + "myAgent" + '/')
ql = qlearn(256,3,1.0)


num_trials = 1
counter = 0
episodic_reward = []
n_episodes = 0
for i in range(num_trials):
	total_reward = 0
	prev_act = -1
	prev_state = -1						

	act = random.randint(3)
	cond = True
	reset_state = encoder.encode(env.reset())
	ql.Reset(reset_state)
	while cond:
		env.render()
		vals = env.step(act)
		counter+=1
		norm_reward = vals[1]
		curr_state = encoder.encode(vals[0])

		if n_episodes >= 1000:
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
			act = ql.Act(norm_reward,curr_state) #This tuple will not update belief when reset is called above, it will only update current_state in tree,
													# which is anyways a repetition from "ql.Reset(curr_state)" above

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
