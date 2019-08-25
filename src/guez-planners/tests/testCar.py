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
ql = qlearn(256,3,0.99)


num_trials = 1
counter = 0
episodic_reward = []
n_episodes = 0
for i in range(num_trials):
	total_reward = 0
	env.reset()
	act = 0
	prev_act = 0
	prev_state = -1						
	ql.Act(-1.0,0)	#To initialize root policy
	cond = True
	reset_state = -1
	while cond:
#	while n_episodes < 1:
#		vals = env.step(env.action_space.sample())
		env.render()
		vals = env.step(act)
		counter+=1
		if n_episodes >= 500:
			cond=False
		if vals[-2] == True:
#			cond=False
			n_episodes+=1
			print('steps-taken ',len(episodic_reward),'episode no.',n_episodes,' 100-step reward. ',sum(episodic_reward[-99:])+vals[1])
#why is steps-taken deterministic value of 200 steps ? Check...
			reset_state = encoder.encode(env.reset())
			episodic_reward.clear()
#			ql.Act(100,reset_state)	<< Dont do this update, as it confuses the algorithm: puts +ve probabs for many left states to come to center with last action taken before episode finish
#			print('episode no.',n_episodes,' 100-step reward. ',sum(episodic_reward[-101:-1]))
#		total_reward+=vals[1]/10.0
		norm_reward = vals[1]
		act = ql.Act(norm_reward,encoder.encode(vals[0])) #<< this is wrong, doing double-updates here, need to seperate act from observe function
#		act = ql.getAction(encoder.encode(vals[0]))
		if (random.uniform() < 0.2):
			act = random.randint(3) #rand required
#		if (vals[-2] != True):
#			ql.Observe(prev_state,prev_act,norm_reward,encoder.encode(vals[0]),act)
		if reset_state!=-1:
			prev_state = reset_state
			reset_state = -1
		else:
			prev_state = encoder.encode(vals[0])
		prev_act = act

#		episodic_reward.append(total_reward)
		episodic_reward.append(vals[1])
#		if sum(episodic_reward[-101:-1]) > 0.97:
#			break
#		print('curr-reward,n-state,n-action',vals[1],vals[0],act)
	ql.Reset()
#	episodic_reward.append(total_reward)

