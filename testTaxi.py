import gym
from wrap2 import derived as qlearn

num_trials = 1
env = gym.make('Taxi-v2')
ql = qlearn(500,6,0.99)

episodic_reward = []
n_episodes = 0
for i in range(num_trials):
	total_reward = 0
	env.reset()
	act = 0
	cond = True
	while cond:
#	while n_episodes < 1:
#		vals = env.step(env.action_space.sample())
		vals = env.step(act)
		if vals[-2] == True:
			n_episodes+=1
			print('steps-taken ',len(episodic_reward),'episode no.',n_episodes,' 100-step reward. ',sum(episodic_reward[-100:])+vals[1])
#why is steps-taken deterministic value of 200 steps ? Check...
			env.reset()
			episodic_reward.clear()
#			print('episode no.',n_episodes,' 100-step reward. ',sum(episodic_reward[-101:-1]))
#		total_reward+=vals[1]/10.0
		norm_reward = (vals[1]+10.0)/30
		act = ql.Act(norm_reward,vals[0])
#		episodic_reward.append(total_reward)
		episodic_reward.append(vals[1])
#		if sum(episodic_reward[-101:-1]) > 0.97:
#			break
#		print('curr-reward,n-state,n-action',vals[1],vals[0],act)
	ql.Reset()
#	episodic_reward.append(total_reward)

#print(episodic_reward[-1],n_episodes)
print(n_episodes)

