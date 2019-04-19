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
	for _ in range(100):
#		vals = env.step(env.action_space.sample())
		vals = env.step(act)
		if vals[-2] == True:
			n_episodes+=1
		total_reward+=vals[1]/10.0
		act = ql.Act(vals[1]/10.0,vals[0])
		episodic_reward.append(total_reward)
#		print('curr-reward,n-state,n-action',vals[1]/10.0,vals[0],act)
	ql.Reset()
#	episodic_reward.append(total_reward)

print(episodic_reward[-1],n_episodes)

