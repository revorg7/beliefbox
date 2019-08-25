import gym
from wrap2 import derived as qlearn
from numpy import random

num_trials = 1
env = gym.make('Taxi-v2')
ql = qlearn(500,6,0.99)

remove the normalized reward of -1 (which is +9) aftere normalization
episodic_reward = []
n_episodes = 0
for i in range(num_trials):
	total_reward = 0
	env.reset()
	act = 0
	prev_act = 0
	prev_state = -1								multi-step
	ql.Act(-1.0,0)	#To initialize root policy
	cond = True
	while cond:
#	while n_episodes < 1:
#		vals = env.step(env.action_space.sample())
		vals = env.step(act)
		if vals[-2] == True:
#			cond=False
			n_episodes+=1
			print('steps-taken ',len(episodic_reward),'episode no.',n_episodes,' 100-step reward. ',sum(episodic_reward[-100:])+vals[1])
#why is steps-taken deterministic value of 200 steps ? Check...
			reset_state = env.reset()
			episodic_reward.clear()
			ql.Act((vals[1]+10.0)/30,reset_state)
#			print('episode no.',n_episodes,' 100-step reward. ',sum(episodic_reward[-101:-1]))
#		total_reward+=vals[1]/10.0
		norm_reward = (vals[1]+10.0)/30
#		act = ql.Act(norm_reward,vals[0])
		act = ql.getAction(vals[0])
		if (random.uniform() < 0.2):
			act = random.randint(6) rand required
		ql.Observe(prev_state,prev_act,norm_reward,vals[0],act)
		prev_state = vals[0]
		prev_act = act

#		episodic_reward.append(total_reward)
		episodic_reward.append(vals[1])
#		if sum(episodic_reward[-101:-1]) > 0.97:
#			break
#		print('curr-reward,n-state,n-action',vals[1],vals[0],act)
	ql.Reset()
#	episodic_reward.append(total_reward)

#print(episodic_reward[-1],n_episodes)
print(n_episodes)
def decode(i):
        out = []
        out.append(i % 4)
        i = i // 4
        out.append(i % 5)
        i = i // 5
        out.append(i % 5)
        i = i // 5
        out.append(i)
        assert 0 <= i < 5
        return reversed(out)
#print(list(decode(489)))
