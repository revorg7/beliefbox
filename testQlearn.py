import gym
from wrap import QLearning as qlearn

num_episodes = 1000
env = gym.make('NChain-v0')
ql = qlearn(5,2,0.99,0,0.8)

episodic_reward = []
for i in range(num_episodes):
	total_reward = 0
	env.reset()
	act = 0
	for _ in range(1000):
#		env.step(env.action_space.sample())
		vals = env.step(act)
		total_reward+=vals[1]
		act = ql.Act(vals[1],vals[0])
	ql.Reset()
	episodic_reward.append(total_reward/1000.0)

#print(episodic_reward)


import numpy as np
def medfilt (x, k):
    """Apply a length-k median filter to a 1D array x.
    Boundaries are extended by repeating endpoints.
    """
    assert k % 2 == 1, "Median filter length must be odd."
    assert x.ndim == 1, "Input must be one-dimensional."
    k2 = (k - 1) // 2
    y = np.zeros ((len (x), k), dtype=x.dtype)
    y[:,k2] = x
    for i in range (k2):
        j = k2 - i
        y[j:,i] = x[:-j]
        y[:j,i] = x[0]
        y[:-j,-(i+1)] = x[j:]
        y[-j:,-(i+1)] = x[-1]
    return np.median (y, axis=1)


r = medfilt(np.array(episodic_reward),11)
from matplotlib import pyplot as plt
plt.plot(range(len(r)),r)
plt.show()
