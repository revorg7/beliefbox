/// -*- Mode: c++ -*-
// copyright (c) 2012 by Christos Dimitrakakis <christos.dimitrakakis@gmail.com>
/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ***************************************************************************/

#ifndef DEMONSTRATIONS_H
#define DEMONSTRATIONS_H


#include "Trajectory.h"
#include "Environment.h"
#include "AbstractPolicy.h"
#include "Random.h"

template <class S, class A>
class Demonstrations
{
public:
    std::vector<Trajectory<S, A> > trajectories;
    Trajectory<S,A>* current_trajectory;
    std::vector<real> total_rewards;
    std::vector<real> discounted_rewards;
    Demonstrations() 
        : current_trajectory(NULL)
    {
        //fprintf(stderr, "Creating Demonstrators\n");
        NewEpisode();
    }
    void Observe(S s, A a)
    {
        //fprintf(stderr, "Size of trajectories: %d\n", trajectories.size())
        current_trajectory->Observe(s, a);
    }
    void Observe(S s, A a, real r)
    {
        //fprintf(stderr, "Size of trajectories: %d\n", trajectories.size())
        current_trajectory->Observe(s, a, r);
    }
    void NewEpisode()
    {
        //fprintf(stderr, "Adding Episode in Trajectories\n");
        trajectories.push_back(Trajectory<S, A>());
        current_trajectory = &trajectories[trajectories.size() - 1];
    }
	/// Use a negative horizon to use a geometric stopping distribution
	void Simulate(Environment<S, A>& environment, AbstractPolicy<S, A>& policy, real gamma, int horizon)
	{
		environment.Reset();
		policy.Reset();
		NewEpisode();
		bool running = true;
		real discount = 1.0;
		real total_reward = 0.0;
		real discounted_reward = 0.0;
		int t = 0;
		do {
			S state = environment.getState();
			real reward = environment.getReward();
			total_reward += reward;
			discounted_reward += discount * reward;
			if (horizon >= 0) {
				discount *= gamma;
			}
			policy.Observe(reward, state);
            A action = policy.SelectAction();
			Observe(state, action, reward);
			running = environment.Act(action);
			if (horizon >= 0 && t >= horizon) {
				running = false;
			} else if (horizon < 0) {
				if (urandom() < 1.0 - gamma) {
					running = false;
				}
			}
		} while (running);
        total_rewards.push_back(total_reward);
        discounted_rewards.push_back(discounted_reward);
	}

    int size() const
    {
        if (trajectories[trajectories.size() - 1].size() > 0) {
            return trajectories.size();
        }
        return trajectories.size()  - 1;
    }
};

#endif
