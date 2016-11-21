// -*- Mode: c++ -*-
// copyright (c) 2016 by Christos Dimitrakakis <christos.dimitrakakis@gmail.com>
/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ***************************************************************************/

#ifndef DISCRETE_MMDP_H
#define DISCRETE_MMDP_H

// This is just a simple template for Multi-Agent MDPs
class DiscreteMMDP
{
protected:
	int n_players;
	int n_actions;
	int n_states;
    int state;
	DiscreteJointAction action;
	DiscreteTransitionDistribution transition_distribution;
	int current_player; ///< We need to know who is playing
public:
    DiscreteMMDP(int n_players_,
				 int n_states_,
				 int n_actions_)
		: n_players(n_players_),
		  n_actions(n_actions_),
		  n_states(n_states_),
		  state(0),
		  action(n_players),
		  transition_distribution(n_players, n_actions, ..... FIXME
	{
	}
  virtual ~MDP() {}
	int getState() const
	{
		return state;
	}
    virtual real getTransitionProbability (const StateType& s,
                                           const ActionType& a,
                                           const StateType& s2) const
    {
        return transition_distribution.pdf(s, a, s2);
    }
    virtual real getExpectedReward (const StateType& s,
                                    const ActionType& a) const
    {
        return reward_distribution.expected(s, a);
    }
	
    virtual StateType generateState(const StateType& s,
                            const ActionType& a) const
    {
        return transition_distribution.generate(s, a);
    }
    virtual real generateReward(const StateType& s,
                                const ActionType& a) const
    {
        return reward_distribution.generate(s, a);
    }
    // generate a new state given the current state and action, then set the current state to be the new state.
    virtual real Act (ActionType& a)
    {
        real r = generateReward(state, a);
        state = generateState(state, a);
        return r;
    }
    virtual StateType generateState(const ActionType& a)
    {
        return generateState(state, a);
    }
};

#endif
	
