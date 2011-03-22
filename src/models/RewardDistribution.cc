// -*- Mode: c++ -*-
// copyright (c) 2011 by Christos Dimitrakakis <christos.dimitrakakis@gmail.com>
/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ***************************************************************************/

#include "RewardDistribution.h"

// -- Goal state reward distribution -- //

/// Constructor. Set up the default reward to be received
GoalStateRewardDistribution::    GoalStateRewardDistribution(real default_reward_, 
															 int goal_state_,
															 real goal_reward_)
  : default_reward(default_reward_),
    goal_state(goal_state_),
    goal_reward(goal_reward_)
{
    // empty
}

/// Destructor - nothing to do
GoalStateRewardDistribution::~GoalStateRewardDistribution() 
{
}

/// Generate a value - always the expected value
real GoalStateRewardDistribution::generate(int state, int action) const
{
	return expected(state, action);
}

/// Get expected value
real GoalStateRewardDistribution::expected(int state, int action) const
{
    if (state == goal_state) {
        return goal_reward;
    }
    return default_reward;
}

/// Generate pdf -- in fact, a probability since there are just two outcomes.
real GoalStateRewardDistribution::pdf(int state, int action, real reward) const
{
    real r = expected(state, action);
    if (fabs(r - reward) < 1e-6) {
        return 1.0;
    } else {
        return 0.0;
    }
    
}

// -- General reward distribution for discrete environments -- //

/// Constructor. Set up the default reward to be received
DiscreteSpaceRewardDistribution::    DiscreteSpaceRewardDistribution(int n_states_, int n_actions_)
    : n_states(n_states_),
      n_actions(n_actions_)
{
    // empty
}

/// Destructor - nothing to do
DiscreteSpaceRewardDistribution::~DiscreteSpaceRewardDistribution() 
{
}

/// Generate a value - always the expected value
real DiscreteSpaceRewardDistribution::generate(int state, int action) const
{
	return expected(state, action);
}

/// Get expected value
real DiscreteSpaceRewardDistribution::expected(int state, int action) const
{
    if (state == goal_state) {
        return goal_reward;
    }
    return default_reward;
}

/// Generate pdf -- in fact, a probability since there are just two outcomes.
real DiscreteSpaceRewardDistribution::pdf(int state, int action, real reward) const
{
    real r = expected(state, action);
    if (fabs(r - reward) < 1e-6) {
        return 1.0;
    } else {
        return 0.0;
    }
    
}
