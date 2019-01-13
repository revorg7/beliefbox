// -*- Mode: c++ -*-
// copyright (c) 2006 by Christos Dimitrakakis <christos.dimitrakakis@gmail.com>
// $Revision$
/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 *   Created by - Divya Grover
 ***************************************************************************/

#ifndef RTDP1_H
#define RTDP1_H

#include "DiscreteMDP.h"
#include "DiscretePolicy.h"
#include "Matrix.h"
#include "Vector.h"
#include "real.h"
#include <vector>

// Just simplfied version of previous RTDP algorithm, for maybe faster processing (not verified yet)
//Class and actions are assumed int, athough templating can be done later
class RTDP1
{
protected:
public:
    DiscreteMDP* mdp; ///< pointer to the MDP
    real gamma; ///< discount factor
    int n_states; ///< number of states
    int n_actions; ///< number of actions
	int init_state; ///< intial state
	Vector V; ///< state values
    Vector pV; ///< Used here to keep state-visit counts
    Matrix Q; ///< state-action value
    Matrix pQ; ///< previous state-action values
    real Delta;
	real baseline; ///< Something needed for VI, I dont know if I need it
    RTDP1(DiscreteMDP* mdp, real gamma, int init_state, real baseline=0.0);
    ~RTDP1();
    void Reset();
	int Act(int state);

    /// Set the MDP to something else
    inline void setMDP(DiscreteMDP* mdp_)
    {
        mdp = mdp_;
    }

	inline FixedDiscretePolicy* getPolicy()
	{
//change it before trying lrtdp
    	return new FixedDiscretePolicy(n_states, n_actions, Q);
	}



};
#endif

