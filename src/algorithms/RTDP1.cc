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

#include "RTDP1.h"

RTDP1::RTDP1(DiscreteMDP* mdp, real gamma, int init_state_, real baseline)
{
    assert (mdp);
    assert (gamma>=0 && gamma <=1);
    this->mdp = mdp;
    this->gamma = gamma;
    this->baseline = baseline;
	init_state = init_state_;
    n_actions = mdp->getNActions();
    n_states = mdp->getNStates();
    Reset();
}

RTDP1::~RTDP1()
{
	delete mdp;
}

void RTDP1::Reset()
{
    V.Resize(n_states);
    pV.Resize(n_states);

    Q.Resize(n_states, n_actions);
    pQ.Resize(n_states, n_actions);

    
    for (int s=0; s<n_states; s++) {
        pV(s) = 0.0;
		real max = mdp->getExpectedReward(s, 0);
        for (int a=0; a<n_actions; a++) {
			real val = mdp->getExpectedReward(s, a);
            Q(s, a) = val;
            pQ(s, a) = 0.0; ///< Not used in this algorithm
			if (val > max) max = val;
        }
        V(s) = max;
    }

}

int RTDP1::Act(int state)
{
			init_state = state;
			mdp->Reset(init_state);
			int s = init_state;
			std::set<int> all_next_states;
			//Selecting best-action
            for (int a=0; a<n_actions; a++) {
                real V_next_sa = 0.0;
                const DiscreteStateSet& next = mdp->getNextStates(s, a); ///< Changing it to mdp->generateState(s,a) is not going to help, I've already tried
                for (DiscreteStateSet::iterator i=next.begin();
                     i!=next.end();
                     ++i) {
                    int s2 = *i;
					all_next_states.insert(s2);
                    real P = mdp->getTransitionProbability(s, a, s2);
                    V_next_sa += P * V(s2);
                }
                Q(s, a) = mdp->getExpectedReward(s, a) - baseline + gamma * V_next_sa;
            }
			Vector max_vec = Q.getRow(s);

			std::vector<int> actions = ArgMaxs(max_vec);
			int index = rand() % actions.size();

			V(s) = Max(max_vec);
			pV(s)+=1;
//			init_state = mdp->generateState(s, actions[index]);
			return actions[index];
}
