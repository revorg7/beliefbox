// -*- Mode: c++ -*-
// copyright (c) 2005-2007 by Christos Dimitrakakis <christos.dimitrakakis@gmail.com>
// $Revision$
/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ***************************************************************************/

#include "DiscreteMDPCounts.h"
#include "Random.h"
#include <stdexcept>

/** Create a counting model of an MDP.
	
   \arg n_states the number of MDP states
   \arg n_actions the number of MDP actions
   \arg init_transition_count the prior for the Dirichlet. The higher this is, the more the model will expect to see unseen transitions.
   \arg init_reward_count the prior number of counts for the reward.  This should be used in conjuction with the prior reward average to bias the rewards.
   \arg The prior reward average. This can be used to bias the average to some particular value.
 */
DiscreteMDPCounts::DiscreteMDPCounts (int n_states, int n_actions, real init_transition_count, int init_reward_count, real init_reward) :
    MDPModel(n_states, n_actions),
    mean_mdp(n_states, n_actions, NULL)
{
    mdp_dbg("Creating DiscreteMDPCounts with %d states and %d actions\n",  n_states, n_actions);
    N = n_states * n_actions;
    P.resize(N);
    ER.resize(N);
    for (int i=0; i<N; ++i) {
        P[i].resize(n_states, init_transition_count);
		//        ER[i].Reset(init_reward, init_reward_count);
		ER[i].beta = 1.0;
		ER[i].alpha = 1.0;
    }
}

DiscreteMDPCounts::~DiscreteMDPCounts()
{
    //printf ("COUNTS MODEL\n");
    //ShowModel();
}


DiscreteMDP* DiscreteMDPCounts::CreateMDP()
{
    mdp_dbg("Making a DiscreteMDP with %d states, %d actions from model\n", n_states, n_actions);
#if 0
	DiscreteMDP* tmp_mdp = *getMeanMDP();
    DiscreteMDP* mdp = new DiscreteMDP(*tmp_mpd);
	delete tmp_mdp;
#else
	DiscreteMDP* mdp = new DiscreteMDP(n_states, n_actions);
	CopyMeanMDP(mdp);
#endif

#if 0
    for (int i=0; i<n_states; ++i) {
        for (int a=0; a<n_actions; ++a) {
            real sum_p = 0.0;
            for (int j=0; j<n_states; ++j) {
                real p = mdp->getTransitionProbability(i, a, j);
                sum_p += p;
            }
            if (fabs(sum_p - 1.0) > 0.001) {
                printf ("sum_s' p(s'|s=%d, a=%d) = %f\n", i, a, sum_p);
                assert(0);
                exit(-1);
            }
        }
    }
    mdp->Check();
#endif
    return mdp;
}


void DiscreteMDPCounts::AddTransition(int s, int a, real r, int s2)
{
    int ID = getID (s, a);
    //printf ("(%d, %d) [%.2f] -> %d\n", s, a, r, s2);
    P[ID].Observe(s2);
    ER[ID].Observe(r);

    Vector C =  P[ID].GetMean();
    real expected_reward = getExpectedReward(s,a);
    mean_mdp.reward_distribution.setFixedReward(s, a, expected_reward);
    for (int s_next=0; s_next<n_states; s_next++) {
        mean_mdp.setTransitionProbability(s, a, s_next, C[s_next]);
    }
    
}

//void DiscreteMDPCounts::SetNextReward(int s, int a, real r)
//{
//    ER[getID (s, a)].mean = r;
//}

/// Generate a reward from the marginal distribution 
real DiscreteMDPCounts::GenerateReward (int s, int a) const
{
    return ER[getID (s, a)].generate();
}

/// Generate a transition from the marginal distribution
int DiscreteMDPCounts::GenerateTransition (int s, int a) const
{
    Vector p = P[getID (s,a)].GetMean();
    real d=urandom();
    real sum = 0.0;
    int n_outcomes = p.Size();
    for (int i=0; i<n_outcomes; i++) {
        sum += p[i];
        if (d < sum) {
            return i;
        }
    }
    return rand()%n_outcomes;
}

real DiscreteMDPCounts::getTransitionProbability (int s, int a, int s2) const
{
    Vector p = P[getID (s,a)].GetMean();
    return p[s2];
}

Vector DiscreteMDPCounts::getTransitionProbabilities (int s, int a) const
{
    return P[getID (s,a)].GetMean();
}

real DiscreteMDPCounts::getExpectedReward (int s, int a) const
{
    return ER[getID (s,a)].getMean();
}

void DiscreteMDPCounts::Reset()
{
}


void DiscreteMDPCounts::ShowModel() const
{
	printf ("# mean model\n");
    for (int a=0; a<n_actions; a++) {
        for (int i=0; i<n_states; i++) {
            std::cout << a << "," << i << ":";
            for (int j=0; j<n_states; j++) {
                real p = getTransitionProbability(i, a, j);
                if (p<0.01) p =0.0f;
                std::cout << p << " ";
            }
            std::cout << " ["
                      << P[getID(i,a)].GetParameters().Sum()
                      << "]\n";
        }
    }

   for (int a=0; a<n_actions; a++) {
        for (int i=0; i<n_states; i++) {
            std::cout << "R(" << a << "," << i 
                      << ") = " << getExpectedReward(i, a)
				//<< " [" << ER[getID(i,a)].n_samples << "]"
                      << std::endl; 
        }
   }
}

DiscreteMDP* DiscreteMDPCounts::generate() 
{
    DiscreteMDP* mdp = new DiscreteMDP(n_states, n_actions, NULL);
    for (int s=0; s<n_states; s++) {
        for (int a=0; a<n_actions; a++) {
            //Vector C =  P[getID (s,a)].GetMean();
            Vector C =  P[getID (s,a)].generate();
            real expected_reward = GenerateReward(s,a);
            mdp->reward_distribution.addFixedReward(s, a, expected_reward);
            for (int s2=0; s2<n_states; s2++) {
                if (C[s2]) {
                    mdp->setTransitionProbability(s, a, s2, C[s2]);
                }
            }
        }
    }
    
    return mdp;
}


/// Get a pointer to the mean MDP
const DiscreteMDP * const DiscreteMDPCounts::getMeanMDP() const
{
	DiscreteMDP* mdp = new DiscreteMDP(n_states, n_actions);
	CopyMeanMDP(mdp);
	return mdp;
	//return &mean_mdp;
}

void DiscreteMDPCounts::CopyMeanMDP(DiscreteMDP* mdp) const
{
    if (mdp->GetNStates() != n_states) {
        throw std::runtime_error("incorrect number of states");
    }

    if (mdp->GetNActions() != n_actions) {
        throw std::runtime_error("incorrect number of actions");
    }

    for (int s=0; s<n_states; s++) {
        for (int a=0; a<n_actions; a++) {
            Vector C =  P[getID (s,a)].GetMean();
            real expected_reward = getExpectedReward(s,a);
            mdp->reward_distribution.addFixedReward(s, a, expected_reward);
            for (int s2=0; s2<n_states; s2++) {
                mdp->setTransitionProbability(s, a, s2, C[s2]);
            }
        }
    }
    
}


