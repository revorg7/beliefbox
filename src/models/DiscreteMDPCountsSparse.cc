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

#include "DiscreteMDPCountsSparse.h"
#include "Random.h"
#include "SingularDistribution.h"

#include <stdexcept>

//#define TBRL_DEBUG
//#define SPARSE_DEBUG

/** Create a counting model of an MDP.
	
	\arg n_states the number of MDP states
	\arg n_actions the number of MDP actions
	\arg init_transition_count the prior for the Dirichlet. The higher this is, the more the model will expect to see unseen transitions.
	\arg init_reward_count the prior number of counts for the reward.  This should be used in conjuction with the prior reward average to bias the rewards.
	\arg The prior reward average. This can be used to bias the average to some particular value.
*/
DiscreteMDPCountsSparse::DiscreteMDPCountsSparse (int n_states, int n_actions, real init_transition_count, RewardFamily reward_family_)
    : 
    MDPModel(n_states, n_actions),
	transitions(n_states, n_actions, init_transition_count),
    reward_family(reward_family_)
{
    logmsg("Creating DiscreteMDPCountsSparse with %d states and %d actions\n",  n_states, n_actions);
    N = n_states * n_actions;
    ER.resize(N);
    for (int i=0; i<N; ++i) {
        switch(reward_family) {
        case BETA:
            ER[i] = new BetaDistribution();
            break;
        case NORMAL:
            ER[i] = new NormalUnknownMeanPrecision();
            break;
        case FIXED:
            ER[i] = new UnknownSingularDistribution();
            break;
        default:
            Serror("Unknown distribution family %d\n", reward_family);
        }
    }
}


DiscreteMDPCountsSparse::DiscreteMDPCountsSparse(const DiscreteMDPCountsSparse& model) :
	MDPModel(model.n_states, model.n_actions),
//	use_sampling(model.use_sampling),
	transitions(model.transitions),
	reward_family(model.reward_family),
	N(model.N)
{
	//logmsg("Copying DiscreteMDPCountsSparse with %d states and %d actions\n",  n_states, n_actions);
	
	ER.resize(N);
	for (int i=0; i<N; ++i) {
		switch(reward_family) {
		case BETA:
			ER[i] = model.ER[i]->Clone();
			break;
		case NORMAL:
			ER[i] = model.ER[i]->Clone();
			break;
		case FIXED:
			ER[i] = model.ER[i]->Clone();
			break;
		default:
			Serror("Unknown distribution family %d\n", reward_family);
		}
	}
}

/// CHECK: Some parameters are not copied
DiscreteMDPCountsSparse* DiscreteMDPCountsSparse::Clone () const
{
	DiscreteMDPCountsSparse* clone = new DiscreteMDPCountsSparse(*this);
	return clone;
}

real DiscreteMDPCountsSparse::CalculateDistance(MDPModel* target_belief,int s, int a) const
{			
	Vector v1 = getTransitionProbabilities (s, a);
	Vector v2 = target_belief->getTransitionProbabilities (s, a);

	real min_ratio = 1.0;
	real max_ratio = 0.0;
    	for (int s_n=0; s_n<n_states; s_n++) {
			if ( v2[s_n]/v1[s_n] > max_ratio) max_ratio = v2[s_n]/v1[s_n];
			if ( v2[s_n]/v1[s_n] < min_ratio) min_ratio = v2[s_n]/v1[s_n];
		}

	return log(max_ratio) - log(min_ratio);
}


DiscreteMDPCountsSparse::~DiscreteMDPCountsSparse()
{
    for (int i=0; i<N; ++i) {
        delete ER[i];
    }
    //printf ("COUNTS MODEL\n");
    //ShowModel();
}

#if 0
/// Copy the mean MDP
DiscreteMDP* DiscreteMDPCounts::CreateMDP() const
{
    mdp_dbg("Making a DiscreteMDP with %d states, %d actions from model\n", n_states, n_actions);
	DiscreteMDP* mdp = new DiscreteMDP(n_states, n_actions);
	CopyMeanMDP(mdp);
    return mdp;
}
#endif

void DiscreteMDPCountsSparse::setFixedRewards(const Matrix& rewards)
{
	//logmsg("Setting fixed rewards\n");
    for (int s=0; s<n_states; ++s) {
        for (int a=0; a<n_actions; ++a)  {
            int ID = getID(s, a);
            delete ER[ID];
            ER[ID] = new UnknownSingularDistribution();
            ER[ID]->Observe(rewards(s,a));
			//printf("R: %d %d %f -> %f\n",
			//	   s, a, rewards(s,a), ER[ID]->getMean());
        }
    }
}

void DiscreteMDPCountsSparse::AddTransition(int s, int a, real r, int s2)
{
    int ID = getID (s, a);
    //printf ("(%d, %d) [%.2f] -> %d\n", s, a, r, s2);
    transitions.Observe(s, a, s2);
    ER[ID]->Observe(r);

    real expected_reward = getExpectedReward(s,a);
    
}

//void DiscreteMDPCounts::SetNextReward(int s, int a, real r)
//{
//    ER[getID (s, a)].mean = r;
//}

/// Generate a reward from the marginal distribution 
real DiscreteMDPCountsSparse::GenerateReward (int s, int a) const
{
    return ER[getID (s, a)]->generateMarginal();
}

/// Generate a transition from the marginal distribution
int DiscreteMDPCountsSparse::GenerateTransition (int s, int a) const
{
	return transitions.marginal_generate(s, a);
}

/// Get the specific transition probability
real DiscreteMDPCountsSparse::getTransitionProbability (int s, int a, int s2) const
{
    return transitions.marginal_pdf(s, a, s2);
}

/// Get the specific reward probability
real DiscreteMDPCountsSparse::getRewardProbability (int s, int a, real r) const
{
    return ER[getID (s, a)]->marginal_pdf(r);
}

/// Get a vector of transition probabilities
Vector DiscreteMDPCountsSparse::getTransitionProbabilities (int s, int a) const
{
    return transitions.getMarginal(s, a);
}

/// get the expected reward
real DiscreteMDPCountsSparse::getExpectedReward (int s, int a) const
{
    return ER[getID (s,a)]->getMean();
}

/// Reset at the end of an episode 
void DiscreteMDPCountsSparse::Reset()
{
}

/// Show the model.
void DiscreteMDPCountsSparse::ShowModel() const
{
	printf ("# mean model\n");
    for (int a=0; a<n_actions; a++) {
        for (int i=0; i<n_states; i++) {
            std::cout << "P: " << a << "," << i << ":";
            for (int j=0; j<n_states; j++) {
                real p = getTransitionProbability(i, a, j);
                //if (p<0.01) p =0.0f;
                std::cout << p << " ";
            }
            std::cout << " ["
                      << transitions.getParameters(i, a).Sum()
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

/// Show the model.
void DiscreteMDPCountsSparse::ShowModelStatistics() const
{
	printf ("# model statistics\n");
    for (int a=0; a<n_actions; a++) {
        for (int i=0; i<n_states; i++) {
            std::cout << "P: " << a << "," << i << ":";
			transitions.getParameters(i, a).print(stdout);
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


DiscreteMDP* DiscreteMDPCountsSparse::generate() const
{
    DiscreteMDP* mdp = new DiscreteMDP(n_states, n_actions, NULL);
    for (int s=0; s<n_states; s++) {
        for (int a=0; a<n_actions; a++) {
            //Vector C =  P[getID (s,a)].getMarginal();
            Vector C =  transitions.generate(s,a);
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
const DiscreteMDP * DiscreteMDPCountsSparse::getMeanMDP() const
{
	//DiscreteMDP* mdp = new DiscreteMDP(n_states, n_actions);
	//CopyMeanMDP(mdp);
    //    return mdp;
    return nullptr;
}



