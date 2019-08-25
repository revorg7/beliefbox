// -*- Mode: c++ -*-
// copyright (c) 2017 by Christos Dimitrakakis <christos.dimitrakakis@gmail.com>
/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ***************************************************************************/

#ifndef MC_TS_H
#define MC_TS_H

#include "real.h"
#include "Matrix.h"
#include "OnlineAlgorithm.h"
#include "MDPModel.h"
#include "PolicyIteration.h"
#include "ValueIteration.h"
#include "RTDP.h"
#include "DiscretePolicy.h"
#include "PolicyEvaluation.h"
#include "AveragePolicyEvaluation.h"


/**
Implements 3 algorithms: MCTS, TS (for n_polices=1), EXPLOIT
**/
class MCThompsonSampling : public OnlineAlgorithm<int, int>
{
public:
	enum LeafNodeValue {
		NONE = 0x0, V_MIN, V_MAX, V_MEAN, V_UTS, V_LTS
	};
	enum WhichAlgo {
		MCTS = 0x0, EXPLOIT
	};
    MCThompsonSampling(int n_states_, ///< number of states
            int n_actions_, ///< number of actions
            real gamma_, ///< discount factor
            MDPModel* belief_, ///< belief about the MDP
			LeafNodeValue leaf_node_expansion = NONE,WhichAlgo algo = MCTS,bool useRTDP = false);
protected:
	FixedDiscretePolicy* root_policy;
    int T; ///< time passed
    MDPModel* belief; ///< pointer to the base MDP model
    real gamma; ///< discount factor
    int current_action; ///< current action
    LeafNodeValue leaf_node_expansion; ///< how to expand the leaf node
    WhichAlgo algorithm; //< which algorithm to use
	bool Use_RTDP;
public:
    int current_state; ///< current state
    const int n_states;
    const int n_actions; 
    virtual void Reset();
    virtual void Reset(int state);
    virtual int Act(real reward, int next_state);
    virtual real getValue (int state, int action)
    {
		Serror("Not implemented in this context\n");
        return 0;
    }
    /// Full observation
    virtual real Observe (int state, int action, real reward, int next_state, int next_action);
    /// Partial observation 
    virtual real Observe (real reward, int next_state, int next_action);
    virtual void setFixedRewards(const Matrix& rewards)
    {
        belief->setFixedRewards(rewards);
#if 0
        logmsg("Setting reward matrix\n");
        rewards.print(stdout);
        belief->ShowModel();
#endif
    }
	void CalculateRootPolicy(int n_policies, int K_step,int reference_state);
	void CalculateExploit(int K_step, int reference_state);

};
#endif
