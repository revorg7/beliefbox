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

#ifndef TREE_BRL_POLICY_PYTHON_H
#define TREE_BRL_POLICY_PYTHON_H

//Extra includes for pybind11 create method
#include "DiscreteMDPCountsSparse.h"
//

#include "DiscreteMDP.h"
#include "DiscretePolicy.h"
#include "ExplorationPolicy.h"
#include "Matrix.h"
#include "real.h"
#include "OnlineAlgorithm.h"
#include "MDPModel.h"
#include "MultiMDPValueIteration.h"
#include "ValueIteration.h"
#include "PolicyIteration.h"
#include "RTDP.h"
#include "PolicyEvaluation.h"
#include "QLearning.h"
#include "ExplorationPolicy.h"
#include "RandomPolicy.h"
#include <omp.h>
#include <vector>
#include <memory>
#include "PolicyEvaluation.h"		//for warm-start PI

/// \ingroup ReinforcementLearning
/// @{
    
/** Direct model-based reinforcement learning using trees.
  
 */
class TreeBRLPolicyPython : public OnlineAlgorithm<int, int>
{
public:
	enum LeafNodeValue {
		NONE = 0x0, V_MIN, V_MAX, V_MEAN, V_UTS, V_LTS, Q_LEARNING
	};
	enum WhichAlgo {
		PLCAVG = 0x0, PLC, WRNGPLCAVG, FULL, SPARSE, RANDOM
	};
protected:
	std::shared_ptr<DiscreteEnvironment> environment;
    const int n_states; ///< number of states
    const int n_actions; ///< number of actions
    real gamma; ///< discount factor
    real epsilon; ///< randomness
    int current_state; ///< current state
    int current_action; ///< current action
    int horizon; ///< maximum number of samples to take
    int T; ///< time passed
    int size; ///< size of tree
    Vector Qs; ///< caching the value of the actions for the current state
    LeafNodeValue leaf_node_expansion; ///< how to expand the leaf node
    WhichAlgo algorithm; //< which algorithm to use
    std::vector<FixedDiscretePolicy*> root_policies;///< polcies sampled at the root
    const int n_policies; ///< number of policies sampled in algorithm 
    const int n_samples;
public:
	QLearning* qlearning;
	bool Use_RTDP = false;
    MDPModel* belief; ///< pointer to the base MDP model
    const int K_step;
    FixedDiscretePolicy* root_policy;	//used for taking multiple-steps in real environment in PSRL style
	int getAction(int state) {return ArgMax( root_policy->getActionProbabilities(state) ) ;}
    class BeliefState
    {
    protected:
        TreeBRLPolicyPython& tree; ///< link to the base tree
		MDPModel* belief; ///< current belief
        int state; ///< current state
        int prev_action; ///< action taken to arrive here
        real prev_reward; ///< reward received to arrive here
        real probability; ///< probability of arriving here given previous state and action
		real discount_factor; ///< discount_factor due to non-constant no. of steps from parent to child
        std::vector<BeliefState*> children; ///< next belief states
        BeliefState* prev; ///< previous belief state
        int t; ///< time
    public:
        BeliefState(TreeBRLPolicyPython& tree_,
                    const MDPModel* belief_,
                    int state_);
        BeliefState(TreeBRLPolicyPython& tree_,
                    const MDPModel* belief_,
                    int prev_state_,
                    int prev_action_,
                    int state_,
                    real r,
                    real p,
                    BeliefState* prev_);
	BeliefState(TreeBRLPolicyPython& tree_,
                    MDPModel* belief_,
                    int prev_action_,
                    int state_,
                    real prev_total_r,
                    real p,
                    BeliefState* prev_,
					real discount_factor_);

	~BeliefState();

        // methods for building the tree
        void ExpandAllActions();
        void SparseExpandAllActions(int n_samples);
	void SparserExpandAllActions(int n_samples,int n_policies, int K_step);
	void SparserAverageExpandAllActions(int n_samples,int n_policies, int K_step);
        // methods for calculating action values in the tree
        real CalculateValues(LeafNodeValue leaf_node, int buffer);
		real MeanMDPValue();
        real UTSValue();
        real LTSValue();
		void print() const;
		real Qlearning();
        // methods for adaptively building the tree while calculating values (TODO)
        // real StochasticBranchAndBound(int n_samples);
    };

	//Custom constructor according to Pybind11 docs
	TreeBRLPolicyPython(int n_states,int n_actions, real discounting);


    TreeBRLPolicyPython(std::shared_ptr<DiscreteEnvironment> environment_,
			int n_states_, ///< number of states
            int n_actions_, ///< number of actions
            real gamma_, ///< discount factor
            MDPModel* belief_, ///< belief about the MDP
            int horizon_ = 1,
			LeafNodeValue leaf_node_expansion = NONE,
			WhichAlgo algorithm = PLC,
		        int n_policies_ = 2,
			int n_samples_ = 2,
			int K_step_ = 40,
			bool useRTDP = false);
    virtual ~TreeBRLPolicyPython();
    virtual void Reset();
    virtual void Reset(int state);
    /// Full observation
    virtual real Observe (int state, int action, real reward, int next_state, int next_action);
    /// Partial observation 
    virtual real Observe (real reward, int next_state, int next_action);
    /// Get an action using the current exploration policy.
    /// it calls Observe as a side-effect.
    virtual int Act(real reward, int next_state);
    /** Set the rewards to Singular distributions.

        Since this is a Bayesian approach, we can simply set the belief about the reward in each state to be a singular distribution, if we want. This would correspond to us having a fixed, unshakeable belief about them.
    */
    virtual void setFixedRewards(const Matrix& rewards)
    {
        belief->setFixedRewards(rewards);
#if 0
        logmsg("Setting reward matrix\n");
        rewards.print(stdout);
        belief->ShowModel();
#endif
    }

    virtual real getValue(int state, int action)
    {
		Serror("Not implemented in this context\n");
        return 0;
    }

    TreeBRLPolicyPython::BeliefState CalculateSparseBeliefTree(int n_samples, int n_TS);
    TreeBRLPolicyPython::BeliefState CalculateBeliefTree();
    TreeBRLPolicyPython::BeliefState CalculateSparserBeliefTree(int n_samples, int K_step, int n_TS);
    
};


/// @}
#endif
