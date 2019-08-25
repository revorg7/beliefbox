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

#include "MCThompsonSampling.h"
//#define TBRL_DEBUG

MCThompsonSampling::MCThompsonSampling(int n_states_,
                 int n_actions_,
                 real gamma_,
				 MDPModel* belief_,
				 LeafNodeValue leaf_node,
		 		 WhichAlgo algo,
				 bool useRTDP)
    : n_states(n_states_),
      n_actions(n_actions_),
      gamma(gamma_),
      belief(belief_),
      T(0),
	  leaf_node_expansion(leaf_node),
	  algorithm(algo),
	  Use_RTDP(useRTDP)
{
	const char* leaf_value_name[] = {"None", "V_Min", "V_Max", "V_mean", "V_U", "V_L"};
#ifdef TBRL_DEBUG2
    logmsg("Starting Tree-Bayes-RL with %d states, %d actions, %s bounds\n", n_states, n_actions, leaf_value_name[leaf_node]);
#endif
    current_state = -1;
    current_action = -1;
}



int MCThompsonSampling::Act(real reward, int next_state)
{
    assert(next_state >= 0 && next_state < n_states);

    T++;
    if (current_state >= 0 && current_action >= 0) {
        belief->AddTransition(current_state, current_action, reward, next_state);
    }

    current_state = next_state;

#ifdef TBRL_DEBUG2
	logmsg("Acting belief");
	belief->ShowModelStatistics();
#endif

	if (!root_policy) {Serror("Current Policy not calculated\n"); exit(-1);}
	int next_action = ArgMax( root_policy->getActionProbabilities(current_state) ) ;
	current_action = next_action;
    return current_action;
}


void MCThompsonSampling::CalculateExploit(int K_step, int reference_state)
{

	if (root_policy) delete root_policy;

	DiscreteMDP* mean_mdp = new DiscreteMDP(n_states, n_actions);
    for (int s=0; s<n_states; s++) {
		for (int a=0; a<n_actions; a++) {
			Vector marginal = belief->getTransitionProbabilities(s,a);
			for (int s_next=0; s_next<n_states; s_next++) {
				mean_mdp->setTransitionProbability(s, a, s_next, marginal[s_next]);
				real expected_reward = belief->getExpectedReward(s,a);
				mean_mdp->reward_distribution.setFixedReward(s, a, expected_reward);
			}
		}
	}


	if(Use_RTDP){
		RTDP solver = RTDP(mean_mdp, gamma, reference_state);
		int runs = 5;
		solver.ComputeStateValues(runs,K_step*3); /// << param for RTDP
		root_policy = solver.getPolicy();
	}
	else{
		ValueIteration solver= ValueIteration(mean_mdp, gamma);
		solver.ComputeStateValues(1e-3);
//		root_policy = new FixedDiscretePolicy(n_states,n_actions,solver.policy->p);
		root_policy = solver.getPolicy();
	} 

}


// Generate n-sample MDPs, calculate their optimal policy and run them for K_steps before creating new beliefs
void MCThompsonSampling::CalculateRootPolicy(int n_policies, int K_step, int reference_state)
{
	//Don't use RTDP since we are doing Vanilla TS
	if (n_policies==1) Use_RTDP = false;

	if (current_state < 0) {root_policy = new FixedDiscretePolicy(n_states,n_actions); return;}
#ifdef TBRL_DEBUG
	printf("t: %d\n",T);
	belief->ShowModel();
#endif

	if (algorithm == EXPLOIT) {CalculateExploit(K_step,reference_state);return;}

		std::vector<RTDP*> RT_objects;
		std::vector<ValueIteration*> PI_objects;
    for (int i=0; i<n_policies; ++i) {
	DiscreteMDP* model = belief->generate();
//for (int i=0; i<tree.n_states; i++) for (int j=0;j<tree.n_actions;j++) model->setFixedReward(i, j, tree.environment->getExpectedReward(i,j));
#ifdef TBRL_DEBUG
	printf("sample no.%d \n",i);
	model->ShowModel();
#endif
if(Use_RTDP)	RT_objects.push_back(new RTDP(model, gamma, reference_state));
else	PI_objects.push_back(new ValueIteration(model, gamma));
	//delete model;		//cant delete as PI_objects depend on it
//	models[i] = model;	//So have to collect here to delete them later 
    }

	int runs = 5;
//	if (tree.T < 2000) runs = 1;
//    #pragma omp parallel for num_threads(n_policies)
    for (int i=0; i<n_policies; ++i) {
//	printf("Threads: %d \n",omp_get_num_threads());
	//DiscreteMDP* model = belief->generate();
	//ValueIteration VI(model, tree.gamma);
	//VI.ComputeStateValuesStandard(1e-1);
	//FixedDiscretePolicy* policy = VI.getPolicy();

	//PolicyIteration PI(model, tree.gamma);
if(Use_RTDP)	RT_objects[i]->ComputeStateValues(runs,K_step*3); /// << param for RTDP
else	PI_objects[i]->ComputeStateValues(1e-4);
	//delete model;
    }

    // Creating n_policies policy
    std::vector<FixedDiscretePolicy*> policies;

    // Deleting pointers and adding policies
    for (int i=0; i<n_policies; ++i) {
if(Use_RTDP)	{ policies.push_back(RT_objects[i]->getPolicy()); delete RT_objects[i];}
else	{policies.push_back(PI_objects[i]->getPolicy()); delete PI_objects[i];}
#ifdef TBRL_DEBUG
	printf("Printing policy i: %d\n",i);
	policies[i]->Show();
#endif
	//delete models[i];	//even deleting here says "illegal instruction" error
    }


	if (root_policy) delete root_policy;

	// IF n_polices=1, then its vanilla ThompsonSampling
	if (n_policies==1) {root_policy = new FixedDiscretePolicy(n_states,n_actions,policies[0]->p); return;}	

	const DiscreteMDP* mean_mdp = belief->getMeanMDP();
	Vector values(n_policies);
    for (int i=0; i<n_policies; ++i) {
		PolicyEvaluation polyeval = PolicyEvaluation(policies[i],mean_mdp, gamma);
//		PolicyEvaluation polyeval = AveragePolicyEvaluation(policies[i],mean_mdp, gamma);
		polyeval.ComputeStateValues(1e-2);
		values[i] = polyeval.getValue(reference_state);
	//	printf(" current_state:%d, val of policy[i]:%f ",current_state,values[i]);
	}

    int best_policy = ArgMax(values);

	root_policy = new FixedDiscretePolicy(n_states,n_actions,policies[best_policy]->p);

}







//MCThompsonSampling::~MCThompsonSampling() {}

void MCThompsonSampling::Reset()
{
    current_state = -1;
    current_action = -1;
}

void MCThompsonSampling::Reset(int state)
{
    current_state = state;
    current_action = -1;
}

/// Full observation
real MCThompsonSampling::Observe (int state, int action, real reward, int next_state, int next_action)
{
    if (state>=0) {
        belief->AddTransition(state, action, reward, next_state);
    }
    current_state = next_state;
    current_action = next_action;
    return 0.0;
}
/// Partial observation 
real MCThompsonSampling::Observe (real reward, int next_state, int next_action)
{
    if (current_state >= 0 && current_action >= 0) {
        belief->AddTransition(current_state, current_action, reward, next_state);
    }
    current_state = next_state;
    current_action = next_action;
    return 0.0;
}
