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

#include "TreeBRLPolicyPython.h"
#include <chrono>
#include "fstream"

//Although the previous way of averaging over first actions was better, it is incorrect, since it essentially added the values functions of each policy with same action
// Consider, 3 policy, 2 samples, with 2 policy having same 1st action, now weighting by 1/n_samples for each policy, we are essntially calculating V(pi_1),V(pi_2) and V(pi_3)
// and after that adding V(pi_1)+V(pi_2) = V(action) and then comparing it with the remaing value function, which is V(pi_3). hence it becomes V(pi_1)+V(pi_2) vs V(pi_3).


//#define TBRL_DEBUG9

TreeBRLPolicyPython::TreeBRLPolicyPython(std::shared_ptr<DiscreteEnvironment> environment_,
				int n_states_,
                 int n_actions_,
                 real gamma_,
				 MDPModel* belief_,
                 int horizon_,
		 enum LeafNodeValue leaf_node,
		 enum WhichAlgo algo,
		 int n_policies_,
		 int n_samples_,
		 int K_step_,
		 bool useRTDP)
    : n_states(n_states_),
      n_actions(n_actions_),
      gamma(gamma_),
      belief(belief_),
      horizon(horizon_),
      T(0),
      size(0),
	  leaf_node_expansion(leaf_node),
	 algorithm(algo),
	n_policies(n_policies_),
	n_samples(n_samples_),
	K_step(K_step_),
	Use_RTDP(useRTDP)
{
qlearning = new QLearning(n_states,n_actions,gamma,0.3,0.2,new EpsilonGreedy(n_actions,0.3));
environment = environment_;

	const char* leaf_value_name[] = {"None", "V_Min", "V_Max", "V_mean", "V_U", "V_L"};
#ifdef TBRL_DEBUG2
    logmsg("Starting Tree-Bayes-RL with %d states, %d actions, %d horizon, %s bounds\n", n_states, n_actions, horizon, leaf_value_name[leaf_node]);
#endif
    current_state = -1;

    switch(algorithm) {
	case PLC:  Qs(n_policies); break;
	default: Qs(n_actions); break;
	}
}

TreeBRLPolicyPython::TreeBRLPolicyPython(int n_states_,int n_actions_, real discounting,int n_policies_, int n_samples_, int K_step_):
				n_states(n_states_),
				n_actions(n_actions_),
				gamma(discounting),
				T(0),
				size(0),
				n_policies(n_policies_),
				n_samples(n_samples_),
				K_step(K_step_)
 {

	    // ---- user options ---- //
	    horizon = 1; // << Search Tree depth
		Use_RTDP = false; // << Using RTDP for candidate policy generation
	    leaf_node_expansion = LeafNodeValue::NONE;
	    algorithm = WhichAlgo::PLC;
	    current_state = -1;
    	std::shared_ptr<DiscreteEnvironment> environment=NULL;
		root_policy = NULL;
		qlearning = new QLearning(n_states,n_actions,gamma,0.3,0.2,new EpsilonGreedy(n_actions,0.3));

	    switch(algorithm) {
		case PLC:  Qs(n_policies); break;
		default: Qs(n_actions); break;
		}

		//Making belief inside
		real dirichlet_mass = 2.0;	//INTIAL VALUE IN GUEZ CODE
		enum DiscreteMDPCountsSparse::RewardFamily reward_prior = DiscreteMDPCountsSparse::NORMAL; //Doing Fixed here

	    belief = new DiscreteMDPCountsSparse(n_states, n_actions, dirichlet_mass, reward_prior);

		bool useFixedRewards = false; // << Using Fixed Rewards
//		if(useFixedRewards) belief.setFixedRewards(rewards);

}

int TreeBRLPolicyPython::getAction(int state)
{
	int action = 0;
	if (root_policy != NULL)
	action = ArgMax( root_policy->getActionProbabilities(state) ) ;
	return action;
}


// Note that the belief tree is only created within Act() and
// destroyed immediately. Hence there is no need to remove
// anything here. But on the other hand, this is inefficient.
TreeBRLPolicyPython::~TreeBRLPolicyPython()
{
delete qlearning;
    //printf(" # destroying tree of size %d\n", size);
#ifdef TBRL_DEBUG8
	printf("no. of root policies is:%d \n",root_policies.size());
#endif

// Ideally, this should not be here at all
//    if (root_policies.size()){
//    	for (int i=0; i<n_policies; ++i) {
//		delete root_policies[i];
//    	}
//    }
}

void TreeBRLPolicyPython::Reset()
{
    current_state = -1;
    current_action = -1;
}

void TreeBRLPolicyPython::Reset(int state)
{
    current_state = state;
    current_action = -1;
}

/// Full observation
real TreeBRLPolicyPython::Observe (int state, int action, real reward, int next_state, int next_action)
{
    if (state>=0) {
        belief->AddTransition(state, action, reward, next_state);
    }
    current_state = next_state;
    current_action = next_action;
    return 0.0;
}
/// Partial observation
real TreeBRLPolicyPython::Observe (real reward, int next_state, int next_action)
{
    if (current_state >= 0 && current_action >= 0) {
        belief->AddTransition(current_state, current_action, reward, next_state);
    }
    current_state = next_state;
    current_action = next_action;
    return 0.0;
}


/// Get an action using the current exploration policy.
/// it calls Observe as a side-effect.
int TreeBRLPolicyPython::Act(real reward, int next_state)
{
    assert(next_state >= 0 && next_state < n_states);

    T++;
    if (current_state >= 0 && current_action >= 0) {
		qlearning->Observe(reward,current_state,current_action);
        belief->AddTransition(current_state, current_action, reward, next_state);
//printf("adding transtion %d %d %f %d and deleting root policy\n",current_state, current_action, reward, next_state);
    if (algorithm == PLC) delete root_policy; //this is not a good place to delete root_policy here
    }

    current_state = next_state;

#ifdef TBRL_DEBUG
	logmsg("Acting belief");
	belief->ShowModelStatistics();
#endif

    //int n_MDP_leaf_samples = 1;
    int next_action = -1;
    if (algorithm == PLCAVG || algorithm == WRNGPLCAVG || algorithm == RANDOM){
    BeliefState belief_state = CalculateSparserBeliefTree(n_samples , K_step, n_policies);
    next_action = ArgMax(Qs);
    }
    else if (algorithm == PLC){
    BeliefState belief_state = CalculateSparserBeliefTree(n_samples , K_step, n_policies);
    int policy_index = ArgMax(Qs);

        if (root_policies.size()) {
    	    FixedDiscretePolicy* policy = root_policies[policy_index];
    	    root_policy = new FixedDiscretePolicy(n_states,n_actions,policy->p);
    	    //policy->Reset( current_state );
    	    //int next_action = policy->SelectAction();

    	    next_action = ArgMax( policy->getActionProbabilities(current_state) ) ;
	}
	else {
	    Swarning("root_policies size = 0\n");
	    next_action = 0;
	}
    }
    else if (algorithm == FULL){
    BeliefState belief_state = CalculateBeliefTree();
    next_action = ArgMax(Qs);
    }
    else if (algorithm == SPARSE){
    BeliefState belief_state = CalculateSparseBeliefTree(5,1);
    next_action = ArgMax(Qs);
    }


	//Qs.printf(stdout);

#ifdef TBRL_DEBUG7
	//printf("action values are a0:%f a1:%f \n",Qs(0),Qs(1));
    for (int j=0; j<n_policies; ++j) {
	printf("Printing Q value and policy\n");
	printf("Q-value of policy-%d is %f\n",j,Qs(j) );
	root_policies[j]->Show();
	}
	printf("next policy is:%d\n",policy_index);
#endif
    //policy=NULL;
#ifdef TBRL_DEBUG7
    printf("tree-size%d\n",size);
#endif
#ifdef TBRL_DEBUG8
	printf("no. of root policies here is:%d \n",root_policies.size());
#endif

    //Deleting root_policies after every-step
    if (algorithm == PLC && root_policies.size()){
    	for (int i=0; i<n_policies; ++i) {
		delete root_policies[i];
    	}
    	root_policies.clear();
    }
    current_action = next_action;
    return current_action;
}

    /// Calculate a sparse belief tree where we take n_samples state
    /// samples and use n_TS MDP samples for the upper and lower bounds at
    /// the leaf nodes

TreeBRLPolicyPython::BeliefState TreeBRLPolicyPython::CalculateSparseBeliefTree(int n_samples, int n_TS)
{
    // Initialise the root belief state
    BeliefState belief_state(*this, belief, current_state);
    belief_state.SparseExpandAllActions(n_samples);
    belief_state.CalculateValues(leaf_node_expansion,n_actions);
    //belief_state.CalculateLowerBoundValues(n_TS),
    //belief_state.CalculateUpperBoundValues(n_TS));
    return belief_state;

}

    /// Calculate a sparse belief tree where we take n_samples optimal-MDP polices
    /// and run them for K-steps before doing backup.
    /// We also use n_TS MDP samples for the upper and lower bounds at
    /// the leaf nodes

TreeBRLPolicyPython::BeliefState TreeBRLPolicyPython::CalculateSparserBeliefTree(int n_samples, int K_step, int n_TS)
{
    // Initialise the root belief state
    BeliefState belief_state(*this, belief, current_state);
    int buffer = 0;
    switch(algorithm) {
	case PLCAVG: belief_state.SparserAverageExpandAllActions(n_samples,n_policies,K_step); buffer = n_actions; break;
	case PLC: belief_state.SparserExpandAllActions(n_samples,n_policies,K_step); buffer = n_policies; break;
//	case RANDOM: belief_state.SparserRandomExpandAllActions(n_samples,n_policies,K_step); buffer = n_actions; break;
	case WRNGPLCAVG: belief_state.SparserExpandAllActions(n_samples,n_policies,K_step); buffer = n_actions; break;
	}

    belief_state.CalculateValues(leaf_node_expansion,buffer);
    //belief_state.CalculateLowerBoundValues(n_TS),
    //belief_state.CalculateUpperBoundValues(n_TS));
    return belief_state;

}

/// Calculate a belief tree.
TreeBRLPolicyPython::BeliefState TreeBRLPolicyPython::CalculateBeliefTree()
{
    // Initialise the root belief state
    BeliefState belief_state(*this, belief, current_state);
    belief_state.ExpandAllActions();
	belief_state.CalculateValues(leaf_node_expansion,n_actions);
	return belief_state;
}

//------------- Belief states ----------------//

TreeBRLPolicyPython::BeliefState::BeliefState(TreeBRLPolicyPython& tree_,
                                  const MDPModel* belief_,
                                  int state_) : tree(tree_), state(state_), probability(1), t(0)
{
	belief = belief_->Clone();
    tree.size++;
}

/// Use this to construct a subsequent belief state
TreeBRLPolicyPython::BeliefState::BeliefState(TreeBRLPolicyPython& tree_,
                                  const MDPModel* belief_,
                                  int prev_state_,
                                  int prev_action_,
                                  int state_,
                                  real r,
                                  real p,
                                  BeliefState* prev_)
	: tree(tree_),
	  state(state_),
	  prev_action(prev_action_),
	  prev_reward(r), probability(p), prev(prev_), t(prev_->t + 1)
{

#ifdef TBRL_DEBUG
	logmsg("Cloning belief");
#endif
	belief = belief_->Clone();
#ifdef TBRL_DEBUG
	logmsg("Adding new transition");
#endif
    belief->AddTransition(prev_state_,
						  prev_action,
						  prev_reward,
						  state);
	tree.size++;
#ifdef TBRL_DEBUG
	logmsg("Previous belief");
	belief_->ShowModelStatistics();
#endif
#ifdef TBRL_DEBUG
	printf("%.2f",prev_reward);
	logmsg(" Next belief\n");
	belief->ShowModelStatistics();
	logmsg("Tree size: %d\n", tree.size);
#endif

}

// Using this to construct subsequent belief-state for sparser tree
TreeBRLPolicyPython::BeliefState::BeliefState(TreeBRLPolicyPython& tree_,
                                  MDPModel* belief_,
				  int prev_action_,
                                  int state_,
                                  real prev_total_r,
				  real p,
                                  BeliefState* prev_,
				real discount_factor_				)
	: tree(tree_),
	  belief(belief_),
	  state(state_),
	  prev_action(prev_action_),
	  prev_reward(prev_total_r), probability(p), prev(prev_), t(prev_->t + 1), discount_factor(discount_factor_)
{


	tree.size++;

#ifdef TBRL_DEBUG
	logmsg("Already cloned belief\n");
//	belief_->ShowModelStatistics();
	printf("prev-pointer:%p this-pointer:%p\n",(void*)prev_,(void*)this);
	logmsg("Tree size: %d t:%d s:%d a:%d\n", tree.size,t,state,prev_action);
#endif

}

TreeBRLPolicyPython::BeliefState::~BeliefState()
{
	tree.size--;
	for (uint i=0; i<children.size(); ++i) {
        delete children[i];
    }
	delete belief;
}


// Generate n-sample MDPs, calculate their optimal policy and run them for K_steps before creating new beliefs
void TreeBRLPolicyPython::BeliefState::SparserExpandAllActions(int n_samples,int n_policies, int K_step)
{
    if (t >= tree.horizon) {
        return;
    }
    real p = 1 / (real) (n_samples); /// < should be '*n_policies', no 'n_samples' is fine, since each K-step macro action is taken n_samples time

#ifdef TBRL_DEBUG6
	printf("t: %d\n",t);
//	belief->ShowModel();
#endif

		std::vector<RTDP*> RT_objects;
		std::vector<PolicyIteration*> PI_objects;
		std::vector<PPVI*> PPVI_objects;
    for (int i=0; i<n_policies; ++i) {
	DiscreteMDP* model = belief->generate();
//for (int i=0; i<tree.n_states; i++) for (int j=0;j<tree.n_actions;j++) model->setFixedReward(i, j, tree.environment->getExpectedReward(i,j));
#ifdef TBRL_DEBUG6
	printf("sample no.%d \n",i);
	model->ShowModel();
#endif
if(tree.Use_RTDP)	RT_objects.push_back(new RTDP(model, tree.gamma, state));
//else	PPVI_objects.push_back(new PPVI(model, tree.gamma, 20, 10));
else	PI_objects.push_back(new PolicyIteration(model, tree.gamma));

	//delete model;		//cant delete as PI_objects depend on it
//	models[i] = model;	//So have to collect here to delete them later
    }

	int runs = 5;
  auto start = std::chrono::high_resolution_clock::now();
//	if (tree.T < 2000) runs = 1;
    #pragma omp parallel for num_threads(n_policies)
    for (int i=0; i<n_policies; ++i) {
//	printf("Threads: %d \n",omp_get_num_threads());
	//DiscreteMDP* model = belief->generate();
	//ValueIteration VI(model, tree.gamma);
	//VI.ComputeStateValuesStandard(1e-1);
	//FixedDiscretePolicy* policy = VI.getPolicy();

	//PolicyIteration PI(model, tree.gamma);
if(tree.Use_RTDP)	RT_objects[i]->ComputeStateValues(runs,K_step*5); /// << param for RTDP
//else	PPVI_objects[i]->ComputeStateValues(1e-3); ///< For PPVI
else	PI_objects[i]->ComputeStateValues(10,1e-3);
	//delete model;
    }
  auto finish = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(finish - start);
//  printf("PPVI time-taken:%d\n",duration);


    // Saving policies if they are at the root
    if (t == 0) {
	    for (int i=0; i<n_policies; ++i) {
if(tree.Use_RTDP)		tree.root_policies.push_back(RT_objects[i]->getPolicy());
//else		tree.root_policies.push_back(PPVI_objects[i]->getPolicy());
else		tree.root_policies.push_back(new FixedDiscretePolicy(tree.n_states,tree.n_actions,PI_objects[i]->policy->p));

	    }
    }

    // Creating n_policies policy
    std::vector<FixedDiscretePolicy*> policies;

    // Deleting pointers and adding policies
    for (int i=0; i<n_policies; ++i) {
if(tree.Use_RTDP)	{ policies.push_back(RT_objects[i]->getPolicy()); delete RT_objects[i];}
//else	{policies.push_back(PPVI_objects[i]->getPolicy()); delete PPVI_objects[i];}
else	{policies.push_back(new FixedDiscretePolicy(tree.n_states,tree.n_actions,PI_objects[i]->policy->p)); delete PI_objects[i];}
#ifdef TBRL_DEBUG
	printf("depth of tree is %d\n",t);
	printf("Printing policy i: %d\n",i);
	policies[i]->Show();
#endif
	//delete models[i];	//even deleting here says "illegal instruction" error
    }

    const int underlying_state = state;
    real discount_factor,total_reward;
    real gamma = tree.gamma;
    for (int j=0; j<n_policies; ++j) {
	FixedDiscretePolicy* policy = policies[j];

	for (int i=0; i<n_samples; ++i) {
	int init_state = underlying_state;
	#ifdef TBRL_DEBUG
		printf("init state: %d\n",init_state);
	#endif
	    policy->Reset(init_state);
	    int next_state = init_state;
	    //int next_action = policy->SelectAction();
    	    int next_action = ArgMax( policy->getActionProbabilities(underlying_state) ) ;
	    int initial_action = next_action;
	    total_reward = 0.0;
	    discount_factor = 1.0;
	    MDPModel* belief_clone = belief->Clone();
		real belief_dist = 0.0;
bool cond = true;
	    for (int k=0; k < K_step;++k)
	        {

		    //int next_action = ArgMax(policy->getActionProbabilities(state)); //choosing max_prob_action
//		    mean_mdp->Act(next_action);
//		    real r = mean_mdp->getReward();
//		    next_state = mean_mdp->getState();

/*	PREVIOUSLY HOW I DID IT, UNTIL 16/10/2018
		    next_state = belief_clone->GenerateTransition(next_state,next_action);
//		    real r = belief_clone->GenerateReward(next_state,next_action);
			real r = tree.environment->getExpectedReward(next_state,next_action);
*/

		    next_state = belief_clone->GenerateTransition(init_state,next_action);
//		    real r = belief_clone->GenerateReward(init_state,next_action);
			real r = belief_clone->getExpectedReward(init_state,next_action);
//			real r = tree.environment->getExpectedReward(init_state,next_action);


		    belief_clone->AddTransition(init_state,next_action,r,next_state);
	#ifdef TBRL_DEBUG
		printf("%d %d %d %f\n",init_state,next_action,r,next_state);
	#endif
	#ifdef TBRL_DEBUG9
			belief_dist += belief->CalculateDistance(belief_clone,init_state,next_action);
	#endif
		    policy->Observe (r, next_state);
		    total_reward += discount_factor*r;

		    init_state = next_state;
		    next_action = policy->SelectAction();
		    discount_factor *= gamma;
	#ifdef TBRL_DEBUG9
			if (belief_dist >= 1.0){ cond = false; break;}
	#endif
		}
//if (cond) printf("not reached\n");
        if (tree.algorithm == PLC){ children.push_back(new BeliefState(tree, belief_clone, j, next_state, total_reward, p, this, discount_factor)); }
	else if (WRNGPLCAVG) { children.push_back(new BeliefState(tree, belief_clone, initial_action, next_state, total_reward, p, this, discount_factor)); }
    	}
    }

    // Delete pointers
    for (int i=0; i<n_policies; ++i) {
	delete policies[i];
    }

    for (uint i=0; i<children.size(); ++i) {
        children[i]->SparserExpandAllActions(n_samples,n_policies,K_step);
    }
}

// Generate n-sample MDPs, calculate their optimal policy and run them for K_steps before creating new beliefs, averaging over value of all policies with same first action
// Can calculate probability of each child 'p' only for FixedDiscretePolicy case
void TreeBRLPolicyPython::BeliefState::SparserAverageExpandAllActions(int n_samples,int n_policies, int K_step)
{
    if (t >= tree.horizon) {
        return;
    }

#ifdef TBRL_DEBUG6
	printf("t: %d\n",t);
//	belief->ShowModel();
#endif
#ifdef TBRL_DEBUG
	printf("\nPolicy iteration called\n");
#endif
    std::vector<PolicyIteration*> PI_objects;
    for (int i=0; i<n_policies; ++i) {
	DiscreteMDP* model = belief->generate();
#ifdef TBRL_DEBUG6
	printf("sample no.%d \n",i);
	model->ShowModel();
#endif
	PI_objects.push_back(new PolicyIteration(model, tree.gamma));
	//delete model;		//cant delete as PI_objects depend on it
//	models[i] = model;	//So have to collect here to delete them later
    }

//if (t==0){
    #pragma omp parallel for num_threads(n_policies)
    for (int i=0; i<n_policies; ++i) {
//	printf("Threads: %d \n",omp_get_num_threads());
	//DiscreteMDP* model = belief->generate();
	//ValueIteration VI(model, tree.gamma);
	//VI.ComputeStateValuesStandard(1e-1);
	//FixedDiscretePolicy* policy = VI.getPolicy();

	//PolicyIteration PI(model, tree.gamma);
	PI_objects[i]->ComputeStateValues(-1,1e-3);
	//delete model;
    }
//}


    // Creating n_policies policy
    std::vector<FixedDiscretePolicy*> policies;

    // Deleting pointers and adding policies
    for (int i=0; i<n_policies; ++i) {
	policies.push_back(new FixedDiscretePolicy(tree.n_states,tree.n_actions,PI_objects[i]->policy->p));
	delete PI_objects[i];
#ifdef TBRL_DEBUG
	printf("depth of tree is %d\n",t);
	printf("Printing policy i: %d\n",i);
	policies[i]->Show();
#endif
	//delete models[i];	//even deleting here says "illegal instruction" error
    }


    // Counting frequency of various actions taken as first actions by all the policies
    std::vector<int> freq(tree.n_actions,0.0);
    for (int i=0; i<n_policies; ++i) {
	policies[i]->Reset(state);
	freq[policies[i]->SelectAction()]+=1.0;
    }

    const int underlying_state = state;
    real discount_factor,total_reward;
    real gamma = tree.gamma;
    for (int j=0; j<n_policies; ++j) {
	FixedDiscretePolicy* policy = policies[j];
	policy->Reset(underlying_state);
	real p = 1 / (real) (n_samples*freq[policy->SelectAction()]);

	for (int i=0; i<n_samples; ++i) {
	int init_state = underlying_state;
	#ifdef TBRL_DEBUG
		printf("init state: %d\n",init_state);
	#endif
	    policy->Reset(init_state);
	    int next_state = init_state;
	    //int next_action = policy->SelectAction();
    	    int next_action = ArgMax( policy->getActionProbabilities(underlying_state) ) ;
	    int initial_action = next_action;
	    total_reward = 0.0;
	    discount_factor = 1.0;
	    MDPModel* belief_clone = belief->Clone();
	    for (int k=0; k < K_step;++k)
	        {

		    //int next_action = ArgMax(policy->getActionProbabilities(state)); //choosing max_prob_action
//		    mean_mdp->Act(next_action);
//		    real r = mean_mdp->getReward();
//		    next_state = mean_mdp->getState();

/*	PREVIOUSLY HOW I DID IT, UNTIL 16/10/2018
		    next_state = belief_clone->GenerateTransition(next_state,next_action);
		    real r = belief_clone->GenerateReward(next_state,next_action);
*/
		    next_state = belief_clone->GenerateTransition(init_state,next_action);
		    real r = belief_clone->GenerateReward(init_state,next_action);


		    belief_clone->AddTransition(init_state,next_action,r,next_state);
	#ifdef TBRL_DEBUG
		printf("%d %d %d %f\n",init_state,next_action,r,next_state);
	#endif
		    policy->Observe (r, next_state);
		    total_reward += discount_factor*r;

		    init_state = next_state;
		    next_action = policy->SelectAction();
		    discount_factor *= gamma;
		}
	//children.push_back(new BeliefState(tree, belief_clone, j, next_state, total_reward, p, this));
	children.push_back(new BeliefState(tree, belief_clone, initial_action, next_state, total_reward, p, this, discount_factor));
    	}
    }

    // Delete pointers
    for (int i=0; i<n_policies; ++i) {
	delete policies[i];
    }

    for (uint i=0; i<children.size(); ++i) {
        children[i]->SparserAverageExpandAllActions(n_samples,n_policies,K_step);
    }
}




/// Generate transitions from the current state for all
/// actions. Do this recursively, using the marginal
/// distribution, but using sparse sampling.
///
void TreeBRLPolicyPython::BeliefState::SparseExpandAllActions(int n_samples)
{
    if (t >= tree.horizon) {
        return;
    }
    real p = 1 / (real) n_samples;
    for (int k=0; k<n_samples; ++k) {
        for (int a=0; a<tree.n_actions; ++a) {
            int next_state = belief->GenerateTransition(state, a);
            real reward = belief->GenerateReward(state, a); 	//too stochastic to generate, without sufficient H
	    //real reward = belief->getExpectedReward(state,a);
            // Generate the new belief state and put it in the tree
            children.push_back(new BeliefState(tree, belief, state, a, next_state, reward, p, this));
        }
    }

    for (uint i=0; i<children.size(); ++i) {
        children[i]->SparseExpandAllActions(n_samples);
    }
}
/// Generate transitions from the current state for all
/// actions. Do this recursively, using the marginal distribution. Only expand the children when we're under the horizon.
void TreeBRLPolicyPython::BeliefState::ExpandAllActions()
{
    if (t >= tree.horizon) {
        return;
    }
    for (int a=0; a<tree.n_actions; ++a) {
        for (int next_state=0;
             next_state<tree.n_states;
             ++next_state) {
            real p = belief->getTransitionProbability(state, a, next_state);
		for (int reward=0;reward<2;++reward){
	    //real reward = belief->getExpectedReward(state,a);
			real q = belief->getRewardProbability(state, a, reward);

            children.push_back(new BeliefState(tree, belief, state, a, next_state, reward, p*q, this));
			}
		}
    }

    for (uint i=0; i<children.size(); ++i) {
        children[i]->ExpandAllActions();
    }
}



/// Return the values using Backwards induction on the already
/// constructed MDP.
///
/// If w is the current belief state, and w' the successor, while a is
/// our action, then we can write the value function recursion:
///
/// \f$V_t(w) = \max_a Q_t(w, a) = \max_a E\{r(w,a,w') + \gamma V_{t+1} (w')\}\f$,
///
/// where the expectation is
/// \f$Q_t(w, a) = \sum_{s'} {r(w,a,s') + \gamma P(s' | a, s) V_{t+1} (w')\}\f$ and \f$w' = w( | s, a, s')\f$.
real TreeBRLPolicyPython::BeliefState::CalculateValues(LeafNodeValue leaf_node, int buffer)
{
//    Vector Q(tree.n_policies);   // Keep in mind if n_policies > n_actions
//    Vector Q(tree.n_actions);
    Vector Q(buffer);
    real V = 0;
    real discount = tree.gamma;

    if (t < tree.horizon) {
        for (uint i=0; i<children.size(); ++i) {
            int a = children[i]->prev_action;
			real p = children[i]->probability;
		    if (tree.algorithm == PLC) discount = children[i]->discount_factor;
			real r = children[i]->prev_reward;
			int s_next = children[i]->state;
			real V_next = children[i]->CalculateValues(leaf_node,buffer);
            Q(a) += p * (r + discount * V_next);
#ifdef TBRL_DEBUG2
			printf("t:%d s:%d i:%d a:%d p:%f s2:%d, r:%f v:%f\n",
				   t, state, i, a, p, s_next, r, V_next);
#endif
        }
        V += Max(Q);
#ifdef TBRL_DEBUG
		Q.print(stdout); printf(" %d/%d\n", t, tree.horizon);
#endif
    } else {
		switch(leaf_node) {
		case NONE: V = 0; break;
		case Q_LEARNING: V = Qlearning(); break;
		case V_MIN: V = 0; break;
		case V_MAX: V = 1.0 / (1.0 - discount); break;
		case V_MEAN: V = MeanMDPValue(); break;
		case V_UTS: V = UTSValue(); break;
		case V_LTS: V = LTSValue(); break;
		}
    }



    if (t==0) {
        tree.Qs = Q;
    }
    return V;
}

/// Return the values using an upper bound
real TreeBRLPolicyPython::BeliefState::UTSValue()
{
	const int n_samples = 2;
	real V = 0;
    std::vector<ValueIteration*> VI_objects;
	for (int i=0; i<n_samples; ++i) {
		DiscreteMDP* model = belief->generate();
		VI_objects.push_back(new ValueIteration(model, tree.gamma));
	}
    #pragma omp parallel for num_threads(n_samples)
	for (int i=0; i<n_samples; ++i) {
//	printf("Threads: %d \n",omp_get_num_threads());
		VI_objects[i]->ComputeStateValuesStandard(1e-2);
	}
	for (int i=0; i<n_samples; ++i) {
		V += VI_objects[i]->getValue(state);
		delete VI_objects[i];
	}
	V /=  (real) n_samples;
	return V;
}

//real TreeBRLPolicyPython::BeliefState::UTSValue()
//{
//	int n_samples = 2;
//	real V = 0;
//	for (int i=0; i<n_samples; ++i) {
//		DiscreteMDP* model = belief->generate();
//		ValueIteration VI(model, tree.gamma);
//		VI.ComputeStateValuesStandard(1e-3);
//		V += VI.getValue(state);
//		delete model;
//	}
//	V /=  (real) n_samples;
//	return V;
//}


/// Return the values using a lower bound
real TreeBRLPolicyPython::BeliefState::LTSValue()
{
    real discount = tree.gamma;
	int n_samples = 2;
	const DiscreteMDP* model = belief->getMeanMDP();
	ValueIteration VI(model, discount);
	VI.ComputeStateValuesStandard(1e-3);
	FixedDiscretePolicy* policy = VI.getPolicy();
	real V_next = 0;
	for (int i=0; i<n_samples; ++i) {
		DiscreteMDP* model = belief->generate();
		PolicyEvaluation PI(policy, model, tree.gamma);
		PI.ComputeStateValues(1e-3);
		V_next += PI.getValue(state);
		delete model;
	}
	real V = V_next / (real) n_samples;

    return V;
}

/// Return the values using the mean MDP
real TreeBRLPolicyPython::BeliefState::MeanMDPValue()
{
    real discount = tree.gamma;
	const DiscreteMDP* model = belief->getMeanMDP();
	ValueIteration VI(model, discount);
	VI.ComputeStateValuesStandard(1e-3);
	return VI.getValue(state);
}

/// Return the values using the mean MDP
real TreeBRLPolicyPython::BeliefState::Qlearning()
{
	VFExplorationPolicy* policy = tree.qlearning->exploration_policy->Clone();
	int n_samples = tree.n_samples;
	n_samples = 1;
	Vector vals(n_samples);
	for (int i=0; i<n_samples; ++i) {

		policy->Observe(0,state);

		int init_state,next_state,K_step,next_action;
		K_step = 80 - int(t*tree.K_step);
		init_state = state;
		next_action = policy->SelectAction();

	    real total_reward = 0.0;
	    real discount_factor = 1.0;
		real gamma = tree.gamma;
	    MDPModel* belief_clone = belief->Clone();
	    for (int k=0; k < K_step;++k)
	        {
		    next_state = belief_clone->GenerateTransition(init_state,next_action);
//		    real r = belief_clone->GenerateReward(init_state,next_action);
/*	PREVIOUSLY HOW I DID IT, UNTIL 16/10/2018
			real r = tree.environment->getExpectedReward(next_state,next_action);
*/
			real r = tree.environment->getExpectedReward(init_state,next_action);

		    belief_clone->AddTransition(init_state,next_action,r,next_state);
		    policy->Observe (r, next_state);
		    total_reward += discount_factor*r;

		    init_state = next_state;
		    next_action = policy->SelectAction();
		    discount_factor *= gamma;
		}
		delete belief_clone;
		vals[i] = total_reward;
	}
	delete policy;
	return vals.Sum()/n_samples;
}

void TreeBRLPolicyPython::saveBelief(int timestep,int other_encoding,std::string path)
{
	std::ofstream file_obj;
	std::string name;
	if (path == "") {
		if (other_encoding!=-1) name = std::string("Belief-"+std::to_string(timestep)+"-"+std::to_string(other_encoding));
		else name = std::string("Belief-"+std::to_string(timestep));
	}
	else {
		if (other_encoding!=-1) name = std::string(path+"Belief-"+std::to_string(timestep)+"-"+std::to_string(other_encoding));
		else name = std::string(path+"Belief-"+std::to_string(timestep));
	}
  file_obj.open(name, std::ios::trunc);

	DiscreteMDPCountsSparse* pd;
  pd = static_cast<DiscreteMDPCountsSparse*>(belief);//Before writing to file, downcasting (Base to Derived) is necessary. Dynamic or Static both should work
	file_obj.write((char*)pd, sizeof(*pd));

	file_obj.close();
}

void TreeBRLPolicyPython::loadBelief(std::string name)
{
	std::fstream is;
//	std::string name("Belief-"+std::to_string(timestep));
	is.open(name, std::ios::in);

	is.seekg (0, is.end);
	int length = is.tellg();
	is.seekg (0, is.beg);
	char * buffer = new char [length];
	is.read(buffer, length);
	is.close();

	MDPModel *obj = reinterpret_cast<MDPModel *>(buffer);
//	obj->ShowModelStatistics();
}
