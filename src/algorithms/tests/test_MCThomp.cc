/* -*- Mode: C++; -*- */
// copyright (c) 2006-2017 by Christos Dimitrakakis <christos.dimitrakakis@gmail.com>
/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ***************************************************************************/
#ifdef MAKE_MAIN

/// Bayesian RL includes
#include "DiscreteMDPCountsSparse.h"

/// The main thing to test
#include "MCThompsonSampling.h"
#include "ValueIteration.h"
#include "UCRL2.h"
#include "TreeBRLPolicy.h"
//for comparision
#include "MDPModel.h"
//#include "SampleBasedRL.h"

/// The basic environments
#include "GuezEnv.h"
#include "simulator.h"
#include "grid.h"
#include "maze.h"
#include "RiverSwim.h"
//#include "chain.h"
#include "DiscreteChain.h"
#include "RandomMDP.h"

/// Other things
#include "MersenneTwister.h"

/// STD
#include <iostream>
#include <memory>
#include <algorithm> //for shuffle
using namespace std;



real time_limit = 1; // << time-limit per-step in seconds


real RunExperiment(shared_ptr<DiscreteEnvironment> environment,
                   MCThompsonSampling& tree, 
                   int n_steps);
real RunExperiment(shared_ptr<DiscreteEnvironment> environment,
                   TreeBRLPolicy& tree, 
                   int n_steps);
real RunExperiment(shared_ptr<DiscreteEnvironment> environment,
                   int n_steps,UCRL2& ucrl);
real RunExperiment(shared_ptr<DiscreteEnvironment> environment, 
                   int n_steps);


int main(int argc, char** argv) {


    // ONly needed for UCRL2
    RandomNumberGenerator* rng;
    MersenneTwisterRNG mersenne_twister;
    rng = (RandomNumberGenerator*) &mersenne_twister;
    rng->manualSeed(23);
	real delta = 0.05; // << For UCRL2
	////

    int n_states = 5;
    int n_actions = 2;
    real discounting = 0.99;
    int n_steps = 1;
    int n_experiments = 1;

    // ---- user options ---- //
	int planner = 0; // 0 - MCTS, 1 - UCRL2, 2 - TreeBRL
    int useOptimalPolicy = 1; // << To compute regret later
	int planning_horizon = 2;
    int leaf_value = MCThompsonSampling::LeafNodeValue::NONE;
    int algorithm = MCThompsonSampling::WhichAlgo::MCTS;
	real dirichlet_mass = 2.0;
	bool useFixedRewards = false; // << Using Fixed Rewards
	bool useRTDP = false; // << Using RTDP for candidate policy generation

	if (argc > 1) {
		algorithm = atoi(argv[1]);
	}
	if (argc > 2) {
//		useOptimalPolicy = atoi(argv[2]);
		useRTDP = atoi(argv[2]);
	}
	if (argc > 3) {
		n_steps = atoi(argv[3]);
	}
	if (argc > 4) {
		n_experiments = atoi(argv[4]);
	}
	if (argc > 5) {
//		leaf_value = atoi(argv[2]);
		discounting = atof(argv[2]);
	}
	
    //printf("# Making environment\n");
    shared_ptr<DiscreteEnvironment> environment;

	//SIMULATOR* sim = new Maze(discounting);
	//SIMULATOR* sim = new Grid(5,discounting);
	//SIMULATOR* sim = new Dloop(discounting);
	//SIMULATOR* sim = new Chain(discounting);
	//double p[16] = {0.6,0.6,0.6,0.6,0.6,0.6,0.6,0.6,0.6,0.6,0.6,0.6,0.9,0.6,0.6,0.6};
	//SIMULATOR* sim = new BANDIT(16,p,discounting);
	//environment = make_shared<GuezEnv>(sim);
	environment = make_shared<RiverSwim>();
	//environment = make_shared<DiscreteChain>(n_states);
	//environment = make_shared<RandomMDP>(50,2,0.2,-0.1,-5,20,rng,false);
	

	//Collecting information for belief
    n_actions = environment->getNActions();
    n_states = environment->getNStates();
	
    //simplify things by fixing the rewards
    Matrix rewards(n_states, n_actions);
    for (int s=0; s<n_states; ++s) {
        for (int a=0; a<n_actions; ++a) {
            rewards(s,a) = environment->getExpectedReward(s, a);
        }
    }

	time_t past_time = time(NULL);
    Vector U(n_experiments);
    for (int experiment=0;
         experiment<n_experiments;
         experiment++) {

	if (planner == 0) {
	    enum DiscreteMDPCountsSparse::RewardFamily reward_prior = DiscreteMDPCountsSparse::BETA;
	    DiscreteMDPCountsSparse belief(n_states, n_actions, dirichlet_mass,reward_prior);
if(useFixedRewards) belief.setFixedRewards(rewards);

      MCThompsonSampling tree (n_states, n_actions, discounting, &belief, (MCThompsonSampling::LeafNodeValue) leaf_value,(MCThompsonSampling::WhichAlgo) algorithm,useRTDP);
	  real total_reward = 0;
	  if (useOptimalPolicy)
      total_reward = RunExperiment(environment, n_steps); ///<This real leaks memory
	  else
      total_reward = RunExperiment(environment, tree, n_steps);
	if (total_reward == -1) return 0;
        printf("H:%d,\tV:%d,\tR:%f\n", planning_horizon, leaf_value,total_reward);
        U(experiment) = total_reward;
	}


		if (planner == 1) {
		//For UCRL2
		DiscreteMDPCounts* ucrl_model = new DiscreteMDPCounts(n_states,n_actions,dirichlet_mass, DiscreteMDPCounts::BETA);
if(useFixedRewards) ucrl_model->setFixedRewards(rewards);
		UCRL2 ucrl = UCRL2(n_states,n_actions,discounting,ucrl_model,rng,delta);
        real total_reward = RunExperiment(environment,n_steps,ucrl); ///<This real leaks memory
	if (total_reward == -1) return 0;
        printf("H:%d,\tV:%d,\tR:%f\n", planning_horizon, leaf_value,total_reward);
        U(experiment) = total_reward;
		}


	if (planner == 2) {
	    enum DiscreteMDPCountsSparse::RewardFamily reward_prior = DiscreteMDPCountsSparse::NORMAL;
	    DiscreteMDPCountsSparse belief(n_states, n_actions, dirichlet_mass,reward_prior);
if(useFixedRewards) belief.setFixedRewards(rewards);
		int n_policies = 2;
		int n_samples = 16;
		int K_step = 5;
        TreeBRLPolicy tree (environment,n_states, n_actions, discounting, &belief, rng, planning_horizon, (TreeBRLPolicy::LeafNodeValue) leaf_value,(TreeBRLPolicy::WhichAlgo) 1,
							n_policies,n_samples, K_step, useRTDP);
	  real total_reward = 0;
      total_reward = RunExperiment(environment, tree, n_steps);
	if (total_reward == -1) return 0;
        printf("H:%d,\tV:%d,\tR:%f\n", planning_horizon, leaf_value,total_reward);
        U(experiment) = total_reward;
	}




	}



    printf("L:%f,\tM:%f,\tU:%f T:%f\n",
           Min(U),
           U.Sum() / (real) U.Size(),
           Max(U),
			difftime(time(NULL),past_time));

	return 0;

}


real RunExperiment(shared_ptr<DiscreteEnvironment> environment,
                   MCThompsonSampling& tree, int n_steps)
{
    environment->Reset();
//    tree.Reset(environment->getState());
    tree.Reset();
	int n_actions = tree.n_actions;
	int n_states = tree.n_states;
    real reward = environment->getReward();
    real total_reward = 0;
	tree.CalculateRootPolicy(1,10,-1);
	real *arr;arr = (real *)calloc(n_states*n_actions , sizeof(real));
	real *global;global = (real *)calloc(n_states*n_actions , sizeof(real));
	time_t past_time_exp = time(NULL);
    for (int t=0; t<n_steps; ++t) {
        int state = environment->getState();
//printf("s:%d,st:%d\n",state,tree.current_state);
       int action = tree.Act(reward, state);
//		tree.CalculateRootPolicy(4,10,state);
		arr[state*n_actions+action] += 1.0f;
		if ( arr[state*n_actions+action] > global[state*n_actions+action] ){
//printf("state:%d,action:%d global:%f addition:%f ",state,action,global[state*n_actions+action],arr[state*n_actions+action]); 
			for (int i=0; i<tree.n_states; i++) for(int j=0;j<tree.n_actions;j++) { global[state*n_actions+action] += arr[state*n_actions+action]; arr[state*n_actions+action]=0.0f; }
//printf(" new-global:%f\n",global[state*n_actions+action]);
			tree.CalculateRootPolicy(1,20,state);
		}
//	if ( difftime(time(NULL),past_time_exp) > time_limit*n_steps ) {printf("Time limit passed"); return -1;}

        bool action_OK = environment->Act(action);
//        reward = environment->getExpectedReward(state,action);
        reward = environment->getReward();
        total_reward += reward;
        printf("%d %d %d %f %d # s a r next_s\n", t, state, action, reward,environment->getState());
        if (!action_OK) {
//		tree.CalculateRootPolicy(4,10);
            state = environment->getState();
            reward = environment->getReward();
            tree.Act(reward, state);
            environment->Reset();
            tree.Reset(environment->getState());
        }

    }
    return total_reward;
}


real RunExperiment(shared_ptr<DiscreteEnvironment> environment,
                   int n_steps)
{
    environment->Reset();
    int n_actions = environment->getNActions();
    int n_states = environment->getNStates();

	PolicyIteration PI(environment->getMDP(), 0.99); // << interestingly somehow PolicyIteration is giving wrong answer for Chain all the way, need to check that !!
	PI.ComputeStateValues(1e-2);
	FixedDiscretePolicy optimalPolicy = FixedDiscretePolicy(n_states,n_actions,PI.policy->p);
//optimalPolicy.Show();

    real reward = environment->getReward();
    real total_reward = 0;
    for (int t=0; t<n_steps; ++t) {
        int state = environment->getState();
       int action = ArgMax( optimalPolicy.getActionProbabilities(state) ) ;

        bool action_OK = environment->Act(action);
//        reward = environment->getExpectedReward(state,action);
        reward = environment->getReward();
        total_reward += reward;
        printf("%d %d %d %f %d # s a r next_s\n", t, state, action, reward,environment->getState());
        if (!action_OK) {
            state = environment->getState();
            reward = environment->getReward();
            environment->Reset();
        }

    }
    return total_reward;
}


real RunExperiment(shared_ptr<DiscreteEnvironment> environment,
                   int n_steps,UCRL2& ucrl )
{
    environment->Reset();
    ucrl.Reset();
    real reward = environment->getReward();
    real total_reward = 0;
	time_t past_time_exp = time(NULL);
    for (int t=0; t<n_steps; ++t) {
        int state = environment->getState();
		int action = ucrl.Act(reward,state);
	if ( difftime(time(NULL),past_time_exp) > time_limit*n_steps ) {printf("Time limit passed"); return -1;}
        bool action_OK = environment->Act(action);
//        reward = environment->getExpectedReward(state,action);
        reward = environment->getReward();
        total_reward += reward;
        printf("%d %d %d %f %d # s a r next_s\n", t, state, action, reward,environment->getState());
        if (!action_OK) {
            state = environment->getState();
            reward = environment->getReward();
            ucrl.Act(reward, state);
            environment->Reset();
            ucrl.Reset();
        }
    }
    return total_reward;
}

/*
real RunExperiment(shared_ptr<DiscreteEnvironment> environment,
                   TreeBRLPolicy& tree, int n_steps)
{
    environment->Reset();
    tree.Reset(environment->getState());
//    tree.Reset();
    real reward = environment->getReward();
    real total_reward = 0;
	time_t past_time_exp = time(NULL);
    for (int t=0; t<n_steps; ++t) {
        int state = environment->getState();
       int action = tree.Act(reward, state);
//	if ( difftime(time(NULL),past_time_exp) > time_limit*n_steps ) {printf("Time limit passed"); return -1;}

        bool action_OK = environment->Act(action);
//        reward = environment->getExpectedReward(state,action);
        reward = environment->getReward();
        total_reward += reward;
        printf("%d %d %d %f %d # s a r next_s\n", t, state, action, reward,environment->getState());
        if (!action_OK) {
            state = environment->getState();
            reward = environment->getReward();
            tree.Act(reward, state);
            environment->Reset();
            tree.Reset(environment->getState());
        }

    }
    return total_reward;
}
*/

real RunExperiment(shared_ptr<DiscreteEnvironment> environment,
                   TreeBRLPolicy& tree, int n_steps)
{

    int n_actions = environment->getNActions();
    int n_states = environment->getNStates();

	//Doubling-trick
	real *arr;arr = (real *)calloc(n_states*n_actions , sizeof(real));
	real *global;global = (real *)calloc(n_states*n_actions , sizeof(real));
	bool condition = true;
	int counter = 0;

	//Setting env
    environment->Reset();
    tree.Reset(environment->getState());
    real reward = environment->getReward();
    real total_reward = 0;

	while (counter < n_steps) {
        int state = environment->getState();
        int action = tree.Act(reward, state);
	while (condition)
	{
        bool action_OK = environment->Act(action);
        reward = environment->getExpectedReward(state,action);
//        if (!action_OK) reward = environment->getReward(); //For Mazeworld only
        total_reward += reward;
        if (!action_OK) {
            environment->Reset();
        }
        printf("%d %d %f %d # s a r next_s\n", state, action, reward,environment->getState());
        tree.belief->AddTransition(state, action, reward, environment->getState());

		//Conditions
		arr[state*n_actions+action] += 1.0f;
		if ( arr[state*n_actions+action] > global[state*n_actions+action] ){
			for (int i=0; i<n_states; i++) for(int j=0;j<n_actions;j++) { global[state*n_actions+action] += arr[state*n_actions+action]; arr[state*n_actions+action]=0.0f; }
			condition = false;
		}
		counter+=1;

		if (counter >= n_steps) break;
		//----------------

        state = environment->getState();
		action = ArgMax( tree.root_policy->getActionProbabilities(state) ) ;
	}
	delete tree.root_policy;
	tree.Reset(-1); //cause we dont want to add the transition again when calling Act
	condition = true;
    }
    return total_reward;
}


#endif
