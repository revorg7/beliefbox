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

/// Other things
#include "MersenneTwister.h"

/// For Guez Envs
#include "GuezEnv.h"
#include "grid.h"
#include "simulator.h"
#include "maze.h"
#include "doubleloop.h"
#include "chain.h"
//#include "banditSim.h"

/// Bayesian RL includes
#include "DiscreteMDPCountsSparse.h"

/// The main thing to test
#include "TreeBRLPolicy.h"
//for comparision
#include "MDPModel.h"
#include "SampleBasedRL.h"
#include "UCRL2.h"

/// STD
#include <iostream>
#include <memory>
#include <time.h> 
using namespace std;

real time_limit = 1; // << time-limit per-step in seconds

real RunExperiment(shared_ptr<DiscreteEnvironment> environment,
                   int n_steps, SampleBasedRL* sampling);
real RunExperiment(shared_ptr<DiscreteEnvironment> environment,
                   int n_steps,UCRL2& ucrl);
real RunExperiment(shared_ptr<DiscreteEnvironment> environment,
                   TreeBRLPolicy& tree,
                   int n_steps);

int main(int argc, char** argv) {
    // use a high-quality RNG for the main program
    RandomNumberGenerator* rng;
    MersenneTwisterRNG mersenne_twister;
    rng = (RandomNumberGenerator*) &mersenne_twister;

    rng->seed();
    real discounting = 0.95;
    int n_steps = 1000;
    int n_experiments = 1;



    //    int n_samples = 2; ///< number of state samples when branching
    //int n_mdp_samples = 2; ///< number of MDP samples at leaf nodes

    // ---- user options ---- //
    int planning_horizon = 1; 
    int leaf_value = TreeBRLPolicy::LeafNodeValue::NONE;
    int algorithm = TreeBRLPolicy::WhichAlgo::PLC;
    int n_policies = 4;
	int n_samples = 4;
	int K_step = 13;
	real dirichlet_mass = 2.0;	//INTIAL VALUE IN GUEZ CODE
	int planner = 0; // 0 - Sparser, 1 - UCRL2, 2 - SampleBased
	real delta = 0.99; // << For UCRL2
	int max_samples = 2000; // << For SampleBased
	real epsilon = 0.01; // << For SampleBased
	bool useRTDP = false; // << Using RTDP for candidate policy generation
	bool useFixedRewards = true; // << Using Fixed Rewards

	if (argc > 1) {
		n_policies = atoi(argv[1]);
	}
	if (argc > 2) {
		n_samples = atoi(argv[2]);
	}
	if (argc > 3) {
		K_step = atoi(argv[3]);
	}
	if (argc > 4) {
		useRTDP = atoi(argv[4]);
	}
	if (argc > 5) {
		planning_horizon = atoi(argv[5]);
	}
	if (argc > 6 && planner==0) {
		planner = atoi(argv[6]);
	}
	if (argc > 7) {
		useFixedRewards = atoi(argv[7]);
	}
    //printf("# Making environment\n");
    shared_ptr<DiscreteEnvironment> environment;

	//SIMULATOR* sim = new Maze(discounting);
	SIMULATOR* sim = new Grid(5,discounting);
	//SIMULATOR* sim = new Dloop(discounting);
	//SIMULATOR* sim = new Chain(discounting);
	//double p[16] = {0.6,0.6,0.6,0.6,0.6,0.6,0.6,0.6,0.6,0.6,0.6,0.6,0.9,0.6,0.6,0.6};
	//SIMULATOR* sim = new BANDIT(16,p,discounting);
	environment = make_shared<GuezEnv>(sim);
    
	//Collecting information for belief
    int n_actions = environment->getNActions();
    int n_states = environment->getNStates();
	
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
		// For TreeBRLPolicy
	    enum DiscreteMDPCountsSparse::RewardFamily reward_prior = DiscreteMDPCountsSparse::BETA;
	    DiscreteMDPCountsSparse belief(n_states, n_actions, dirichlet_mass, reward_prior);
if(useFixedRewards) belief.setFixedRewards(rewards);
        TreeBRLPolicy tree (environment,n_states, n_actions, discounting, &belief, rng, planning_horizon, (TreeBRLPolicy::LeafNodeValue) leaf_value,(TreeBRLPolicy::WhichAlgo) algorithm,
							n_policies,n_samples, K_step, useRTDP);
        real total_reward = RunExperiment(environment, tree, n_steps); ///<This real leaks memory
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
		// Adding for comparision to USAMPLING
	    enum DiscreteMDPCountsSparse::RewardFamily reward_prior = DiscreteMDPCountsSparse::BETA;
        DiscreteMDPCountsSparse discrete_mdp =  DiscreteMDPCountsSparse(n_states, n_actions, dirichlet_mass, reward_prior);
if(useFixedRewards) discrete_mdp.setFixedRewards(rewards);
		
        SampleBasedRL* sampling = new SampleBasedRL(n_states, ///<this leaks memory too
                                          n_actions,
                                          discounting,
                                          epsilon,
                                          &discrete_mdp,
                                          rng,
                                          max_samples,
                                          true);

        real total_reward = RunExperiment(environment,n_steps,sampling); ///<This real leaks memory
	if (total_reward == -1) return 0;
        printf("H:%d,\tV:%d,\tR:%f\n", planning_horizon, leaf_value,total_reward);
        U(experiment) = total_reward;
		}
    }
    printf("L:%f,\tM:%f,\tU:%f T:%f\n",
           Min(U),
           U.Sum() / (real) U.Size(),
           Max(U),
			difftime(time(NULL),past_time) );

    return 0;
}

real RunExperiment(shared_ptr<DiscreteEnvironment> environment,
                   TreeBRLPolicy& tree, int n_steps)
{
//    environment->Reset();
//    tree.Reset(environment->getState());
    tree.Reset();
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


real RunExperiment(shared_ptr<DiscreteEnvironment> environment,
                   int n_steps,UCRL2& ucrl )
{
//    environment->Reset();
//    tree.Reset(environment->getState());
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

real RunExperiment(shared_ptr<DiscreteEnvironment> environment,
                   int n_steps,SampleBasedRL* sampling)
{
//    environment->Reset();
//    tree.Reset(environment->getState());
    sampling->Reset();
    real reward = environment->getReward();
    real total_reward = 0;
	time_t past_time_exp = time(NULL);
    for (int t=0; t<n_steps; ++t) {
        int state = environment->getState();
		int action = sampling->Act(reward,state);
	if ( difftime(time(NULL),past_time_exp) > time_limit*n_steps ) {printf("Time limit passed"); return -1;}

        bool action_OK = environment->Act(action);
//        reward = environment->getExpectedReward(state,action);
        reward = environment->getReward();
        total_reward += reward;
        printf("%d %d %d %f %d # s a r next_s\n", t, state, action, reward,environment->getState());
        if (!action_OK) {
            state = environment->getState();
            reward = environment->getReward();
            sampling->Act(reward, state);
            environment->Reset();
            sampling->Reset();
        }
    }
    return total_reward;
}

#endif
