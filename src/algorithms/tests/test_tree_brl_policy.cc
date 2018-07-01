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


/// Bayesian RL includes
#include "DiscreteMDPCounts.h"

/// The main thing to test
#include "TreeBRLPolicy.h"
//for comparision
#include "MDPModel.h"
#include "SampleBasedRL.h"

/// The basic environments
//#include "ContextBandit.h"
#include "DiscreteChain.h"
#include "DoubleLoop.h"
//#include "Blackjack.h"
//#include "InventoryManagement.h"
//#include "OptimisticTask.h"
#include "Gridworld.h"
#include "RandomMDP.h"

/// STD
#include <iostream>
#include <memory>
#include <algorithm> //for shuffle
using namespace std;

real RunExperiment(shared_ptr<DiscreteEnvironment> environment,
                   TreeBRLPolicy& tree, 
                   int n_steps, SampleBasedRL* sampling,unordered_map<int,int> rotator);
//real RunExperiment(shared_ptr<DiscreteEnvironment> environment,
//                   TreeBRL& tree, 
//                   int n_steps, unordered_map<int,int> rotator);

int main(int argc, char** argv) {
    // use a high-quality RNG for the main program
    RandomNumberGenerator* rng;
    MersenneTwisterRNG mersenne_twister;
    rng = (RandomNumberGenerator*) &mersenne_twister;

    // use some arbitrary sequence (e.g. a fixed file) for generating environments, to ensure consistency across runs
    DefaultRandomNumberGenerator default_rng;
    RandomNumberGenerator* env_rng = (RandomNumberGenerator*) &default_rng;
    rng->seed();
    env_rng->manualSeed(982374523);
    int n_states = 5;
    int n_actions = 2;
    int n_policies = 3;
    real discounting = 0.99;
    int n_steps = 1000;

    // To remove any indexing bias
    std::vector<int> action_list;
    for (int i=0;i<n_actions;++i) action_list.push_back(i);
    std::random_shuffle(action_list.begin(),action_list.end());
    std::unordered_map<int, int> randomizer;
    for (int i=0;i<n_actions;++i) randomizer[i]=action_list[i];
    //for (int i=0;i<n_actions;++i) printf("action %d is %d\n",i,randomizer[i]);
    ///


    //    int n_samples = 2; ///< number of state samples when branching
    //int n_mdp_samples = 2; ///< number of MDP samples at leaf nodes

    // ---- user options ---- //
    int planning_horizon = 2; 
    int leaf_value = TreeBRLPolicy::LeafNodeValue::NONE;
    int algorithm = TreeBRLPolicy::WhichAlgo::PLC;
    int n_experiments = 5;

	if (argc > 1) {
		planning_horizon = atoi(argv[1]);
	}
	if (argc > 2) {
		leaf_value = atoi(argv[2]);
	}
	if (argc > 3) {
		n_experiments = atoi(argv[3]);
	}
	if (argc > 4) {
		n_steps = atoi(argv[4]);
	}
	if (argc > 5) {
		discounting = atof(argv[5]);
	}
	
    //printf("# Making environment\n");
    shared_ptr<DiscreteEnvironment> environment;
    environment = make_shared<DiscreteChain>(n_states);
    //environment = make_shared<DoubleLoop>();
    //environment = make_shared<OptimisticTask>(0.1,0.7); //2nd argument is success probablity of transition
    //environment = make_shared<Gridworld>("../../../dat/maze01");

    
    //environment = make_shared<ContextBandit>(n_states, n_actions, env_rng, false);
    //environment = make_shared<Blackjack>(env_rng);
    //environment = make_shared<RandomMDP>(n_states, n_actions, 0.1, -0.1, -1, 1, env_rng);
	// environment = make_shared<InventoryManagement>(10, 5, 0.2, 0.1);-=  n_states = environment->getNStates();
    n_actions = environment->getNActions();
    n_states = environment->getNStates();
#if 0
    //simplify things by fixing the rewards
    printf("# Setting up belief\n");
    Matrix rewards(n_states, n_actions);
    for (int s=0; s<n_states; ++s) {
        for (int a=0; a<n_actions; ++a) {
            rewards(s,a) = environment->getExpectedReward(s, a);
        }
    }
    belief.setFixedRewards(rewards);
#endif

//    printf("# full sampling\n");


	
    real dirichlet_mass = 0.5;
    enum DiscreteMDPCounts::RewardFamily reward_prior = DiscreteMDPCounts::BETA;
    DiscreteMDPCounts belief(n_states, n_actions, dirichlet_mass, reward_prior);
	//NullMDPModel belief(n_states, n_actions);
	

	// Adding for comparision to USAMPLING
            DiscreteMDPCounts* discrete_mdp =  new DiscreteMDPCounts(n_states, n_actions,
                                                  dirichlet_mass,
                                                  reward_prior);
            MDPModel* model= (MDPModel*) discrete_mdp;

	    int max_samples = 2;
	    real epsilon = 0.01;
            SampleBasedRL* sampling = new SampleBasedRL(n_states, ///<this leaks memory too
                                          n_actions,
                                          discounting,
                                          epsilon,
                                          model,
                                          rng,
                                          max_samples,
                                          true);
          //  if (use_sampling_threshold) {
          //      sampling->setSamplingThreshold(sampling_threshold);
          //  }
	////

    Vector U(n_experiments);
    for (int experiment=0;
         experiment<n_experiments;
         experiment++) {
        
        TreeBRLPolicy tree (n_states, n_actions, discounting, &belief, rng, planning_horizon, (TreeBRLPolicy::LeafNodeValue) leaf_value,(TreeBRLPolicy::WhichAlgo) algorithm);
        // Set state to 0

        real total_reward = RunExperiment(environment, tree, n_steps,sampling,randomizer); ///<This real leaks memory
        printf("H:%d,\tV:%d,\tR:%f\n", planning_horizon, leaf_value,total_reward);
        U(experiment) = total_reward;
    }
    printf("L:%f,\tM:%f,\tU:%f\n",
           Min(U),
           U.Sum() / (real) U.Size(),
           Max(U));

    return 0;
}

real RunExperiment(shared_ptr<DiscreteEnvironment> environment,
                   TreeBRLPolicy& tree, 
                   int n_steps,SampleBasedRL* sampling,unordered_map<int,int> rotater )
//real RunExperiment(shared_ptr<DiscreteEnvironment> environment,
//                   TreeBRL& tree, int n_steps,
//                   unordered_map<int,int> rotater )
{
    environment->Reset();
    tree.Reset(environment->getState());
    real reward = environment->getReward();
    real total_reward = 0;
    for (int t=0; t<n_steps; ++t) {
        int state = environment->getState();
        int action = tree.Act(reward, state);

	//int action = sampling->Act(reward,state);
	//action = rotater[action];	//rotater should be implemented after env calls

        bool action_OK = environment->Act(action);
        reward = environment->getExpectedReward(state,action);
        //reward = environment->getReward();
        total_reward += reward;
        printf("%d %d %f %d # s a r next_s\n", state, action, reward,environment->getState());
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

#endif
