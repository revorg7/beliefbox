/* -*- Mode: C++; -*- */
/* VER: $Id: Distribution.h,v 1.3 2006/11/06 15:48:53 cdimitrakakis Exp cdimitrakakis $*/
// copyright (c) 2006 by Christos Dimitrakakis <christos.dimitrakakis@gmail.com>
/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ***************************************************************************/

#ifdef MAKE_MAIN
#include "PolicyEvaluation.h"
#include "ValueIteration.h"
#include "RandomMDP.h"
#include "InventoryManagement.h"
#include "DiscretePolicy.h"
#include "Environment.h"
#include "ExplorationPolicy.h"
#include "Sarsa.h"

struct Statistics
{
    real total_reward;
    real discounted_reward;
    int steps;
};


std::vector<Statistics> EvaluateAlgorithm(int n_steps,
					  int n_episodes,
					  OnlineAlgorithm<int,int>* algorithm,
					  DiscreteEnvironment* environment,
					  real gamma);

int main (void)
{
    int n_actions = 4;
    int n_states = 4;
    real gamma = 0.99;
    real lambda = 0.9;
    real alpha = 0.1;
    real randomness = 0.0;
    real pit_value = -100.0;
    real goal_value = 10.0;
    real step_value = -0.1;
    real epsilon = 0.1;
    int n_runs = 100;
    int n_episodes = 100;
    int n_steps = 1000;

    std::cout << "Starting test program" << std::endl;
    
    std::cout << "Creating exploration policy" << std::endl;
    ExplorationPolicy* exploration_policy = NULL;
    exploration_policy = new EpsilonGreedy(n_actions, epsilon);
    
    
    std::cout << "Creating online algorithm" << std::endl;
    OnlineAlgorithm<int, int>* algorithm = NULL;
    algorithm = new Sarsa(n_states,
                          n_actions,
                          gamma,
                          lambda,
                          alpha,
                          exploration_policy);

    std::cout << "Creating environment" << std::endl;
    DiscreteEnvironment* environment = NULL;
    environment = new RandomMDP (n_actions,
                                 n_states,
                                 randomness,
                                 step_value,
                                 pit_value,
                                 goal_value);

    //const DiscreteMDP* mdp = environment->getMDP();
    //assert(n_states == mdp->GetNStates());
    //assert(n_actions == mdp->GetNActions());
    
    
    std::cout << "Starting evaluation" << std::endl;

    // remember to use n_runs

    std::vector<Statistics> statistics = EvaluateAlgorithm(n_steps, n_episodes, algorithm, environment, gamma);
    for (uint i=0; i<statistics.size(); ++i) {
	std::cout << statistics[i].total_reward << " "
		  << statistics[i].discounted_reward << "# REWARD"
		  << std::endl;
    }
    std::cout << "Done" << std::endl;

    delete environment;
    delete algorithm;
    delete exploration_policy;
    
    return 0;
}

std::vector<Statistics> EvaluateAlgorithm(int n_steps,
					  int n_episodes,
					  OnlineAlgorithm<int, int>* algorithm,
					  DiscreteEnvironment* environment,
					  real gamma)
{
    std:: cout << "Evaluating..." << std::endl;
 
    std::vector<Statistics> statistics(n_episodes);

    for (int episode = 0; episode < n_episodes; ++episode) {
	statistics[episode].total_reward = 0.0;
	statistics[episode].discounted_reward = 0.0;
	statistics[episode].steps = 0;
	real discount = 1.0;
	environment->Reset();
	for (int t=0; t < n_steps; ++t) {
	    int state = environment->getState();
	    real reward = environment->getReward();
	    std::cout << state << " " << reward << std::endl;
	    statistics[episode].total_reward += reward;
	    statistics[episode].discounted_reward += discount * reward;
	    discount *= gamma;
	    statistics[episode].steps = t;
	    int action = algorithm->Act(reward, state);
	    bool action_ok = environment->Act(action);
	    if (!action_ok) {
		break;
	    }
	}

    }
    return statistics;
}

#endif
