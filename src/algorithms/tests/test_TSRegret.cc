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
#include "DiscreteMDPCounts.h"

/// The main thing to test
#include "ValueIteration.h"
//for comparision
#include "MDPModel.h"
#include "SampleBasedRL.h"

/// The basic environments
#include "RiverSwim.h"
#include "DiscreteChain.h"
#include "RandomMDP.h"

/// Other things
#include "MersenneTwister.h"

/// STD
#include <iostream>
#include <memory>
using namespace std;



real RunExperiment(shared_ptr<DiscreteEnvironment> environment,
                   int n_steps, SampleBasedRL* sampling,int experiment);
real RunExperiment(shared_ptr<DiscreteEnvironment> environment, 
                   int n_steps);


//Global variables
real *arr;
int n_steps;

int main(int argc, char** argv) {


    // RNG
    RandomNumberGenerator* rng;
    MersenneTwisterRNG mersenne_twister;
    rng = (RandomNumberGenerator*) &mersenne_twister;
    rng->manualSeed(23);
	////

    int n_states = 5;
    int n_actions = 2;
    real discounting = 0.99;
    n_steps = 100000000;
    int n_experiments = 20;


    // ---- user options ---- //
	real dirichlet_mass = 2.0;
	bool useFixedRewards = false; // << Using Fixed Rewards
	int max_samples = 1;	// << For USampling
	real epsilon = 0.01;	// << For USampling

	if (argc > 1) {
		n_steps = atoi(argv[1]);
	}
	if (argc > 2) {
		n_experiments = atoi(argv[2]);
	}
	if (argc > 3) {
		useFixedRewards = atoi(argv[3]);
	}
	
	arr = (real *)calloc(n_experiments*n_steps , sizeof(real));

    //printf("# Making environment\n");
    shared_ptr<DiscreteEnvironment> environment;

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


		DiscreteMDPCounts* model = new DiscreteMDPCounts(n_states,n_actions,dirichlet_mass, DiscreteMDPCounts::BETA);
if(useFixedRewards) model->setFixedRewards(rewards);

            SampleBasedRL* sampling = new SampleBasedRL(n_states, ///<this leaks memory too
                                          n_actions,
                                          discounting,
                                          epsilon,
                                          model,
                                          rng,
                                          max_samples,
                                          true);
        real total_reward = RunExperiment(environment,n_steps,sampling,experiment); ///<This real leaks memory
        U(experiment) = total_reward;

	}

	real *avg_arr;
	avg_arr = (real *)calloc(n_steps , sizeof(real));
	for (int t=0; t<n_steps; ++t) {
		float sum = 0;
		for (int experiment=0; experiment<n_experiments;experiment++) {
			sum += arr[experiment*n_steps+t];
		}
		avg_arr[t] = sum/n_experiments;
	}
    for (int t=0; t<n_steps; ++t) {
	printf("t:%d r:%f\n",t,avg_arr[t]);
	}

/*
    printf("L:%f,\tM:%f,\tU:%f T:%f\n",
           Min(U),
           U.Sum() / (real) U.Size(),
           Max(U),
			difftime(time(NULL),past_time));
*/
    float avg_reward = RunExperiment(environment, n_steps);
	printf("Average-reward for Optimal policy is %f\n",avg_reward);
	//Freeing Global
	if (arr) free(arr);
	if (avg_arr) free(avg_arr);
	return 0;

}

real RunExperiment(shared_ptr<DiscreteEnvironment> environment,
                   int n_steps)
{
	int n_iters = 1000;
    environment->Reset();
    int n_actions = environment->getNActions();
    int n_states = environment->getNStates();

	auto mdp = environment->getMDP();
	ValueIteration PI(mdp, 0.99); // << interestingly somehow PolicyIteration is giving wrong answer for Chain all the way, need to check that !!
	PI.ComputeStateValues(1e-6);
	FixedDiscretePolicy optimalPolicy = FixedDiscretePolicy(n_states,n_actions,PI.getPolicy()->p);
//optimalPolicy.Show();


	//Getting transition probability
	Matrix mat(n_states,n_states);
    for (int s=0; s<n_states; ++s) {
    	for (int s_n=0; s_n<n_states; ++s_n) {
			mat(s,s_n) = mdp->getTransitionProbability(s, ArgMax( optimalPolicy.getActionProbabilities(s) ) , s_n); 
		}
	}	

	//Calculating steady-state
	Matrix mator(n_states,n_states);
	mator = mat*mat;
    for (int iter=0; iter<n_iters; ++iter) {
		mator = mator*mat;	
	}

	real avg_reward = 0.0f;
    for (int s=0; s<n_states; ++s) {
		avg_reward += mator(0,s)*mdp->getExpectedReward(s,ArgMax( optimalPolicy.getActionProbabilities(s) ));
//printf("s:%d mat:%f v:%f\n",s,mator(0,s),PI.getValue(s));
	}

    for (int t=0; t<n_steps; ++t) {
//        printf("%d %d %d %f %d # s a r next_s\n", t, 0, 0, avg_reward,0);
    }

    return avg_reward;
}


real RunExperiment(shared_ptr<DiscreteEnvironment> environment,
                   int n_steps,SampleBasedRL* sampling,int experiment)
{

    environment->Reset();
    real reward = environment->getReward();
    real total_reward = 0;
    for (int t=0; t<n_steps; ++t) {
        int state = environment->getState();
		int action = sampling->Act(reward,state);

        bool action_OK = environment->Act(action);
//        reward = environment->getExpectedReward(state,action);
        reward = environment->getReward();
        total_reward += reward;
		arr[experiment*n_steps+t] = reward;
//        printf("%d %d %d %f %d # s a r next_s\n",t, state, action, reward,environment->getState());
        if (!action_OK) {
            state = environment->getState();
            reward = environment->getReward();
            environment->Reset();
        }

    }
    return total_reward;
}


#endif
