#ifdef MAKE_MAIN

#include <chrono> 
#include <iostream>

#include "DiscreteMDPCountsSparse.h"
#include "ValueIteration.h"
#include "PolicyIteration.h"
#include "PolicyEvaluation.h"
#include "RTDP.h"

#include "Gridworld.h"
#include "DiscreteChain.h"


int main(int argc, char** argv) {

	int n_states = 25;
	int n_actions = 4;
	real gamma = 0.95;
	int n_steps = 10000;
	int n_sims = 10;

//	DiscreteEnvironment* environment = new DiscreteChain(5);
	DiscreteEnvironment* environment = new Gridworld("../../../dat/maze0",0.2,0,1.0,0);	//WOW: If I not initialize here by pointer, than getMDP in next-line calls the base-class method
	DiscreteMDP* model = environment->getMDP(); // see: https://stackoverflow.com/questions/1444025/c-overridden-method-not-getting-called (google: c++ function not overriding) for discussion
	if (model == NULL) printf("base-class getMDP calledn instead\n");


	PolicyIteration PI = PolicyIteration(model, gamma);
	auto start = std::chrono::high_resolution_clock::now();
	PI.ComputeStateValues(1e-1);
	auto stop = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start); 	
	std::cout << "Time taken by PI function: " << duration.count() << " microseconds" << std::endl;

	DiscreteMDP* model1 = environment->getMDP();
	RTDP rtdp = RTDP(model1,gamma,environment->getState());
	start = std::chrono::high_resolution_clock::now();
	rtdp.ComputeStateValues(1e-1);
	stop = std::chrono::high_resolution_clock::now();
	duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start); 	
	std::cout << "Time taken by RTDP function: " << duration.count() << " microseconds" << std::endl;


	FixedDiscretePolicy* PI_pol = PI.policy;
//	PI_pol->Show();
	FixedDiscretePolicy* rtdp_pol = rtdp.getPolicy();
//	rtdp_pol->Show();

	int n_exp = n_sims;
	do {
    environment->Reset();
    real reward = environment->getReward();
    real total_reward = 0;
    int state = environment->getState();
	PI_pol->Reset(state);
    int action = PI_pol->SelectAction();
    for (int t=0; t<n_steps; ++t) {
        bool action_OK = environment->Act(action);
        reward = environment->getExpectedReward(state,action);
        //reward = environment->getReward();
        total_reward += reward;
        printf("%d %d %f %d # s a r next_s\n", state, action, reward,environment->getState());
        if (!action_OK) {
            environment->Reset();
			state = environment->getState();
			PI_pol->Reset(state);
		}
		state = environment->getState();
		PI_pol->Observe(reward,state);
		action = PI_pol->SelectAction();
	}
	n_exp--;
	}while (n_exp);

	printf("ola\n");
	n_exp = n_sims;
	do {
    environment->Reset();
    real reward = environment->getReward();
    real total_reward = 0;
    int state = environment->getState();
	rtdp_pol->Reset(state);
    int action = rtdp_pol->SelectAction();
    for (int t=0; t<n_steps; ++t) {
        bool action_OK = environment->Act(action);
        reward = environment->getExpectedReward(state,action);
        //reward = environment->getReward();
        total_reward += reward;
        printf("%d %d %f %d # s a r next_s\n", state, action, reward,environment->getState());
        if (!action_OK) {
            environment->Reset();
			state = environment->getState();
			rtdp_pol->Reset(state);
		}
		state = environment->getState();
		rtdp_pol->Observe(reward,state);
		action = rtdp_pol->SelectAction();
	}
	n_exp--;
	}while (n_exp);

	return 0;
}
#endif
