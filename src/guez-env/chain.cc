#include "chain.h"
#include "utils.h"

Chain::Chain(double discount){


	NumObservations = S;
	NumActions = A;
    RewardRange = 1;
	Discount = discount;
	rsas = false;

	R = new double[SA];
	T = new double[SA*S];
	setMDP(R,T);
}

uint Chain::CreateStartState() const
{ 
		return 0; //(rand() % S);
}

bool Chain::Step(uint state, uint action, 
    uint& observation, double& reward) const
{
	utils::rng.multinom(T+state*SA+action*S,S,observation);
	if(rsas)	
		reward = R[state*SA+action*S+observation];
	else
		reward = R[state*A+action];

	//Never terminates
	return false;
}

 
void Chain::setMDP(double* R, double* T){

	for (uint s=0; s<SA*S; ++s) T[s] = 0.0;
	for (uint s=0; s<SA; ++s) R[s] = 0.0;


	//Filling MDP    
	double slip = 0.2;
	double start = 0.2;
	double end = 1.0;
	uint n_states = S;

    for (uint s=0; s<n_states; ++s) {
        uint s_n = s + 1;
        if (s_n > n_states - 1) {
            s_n = n_states - 1;
        }

        // Action 0
		T[encode(s, 0, 0)] =  1.0 - slip;
		T[encode(s, 0, s_n)] =  slip;

        // Action 1
        T[encode(s, 1, s_n)] = 1.0 - slip;
		T[encode(s, 1, 0)] =  slip;
		
		if (s==0) {
			R[encode(s, 0)] = start;
			R[encode(s, 1)] = 0;
		} else if (s == n_states - 1) {
			R[encode(s, 0)] = start;
			R[encode(s, 1)] = end;
		} else {
			R[encode(s, 0)] = start;
			R[encode(s, 1)] = 0;
		}
    }

}

