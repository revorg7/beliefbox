#ifdef MAKE_MAIN

#include <iostream>

#include "DiscreteMDPCountsSparse.h"
#include "fstream"
#include "MersenneTwister.h"
#include "RandomMDP.h"
#include "Gridworld.h"

int main(int argc, char** argv) {

	enum DiscreteMDPCountsSparse::RewardFamily reward_prior = DiscreteMDPCountsSparse::BETA;
	MDPModel* belief =  new DiscreteMDPCountsSparse(4, 1,2.0,reward_prior); //Note that Base class pointer is used, similar use case to production code
	belief->AddTransition(1,0,1,0);
	belief->ShowModelStatistics();
	//DiscreteMDP* mdp = belief.generate();
	//mdp->ShowModel();

    RandomNumberGenerator* rng;
    MersenneTwisterRNG mersenne_twister;
    rng = (RandomNumberGenerator*) &mersenne_twister;
    rng->manualSeed(324235536244234234);


//	real randomness = 1;
//  int n_states = 8;
//   RandomMDP generator = RandomMDP(n_states,1,randomness,0.0,0.0,1.0,rng);
//	const DiscreteMDP* mdp = generator.generateMDP();
//mdp->ShowModel();

//  Gridworld environment = Gridworld("../../../dat/maze001",0,0,1,0);
//  const DiscreteMDP* mdp = environment.getMDP();
 // n_states = environment.getNStates();
 // printf("no. of s is : %d\n",n_states );


	std::ofstream file_obj;
        file_obj.open("Input.txt", std::ios::trunc);
	DiscreteMDPCountsSparse* pd;
    pd = static_cast<DiscreteMDPCountsSparse*>(belief);//Before writing to file, downcasting (Base to Derived) is necessary. Dynamic or Static both should work
	file_obj.write((char*)pd, sizeof(*pd));

	file_obj.close();

	std::fstream is;
	is.open("Input.txt", std::ios::in);

	is.seekg (0, is.end);
	int length = is.tellg();
	is.seekg (0, is.beg);
printf("ola\n");
	DiscreteMDP* model;
	char * buffer = new char [length];
	is.read(buffer, length);
	is.close();
//	std::cout << std::string(buffer,length);
	MDPModel *obj = reinterpret_cast<MDPModel *>(buffer);
//printf("ns:%d\n",obj->getNStates());
obj->ShowModelStatistics();
}
#endif
