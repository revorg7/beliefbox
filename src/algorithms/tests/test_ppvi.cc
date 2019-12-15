#ifdef MAKE_MAIN

#include <chrono>
#include <iostream>

#include "PPVI.h"
#include "PPI.h"
#include "ValueIteration.h"
#include "PolicyIteration.h"
#include "DiscreteMDPCountsSparse.h"

#include "MersenneTwister.h"
#include "RandomMDP.h"
#include "Gridworld.h"

int main(int argc, char** argv) {

//	enum DiscreteMDPCountsSparse::RewardFamily reward_prior = DiscreteMDPCountsSparse::BETA;
//	DiscreteMDPCountsSparse belief =  DiscreteMDPCountsSparse(400, 3,2.0,reward_prior);
//	DiscreteMDP* mdp = belief.generate();
//	mdp->ShowModel();

    RandomNumberGenerator* rng;
    MersenneTwisterRNG mersenne_twister;
    rng = (RandomNumberGenerator*) &mersenne_twister;
    rng->manualSeed(324235536244234234);


	real randomness = 1;
  int n_states = 400;
//    RandomMDP generator = RandomMDP(n_states,2,randomness,0.0,0.0,1.0,rng);
//	const DiscreteMDP* mdp = generator.generateMDP();
//mdp->ShowModel();

  Gridworld environment = Gridworld("../../../dat/maze001",0,0,1,0);
  const DiscreteMDP* mdp = environment.getMDP();
//  n_states = environment.getNStates();
//  printf("no. of s is : %d\n",n_states );


  //ppvi parameters
	real gamma = 0.95;
  int state_decoder_const = 100;
	int part_encoder_const = 20;
  real threshold = 1e-6;


  //PPVI
  auto start = std::chrono::high_resolution_clock::now();
	PPI ppvi = PPI(mdp, gamma, state_decoder_const, part_encoder_const);
  ppvi.ComputeStateValues(threshold);
  auto finish = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(finish - start);
  printf("\nPPVI time-taken:%d\n",duration);
//  printf("PPVI:\n");
//  for (int s=0; s<n_states; s++) printf("s:%d V:%f\n",s,ppvi.V(s));
  //PI
  start = std::chrono::high_resolution_clock::now();
  PolicyIteration pi = PolicyIteration(mdp,gamma);
  pi.ComputeStateValues(10,threshold);
  finish = std::chrono::high_resolution_clock::now();
  duration = std::chrono::duration_cast<std::chrono::microseconds>(finish - start);
  printf("PI time-taken:%d\n",duration);

  //VI
  threshold = 1e-6;
  start = std::chrono::high_resolution_clock::now();
  ValueIteration vi = ValueIteration(mdp,gamma);
  vi.ComputeStateValues(threshold);
  finish = std::chrono::high_resolution_clock::now();
  duration = std::chrono::duration_cast<std::chrono::microseconds>(finish - start);
  printf("VI time-taken:%d\n",duration);
//  printf("VI:\n");
//  for (int s=0; s<n_states; s++) printf("s:%d V:%f\n",s,vi.V(s));

  //Comparing V(s)
  int condition = 1;
  real diff=0.0;
  for (int s=0; s<n_states; s++) if (fabs(vi.V(s) - ppvi.V(s)) > 1e-4) condition = 0; else diff += fabs(vi.V(s) - ppvi.V(s));
  if (condition) printf("V(s) of PPVI and VI match.\n");
  else printf("PPVI-VI mismatch:%f\n",diff);

  condition = 1;
  diff=0.0;
  for (int s=0; s<n_states; s++) if (fabs(vi.V(s) - ppvi.V(s)) > 1e-4) condition = 0; else diff += fabs(vi.V(s) - pi.getValue(s));
  if (condition) printf("V(s) of PI and VI match.\n");
  else printf("PI-VI mismatch:%f\n",diff);

  //Average V(s)
  real avg = 0.0;
  for (int s=0; s<n_states; s++) avg+=vi.V(s);
  printf("\nN_states:%d Avg V(s):%f\n",n_states,avg/n_states);

  //Policy mismatch
  FixedDiscretePolicy* vipol = vi.getPolicy();
  FixedDiscretePolicy* ppvipol = ppvi.getPolicy();
  int different_actions=0;
  diff = 0.0;
  for (int s=0; s<n_states; s++) if (ArgMax( vipol->getActionProbabilities(s) ) != ArgMax( ppvipol->getActionProbabilities(s) )) {
    different_actions+=1;
    diff+= fabs(vi.V(s) - ppvi.V(s));
    int ppvia = ArgMax( ppvipol->getActionProbabilities(s) );
    int via = ArgMax( vipol->getActionProbabilities(s));
    if (vi.V(s) != vi.Q(s,ppvia)) {
      printf("Now this is wrong\n");
//    printf("s:%d ppvi-a:%d vi-a:%d\n",s,ArgMax( ppvipol->getActionProbabilities(s) ),ArgMax( vipol->getActionProbabilities(s) ) );
    printf("s:%d ppvi-a:%d vi-a:%d v(ppvia):%f v(via):%f\n",s,ppvia,via, vi.Q(s,ppvia), vi.V(s) );
//    Vector ppvi_q = ppvi.Q.getRow(s);
//    Vector pi_q = vi.Q.getRow(s);
//    for (int a=0; a<4; a++) printf("a:%d ppvi:%f vi:%f\n",a,ppvi_q(a),pi_q(a));
    }
  }
  printf("No of different actions for PPVI:%d and total diff for them is:%f\n",different_actions,diff);

  different_actions=0;
  for (int s=0; s<n_states; s++) if (ArgMax( vipol->getActionProbabilities(s) ) != ArgMax( pi.policy->getActionProbabilities(s) )) {
    different_actions+=1;
  }
  printf("No of different actions for PI:%d\n",different_actions);

//  printf("Need to put the Mt/(1-gamma) convergence rate criteria, which is not same as Max(HP)/(1-gamma), need to keep track of B(s) for each-state to do that\n");
  printf("The error was not due to above reason, the error was due to the fact that algorithm assumes +ve rewards, make sure of that, everything else was due to ArgMax() output difference\n" );

}
#endif
