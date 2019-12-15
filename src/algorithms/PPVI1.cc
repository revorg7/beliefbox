// -*- Mode: c++ -*-
// copyright (c) 2006 by Christos Dimitrakakis <christos.dimitrakakis@gmail.com>
// $Id: PPVI1.c,v 1.5 2006/11/08 17:20:17 cdimitrakakis Exp cdimitrakakis $
/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ***************************************************************************/

#include "PPVI1.h"
#include "real.h"
#include "MathFunctions.h"
#include "Vector.h"
#include <cmath>
#include <cassert>
//
//#define DEBUG1
//#define DEBUG2

PPVI1::PPVI1(const DiscreteMDP* mdp, real gamma, int state_decoder_const, int part_encoder_const, real baseline)
{
    assert (mdp);
    assert (gamma>=0 && gamma <=1);
    this->mdp = mdp;
    this->gamma = gamma;
    this->baseline = baseline;
    n_actions = mdp->getNActions();
    n_states = mdp->getNStates();
    Reset();

	//Specific to PPVI
	this->state_decoder_const = state_decoder_const;
	this->part_encoder_const = part_encoder_const;
//	this->part_decoder_const = int ( sqrt(n_states) / part_encoder_const) ;
  this->part_decoder_const = int (state_decoder_const / part_encoder_const) ;
	arr = (int *)calloc(n_states , sizeof(int));
	int total_no_of_parts = part_decoder_const*part_decoder_const;
	SDP = (int *)calloc(total_no_of_parts*n_states , sizeof(int));
	InitializePartitions();
}

void PPVI1::InitializePartitions()
{
	for (int s=0; s<n_states; s++) {
		int x1 = s%state_decoder_const;
		int x2 = int((s - x1)/state_decoder_const);
		int p1 = int(x1/part_encoder_const);
		int p2 = int(x2/part_encoder_const);
		int partition_number = p1 + part_decoder_const*p2;
		arr[s] = partition_number;
		auto got = partitions.find(partition_number);
		if (got == partitions.end() ){
			std::vector<int> vect;
			vect.push_back(s);
			partitions[partition_number] = vect;
		} else {
			partitions[partition_number].push_back(s);
		}
	}

  //NOte, SDS is completely different from mdp->getNextStates(), its actually the inverse mapping of getNextStates()
  std::unordered_map<int, std::vector<int>> SDS;
  for (int s=0; s<n_states; s++) {
    for (int a=0; a<n_actions; a++) {
      const DiscreteStateSet& next = mdp->getNextStates(s, a);
	    for (DiscreteStateSet::iterator i=next.begin();i!=next.end();++i) {
	        	int s2 = *i;
            auto got = SDS.find(s2);
            if (got == SDS.end() ){
              std::vector<int> vect;
              vect.push_back(s);
              SDS[s2] = vect;
            } else {
              SDS[s2].push_back(s);
            }
			}
		}
  }

	//
	int total_no_of_parts = part_decoder_const*part_decoder_const;

	//SDP
	for (int part=0; part<total_no_of_parts; part++) {
		for(auto const& s: partitions[part]) {
      for(auto const& s2: SDS[s]) {
      		SDP[s2 + n_states*part] = 1; ///< Row-wise PxS
			}
		}
	}

	//PDS
	int *PDS = (int *)calloc(n_states*total_no_of_parts, sizeof(int));
	for (int s=0; s<n_states; s++) {
	  for(auto const& s2: SDS[s]) {
	    PDS[arr[s2] + total_no_of_parts*s] = 1;	///< Row-wise SxP
		}
	}

  //freeing
  free(PDS);
}

void PPVI1::Reset()
{
    //int N = n_states * n_actions;

    V.Resize(n_states);
    pV.Resize(n_states);
    H.Resize(n_states);
    B.Resize(n_states);
    HP.Resize(n_states);
    HPP.Resize(n_states*n_states);

    Q.Resize(n_states, n_actions);


    for (int s=0; s<n_states; s++) {
        H(s) = 0.0;
        V(s) = 0.0;
        pV(s) = 0.0;
        for (int a=0; a<n_actions; a++) {
            Q(s, a) = 0.0;
        }
    }
}

PPVI1::~PPVI1()
{
if (mdp)
	delete mdp;
if (arr)
	free(arr);		// << Somehow this is freeing up automatically
if (SDP)
	free(SDP);
}


/** Compute state values using asynchronous value iteration.

	The process ends either when the error is below the given threshold,
	or when the given number of max_iter iterations is reached. Setting
	max_iter to -1 means there is no limit to the number of iterations.

    This version updates the current values immediately
*/
void PPVI1::ComputeStateValuesAsynchronous(int current_partition, real threshold, int max_iter)
{
    int n_iter = 0;
    do {
        Delta = 0.0;
        for(auto const& s: partitions[current_partition]) {
            for (int a=0; a<n_actions; a++) {
                real Q_sa = 0.0;
                const DiscreteStateSet& next = mdp->getNextStates(s, a);
                for (DiscreteStateSet::iterator i=next.begin();
                     i!=next.end();
                     ++i) {
                    int s2 = *i;
                    real P = mdp->getTransitionProbability(s, a, s2);
                    real R = mdp->getExpectedReward(s, a) - baseline;
                    Q_sa += P * (R + gamma * V(s2));
                }
                Q(s, a) = Q_sa;
            }
            V(s) = Max(Q.getRow(s));
            Delta += fabs(V(s) - pV(s)); ///< A bit stronger condition than Wingate paper (ICML, 2003), which tells only each V(s) to be epsilon converged.
            pV(s) = V(s);
        }

        if (max_iter > 0) {
            max_iter--;
        }
        n_iter++;
    } while(Delta >= threshold && max_iter != 0);
//    printf("#PPVI1::ComputeStateValues Exiting at d:%f, n:%d\n", Delta, n_iter);
}


/** Compute state values using value iteration.

	The process ends either when the error is below the given threshold,
	or when the given number of max_iter iterations is reached. Setting
	max_iter to -1 means there is no limit to the number of iterations.
*/
void PPVI1::ComputeStateValuesStandard(real threshold, int max_iter)
{
  //
  int total_no_of_parts = part_decoder_const*part_decoder_const;

  //INITIALIZING
  for (int s=0; s<n_states; s++) {
    real max = mdp->getExpectedReward(s,0);
    for (int a=1; a<n_actions; a++)
      if (mdp->getExpectedReward(s,a) > max) max = mdp->getExpectedReward(s,a);
    H(s) = max;
  }
  //
  for (int s=0; s<n_states; s++) B(s) = H(s);
  //
  for (int part=0; part<total_no_of_parts; part++) {
    real max = H(partitions[part][0]);
    for(auto const& s: partitions[part])
      if (H(s) > max)  max = H(s);
    HP(part) = max;
  }
  for (int part1=0; part1<total_no_of_parts; part1++)
    for (int part2=0; part2<total_no_of_parts; part2++)
      HPP(part2 + total_no_of_parts*part1) = 0.0; //Filling row-wise
  //
  int current_part = ArgMax(HP);
  int counter = 0;
  //
  std::unordered_map<int, std::vector<int>> SDP_vect;
  for (int part=0; part<total_no_of_parts; part++) for (int s=0; s<n_states; s++) if (SDP[s+n_states*part]) SDP_vect[part].push_back(s);


  //LOOPING
//  while ( (counter < max_iter && max_iter > 0) || Max(HP) > threshold ) {
  while ( (counter < max_iter && max_iter > 0) || Max(B)/(1-gamma) > threshold ) {
    ComputeStateValuesAsynchronous(current_part,threshold);///< partition threshold has to be same as overall threshold, refer ICMLA, 2003.
    printf("Sum(V):%f\n",Sum(convert<real>(V)) );
    //Updating current_partition Priority, missing in pseudocode of most paper versions, just given in the one uploaded on semanticscholar website
    HP(current_part) = threshold;

    //Updating partition Priority
    for(auto const& s: SDP_vect[current_part]) {
        H(s) = Priority(s,threshold);
        int part = arr[s];
        int condition=0;
        if (HPP(current_part + total_no_of_parts*part) < H(s) ) {
          HPP(current_part + total_no_of_parts*part) = H(s);
          condition = 1;
        }
        if ((H(s) > HP(part)) && condition) HP(part) = H(s);
    }
    //
    current_part = ArgMax(HP);
    counter += 1;
  }

}

real PPVI1::Priority(int s, real threshold) {

  Vector Qsa;
  Qsa.Resize(n_actions);
  for (int a=0; a<n_actions; a++) {
      real Q_sa = 0.0;
      const DiscreteStateSet& next = mdp->getNextStates(s, a);
      for (DiscreteStateSet::iterator i=next.begin(); i!=next.end();++i) {
          int s2 = *i;
          real P = mdp->getTransitionProbability(s, a, s2);
          real R = mdp->getExpectedReward(s, a) - baseline;
          Q_sa += P * (R + gamma * pV(s2));
      }
      Qsa[a] = Q_sa;
  }
  B(s) = Max(Qsa) - pV(s);

/*
  //H2 metric
  if (B(s) > threshold)
    return B(s) + V(s);
  else
    return 0;
*/

  return B(s);
}


/// Create the greedy policy with respect to the calculated value function.
FixedDiscretePolicy* PPVI1::getPolicy()
{
#if 0
  FixedDiscretePolicy* policy = new FixedDiscretePolicy(n_states, n_actions);
    for (int s=0; s<n_states; s++) {
        int argmax_Qa = ArgMax(Q.getRow(s));
        Vector* p = policy->getActionProbabilitiesPtr(s);
        for (int a=0; a<n_actions; a++) {
            (*p)(a) = 0.0;
        }
        (*p)(argmax_Qa) = 1.0;
    }
    return policy;
#else
    return new FixedDiscretePolicy(n_states, n_actions, Q);
#endif
}
