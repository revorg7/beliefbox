// -*- Mode: c++ -*-
// copyright (c) 2006 by Christos Dimitrakakis <christos.dimitrakakis@gmail.com>
// $Id: PPVI.c,v 1.5 2006/11/08 17:20:17 cdimitrakakis Exp cdimitrakakis $
/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ***************************************************************************/

#include "PPVI.h"
#include "real.h"
#include "MathFunctions.h"
#include "Vector.h"
#include <cmath>
#include <cassert>
//
//#define DEBUG1
//#define DEBUG2
//#define DEBUG_BELLMAN

PPVI::PPVI(const DiscreteMDP* mdp, real gamma, int state_decoder_const, int part_encoder_const, real baseline)
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
	PDP = (int *)calloc(total_no_of_parts*total_no_of_parts , sizeof(int));
	InitializePartitions();
}

void PPVI::InitializePartitions()
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

	//PDP
	for (int part=0; part<total_no_of_parts; part++) {
		for(auto const& s: partitions[part]) {
			for (int part2=0; part2<total_no_of_parts; part2++) {
				if (PDS[part2 + total_no_of_parts*s ]) PDP[part2 + total_no_of_parts*part] = 1;

			}
		}
	}

  //freeing
  free(PDS);
}

void PPVI::Reset()
{
    //int N = n_states * n_actions;

    V.Resize(n_states);
    dV.Resize(n_states);
    pV.Resize(n_states);
    Delta.Resize(n_states);
    H.Resize(n_states);
    B.Resize(n_states);
    HP.Resize(n_states);
    HPP.Resize(n_states*n_states);

    Q.Resize(n_states, n_actions);
    dQ.Resize(n_states, n_actions);
    pQ.Resize(n_states, n_actions);


    for (int s=0; s<n_states; s++) {
        H(s) = 0.0;
        V(s) = 0.0;
        dV(s) = 0.0;
        pV(s) = 0.0;
        Delta(s) = 0.0;
        for (int a=0; a<n_actions; a++) {
            Q(s, a) = 0.0;
            dQ(s, a) = 1.0;
            pQ(s, a) = 0.0;
        }
    }
}

PPVI::~PPVI()
{
if (mdp)
	delete mdp;
if (arr)
	free(arr);		// << Somehow this is freeing up automatically
if (SDP)
	free(SDP);
if (PDP)
  free(PDP);
}


/** Compute state values using asynchronous value iteration.

	The process ends either when the error is below the given threshold,
	or when the given number of max_iter iterations is reached. Setting
	max_iter to -1 means there is no limit to the number of iterations.

    This version updates the current values immediately
*/
void PPVI::ComputeStateValuesAsynchronous(int current_partition, real threshold, int max_iter)
{
    int n_iter = 0;
    do {
//        Delta = 0.0;
        for(auto const& s: partitions[current_partition]) {
  //        printf("P:%d s:%d ths:%f\n",current_partition,s,Delta);
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
  //          Delta += fabs(V(s) - pV(s)); ///< A bit stronger condition than Wingate paper (ICML, 2003), which tells only each V(s) to be epsilon converged.
            Delta(s) = fabs(V(s) - pV(s));
//            printf("part:%d s:%d v:%.8lf pv:%.8lf fabs:%.8lf Delta:%.8lf\n",current_partition,s,V(s), pV(s),fabs(V(s) - pV(s)),Delta(s));
            pV(s) = V(s);
        }

        if (max_iter > 0) {
            max_iter--;
        }
        n_iter++;
#ifdef DEBUG_LOOP
//for(auto const& s: partitions[current_partition]) printf("s:%d del:%f\n",s,Delta(s));
//printf("Delta-sum:%f Delta-max:%f\n\n",Sum(convert<real>(Delta)),Max(Delta));
#endif
    } while(Max(Delta) > threshold && max_iter != 0);
#ifdef DEBUG_LOOP
    printf("#PPVI::ComputeStateValues Exiting at part:%d Max(Delta):%.8lf, n:%d\n",current_partition, Max(Delta),n_iter);
#endif
}


/** Compute state values using value iteration.

	The process ends either when the error is below the given threshold,
	or when the given number of max_iter iterations is reached. Setting
	max_iter to -1 means there is no limit to the number of iterations.
*/
void PPVI::ComputeStateValuesStandard(real threshold, int max_iter)
{
  //
  int total_no_of_parts = part_decoder_const*part_decoder_const;

  //Initializing
  for (int s=0; s<n_states; s++) {
    real max = mdp->getExpectedReward(s,0);
    for (int a=1; a<n_actions; a++)
      if (mdp->getExpectedReward(s,a) > max) max = mdp->getExpectedReward(s,a);
    H(s) = max;
  }
  //
  for (int s=0; s<n_states; s++) B(s) = H(s);///< This is not exactly correct since B(s) should represent Potential Change, not Priority metric.
  //
  for (int part=0; part<total_no_of_parts; part++) {
//    printf(" part:%d total:%d\n",part,total_no_of_parts );
//    int index = partitions[part][0];
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
  int max_iter_condition = 1;

  //Looping
  while ( max_iter_condition && Max(HP) > threshold ) {
//  while ( (counter < max_iter && max_iter > 0) || Max(Delta)/(1-gamma) > threshold ) { ///< This one is also not coverging. See notes about it in Drafts email
#ifdef DEBUG_BELLMAN
  real previous = Sum(convert<real>(V));
  printf("\npart:%d sum-v-before:%f\n",current_part, Sum(convert<real>(V)) );
#endif
    ComputeStateValuesAsynchronous(current_part,threshold);///< partition threshold has to be same as overall threshold, refer ICMLA, 2003.
//    ComputeStateValuesElimination(current_part,threshold);///< Checked this one, works slower
#ifdef DEBUG_BELLMAN
if (Sum(convert<real>(V)) - previous < 1e-5) {  for (int s=0; s<n_states; s++) printf("s:%d V(s):%.8lf pV(s):%.8lf B(s):%f\n",s,V(s),pV(s),B(s));
printf("sum-v-after:%f Max(B):%f stopping-criteria:%f\n",Sum(convert<real>(V)),Max(B),Max(B)/(1-gamma)); }
#endif
    //Updating current_partition Priority, missing in pseudocode of most paper versions, just given in the one uploaded on semanticscholar website
    HP(current_part) = threshold;

    //Updating partition Priority
    /*
    for (int s=0; s<n_states; s++) {
      if (SDP[s+n_states*current_part]) {
        H(s) = Priority(s,threshold);
        int part = arr[s];
        int condition=0;
        if (HPP(current_part + total_no_of_parts*part) < H(s) ) {
          HPP(current_part + total_no_of_parts*part) = H(s);
          condition = 1;
        }
//        else h_max = HPP(current_part + total_no_of_parts*part);
//        HPP(current_part + total_no_of_parts*part) = h_max;
        if ((H(s) > HP(part)) && condition) HP(part) = H(s);
      }
    }
    */
    for (int part=0; part<total_no_of_parts; part++) {
      if (PDP[part + total_no_of_parts*current_part]) {
          HPP(current_part + total_no_of_parts*part) = 0;
          real h_max = 0;
          for(auto const& s: partitions[part])
            if (SDP[s+n_states*current_part]) {
              H(s) = Priority(s,threshold);
              if (h_max < H(s)) h_max = H(s);
            }
          HPP(current_part + total_no_of_parts*part) = h_max;
          if (h_max > HP(part)) HP(part) = h_max;///< This line is a bit different from the pseudo-code
      }
    }

    //
    current_part = ArgMax(HP);
#ifdef DEBUG_LOOP
for (int part=0; part<total_no_of_parts; part++) printf("part%d HP:%.8lf\n",HP(part));
#endif
#ifdef DEBUG_BELLMAN
    for (int part=0; part<total_no_of_parts; part++) printf("part%d HP:%.8lf\n",HP(part));
#endif
    counter += 1;
    if (max_iter > 0) if (counter >= max_iter) max_iter_condition = 0;
  }

}

real PPVI::Priority(int s, real threshold) {

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
      Qsa(a) = Q_sa;
  }
  real difference = Max(Qsa) - pV(s);
  B(s) = difference;
#ifdef DEBUG_BELLMAN
  printf("s%d pV:%.8lf diff:%.8lf\n",s,pV(s),difference);
  if (difference> 1e-4) for (int a=0; a<n_actions; a++) printf("a%d Qsa:%.8lf\n",a,Qsa(a));
#endif

  //H2 metric
//  if (B(s) > threshold)
//    return B(s) + V(s);
//  else
//    return 0;


  return B(s);
}


/// Create the greedy policy with respect to the calculated value function.
FixedDiscretePolicy* PPVI::getPolicy()
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
