// -*- Mode: c++ -*-
// copyright (c) 2006 by Christos Dimitrakakis <christos.dimitrakakis@gmail.com>
// $Id: PPI.c,v 1.5 2006/11/08 17:20:17 cdimitrakakis Exp cdimitrakakis $
/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ***************************************************************************/

#include "PPI.h"
#include "real.h"
#include "MathFunctions.h"
#include "Vector.h"
#include <cmath>
#include <cassert>
//
//#define DEBUG_PPI
//#define DEBUG_CONVERGENCE

PPI::PPI(const DiscreteMDP* mdp, real gamma, int state_decoder_const, int part_encoder_const, real baseline)
{
    assert (mdp);
    assert (gamma>=0 && gamma <=1);
    this->mdp = mdp;
    this->gamma = gamma;
    this->baseline = baseline;
    n_actions = mdp->getNActions();
    n_states = mdp->getNStates();

    policy = new FixedDiscretePolicy(n_states, n_actions);
    a_max.resize(n_states);
    Reset();

	//Specific to PPI
	this->state_decoder_const = state_decoder_const;
	this->part_encoder_const = part_encoder_const;
//	this->part_decoder_const = int ( sqrt(n_states) / part_encoder_const) ;
  this->part_decoder_const = int (state_decoder_const / part_encoder_const) ;
	arr = (int *)calloc(n_states , sizeof(int));
	int total_no_of_parts = part_decoder_const*part_decoder_const;
  stability = (int *)calloc(total_no_of_parts , sizeof(int));
	SDP = (int *)calloc(total_no_of_parts*n_states , sizeof(int));
	PDP = (int *)calloc(total_no_of_parts*total_no_of_parts , sizeof(int));
	InitializePartitions();
}

void PPI::InitializePartitions()
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

  //For Convergence
  for (int part=0; part<total_no_of_parts; part++) stability[part] = total_no_of_parts;

  //freeing
  free(PDS);
}

void PPI::Reset()
{
    policy->Reset();
    for (int s=0; s<n_states; s++) {
      a_max[s] = ArgMax(policy->getActionProbabilitiesPtr(s));
 //     Vector* p = policy->getActionProbabilitiesPtr(s);
  //    for (int a=0;a<n_actions;a++) (*p)[a] = 0.0;
    //  (*p)[a_max[s]] = 1.0;
    }
    V.Resize(n_states);
    H.Resize(n_states);
    B.Resize(n_states);
    HP.Resize(n_states);
    HPP.Resize(n_states*n_states);

    for (int s=0; s<n_states; s++) {
        H(s) = 0.0;
//        Delta(s) = 0.0;
    }
}

PPI::~PPI()
{
delete policy;
if (mdp)
	delete mdp;
if (arr)
	free(arr);		// << Somehow this is freeing up automatically
//if (stability)
  free(stability);
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
void PPI::ComputeStateValuesAsynchronous(int current_partition, int max_iter, real threshold)
{
    bool policy_stable = true;
    int iter = 0;
    do {
//        Delta = 0.0;///< Delta is manage by Evaluate function
        // evaluate policy
//printf("begin\n");
        EvaluateStateValues(current_partition,max_iter,threshold);
//printf("end\n");
        //Improving Policy
        for(auto const& s: partitions[current_partition]) {
          #ifdef DEBUG_PPI
          printf("P:%d s:%d ths:%f Delta:%f a-max:%d\n",current_partition,s,threshold,Priority(s,1e-3),a_max[s]);
          #endif
            real max_Qa = getValue(s, 0);
            int argmax_Qa = 0;
            for (int a=1; a<n_actions; a++) {
              real Qa = getValue(s, a);
              if (Qa > max_Qa) {
                max_Qa = Qa;
                argmax_Qa = a;
              }
            }
            //Updating probability vector for policy
            Vector* p = policy->getActionProbabilitiesPtr(s);
            for (int a=0; a<n_actions; a++) {
                (*p)[a] = 0.0;
            }
	    (*p)[argmax_Qa] = 1.0;
            //Checking convergence condition
            if (a_max[s] != argmax_Qa) {
                policy_stable = false;
                #ifdef DEBUG_CONVERGENCE
                if (stability[current_partition] > 1 && stability[current_partition]!=25) printf("part:%d state:%d a_prev:%d a_new:%d\n",current_partition,s,a_max[s],argmax_Qa);
                #endif
                a_max[s] = argmax_Qa;
            }
        }

        if (max_iter >= 0) {
            iter++;
        }
    } while(policy_stable == false && iter < max_iter);
    stability[current_partition] = iter; if (current_partition>=part_decoder_const*part_decoder_const) printf("\nMemory overwrite current_partition:%d\n",current_partition);
    #ifdef DEBUG_CONVERGENCE
    printf("\np:%d iter:%d stab:%d and Policy is\n",current_partition,iter,stability[current_partition]);
//    for(auto const& s: partitions[current_partition]) printf("s:%d a:%d ",s,a_max[s]);
    #endif
}


/** Compute state values using value iteration.

	The process ends either when the error is below the given threshold,
	or when the given number of max_iter iterations is reached. Setting
	max_iter to -1 means there is no limit to the number of iterations.
*/
void PPI::ComputeStateValuesStandard(real threshold, int max_iter)
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
  int arr_sum_condition = 1;
  //Looping
//  while ( max_iter_condition && Max(HP) > threshold ) {
  while ( max_iter_condition && arr_sum_condition && Max(HP) > threshold ) {

    ComputeStateValuesAsynchronous(current_part,10,threshold);///< partition threshold has to be same as overall threshold, refer ICMLA, 2003.
//printf("sum-V:%f counter:%d\n",Sum(convert<real>(V)),counter);
    //Updating current_partition Priority, missing in pseudocode of most paper versions, just given in the one uploaded on semanticscholar website
    HP(current_part) = Delta;

    //Updating partition Priority
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
    counter += 1;
    if (max_iter > 0) if (counter >= max_iter) max_iter_condition = 0;
    int summation=0;
    for (int part=0; part<total_no_of_parts; part++) {
      summation+=stability[part];
      #ifdef DEBUG_CONVERGENCE
      printf("part:%d stability:%d HP:%f\n",part,stability[part],HP(part));
      #endif
    }
    if (summation<=total_no_of_parts) arr_sum_condition = 0;
    #ifdef DEBUG_PPI
    for (int p=0;p<total_no_of_parts;p++) printf("part:%d HP:%f\n",p,HP(p));
    #endif
    #ifdef DEBUG_CONVERGENCE
    printf("\nOuter-iter:%d next_part:%d\n",counter,current_part);
    #endif
  }

}

real PPI::Priority(int s, real threshold) {

  Vector Qsa;
  Qsa.Resize(n_actions);
  for (int a=0; a<n_actions; a++) {
      real Q_sa = 0.0;
      const DiscreteStateSet& next = mdp->getNextStates(s, a);
      for (DiscreteStateSet::iterator i=next.begin(); i!=next.end();++i) {
          int s2 = *i;
          real P = mdp->getTransitionProbability(s, a, s2);
          real R = mdp->getExpectedReward(s, a) - baseline;
          Q_sa += P * (R + gamma * V(s2));
      }
      Qsa(a) = Q_sa;
  }
  real difference = Max(Qsa) - V(s);
  B(s) = difference;

//printf("s%d pV:%.8lf diff:%.8lf\n",s,V(s),difference);
#ifdef DEBUG_PPI
  for (int a=0; a<n_actions; a++) printf("s:%d a%d Qsa:%.8lf\n",s,a,Qsa(a));
#endif
  //H2 metric
//  if (B(s) > threshold)
//    return B(s) + V(s);
//  else
//    return 0;


  return B(s);
}

void PPI::EvaluateStateValues(int current_partition,int max_iter,real threshold)
{
  assert(policy);
  int n_iter = 0;
  do {
      Delta = 0.0;
//      for (int s=0; s<n_states; s++) {
      for(auto const& s: partitions[current_partition]) {
          real pV =0.0;
        #ifdef DEBUG_PPI
        printf (" S: %d a_max:%d ", s,a_max[s]);
        #endif
          for (int a=0; a<n_actions; a++) {
              real p_sa = policy->getActionProbability(s, a);
              if (p_sa > 0.0) {
                #ifdef DEBUG_PPI
		            printf(" a:%d p_sa:%f ",a,p_sa);
                #endif
                pV += p_sa * getValue(s, a);
              }
          }
          #ifdef DEBUG_PPI
          printf ("s:%d V:%f pV:%f\n",s,V(s),pV);
          #endif
          Delta += fabs(V[s] - pV);
          V[s] = pV;
      }

      if (max_iter > 0) {
          max_iter--;
      }
      n_iter++;
      #ifdef DEBUG_PPI
      printf("Eval cond1:%d max_iter:%d\n",Delta>threshold,max_iter);
      #endif
  } while((Delta > threshold) && max_iter != 0);
  #ifdef DEBUG_PPI
  printf ("Exiting Eval at delta = %f, after %d iter\n", Delta, n_iter);
  #endif
}

/// Get the value of a particular state-action pair
real PPI::getValue (int state, int action) const
{
    real V_next = 0.0;
    //for (int s2=0; s2<n_states; s2++) {
    DiscreteStateSet next = mdp->getNextStates(state, action);
    for (DiscreteStateSet::iterator i=next.begin();
         i!=next.end();
         ++i) {
        int s2 = *i;
        real P = mdp->getTransitionProbability(state, action, s2);
        V_next += P * V[s2];
    }
	real V_s = mdp->getExpectedReward(state, action) + gamma*V_next - baseline;
    return V_s;
}

/// Create the greedy policy with respect to the calculated value function.
FixedDiscretePolicy* PPI::getPolicy()
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
    return new FixedDiscretePolicy(n_states,n_actions,policy->p);
#endif
}
