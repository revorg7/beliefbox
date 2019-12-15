// -*- Mode: c++ -*-
// copyright (c) 2006 by Christos Dimitrakakis <christos.dimitrakakis@gmail.com>
// $Revision$
/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ***************************************************************************/

#ifndef PPI_H
#define PPI_H

#include "DiscreteMDP.h"
#include "DiscretePolicy.h"
#include <unordered_map>
#include "real.h"
#include <vector>

/** A Partitioned Prioritized VI for large discrete MDPs */
class PPI
{
protected:
    const DiscreteMDP* mdp; ///< pointer to the MDP
public:
    std::vector<int> a_max;
    FixedDiscretePolicy* policy;
    Vector V;

    real gamma; ///< discount factor
    int n_states; ///< number of states
    int n_actions; ///< number of actions
    real Delta;
    real baseline;

	//Specific to PPI
	int state_decoder_const;///< The constant 'n' in which the states were originially encoded as i.e. nxn
	int part_encoder_const;///< how many states is in one block per dimesion (everything assuming square grid)
	int part_decoder_const;///<  No. of states per dimension/no. of partitions in that dimension.
    int *arr; ///< Partition assignment of a state
    int *stability; ///< Store the latest no. of iterations performed for each partition during its PI, it the sum is == no. of paritions, then policy has converged overall
	std::unordered_map<int, std::vector<int>> partitions;///< gives states belonging to a partition
  std::unordered_map<int, std::vector<int>> SDS;;///< defined as in paper Wingate, 2005. JMRL.
	int *SDP;///< defined as in paper Wingate, 2005. JMRL.
	int *PDP;///< defined as in paper Wingate, 2005. JMRL.

  //For recursive part of the algorithm
  Vector H;///< Priority of States
  Vector HP;///< Priority of Partition
  Vector HPP;///< Relative Priority of Partition
  Vector B;///< Bellman error ("Potential Change"), just used for Priority. Not for Convergence criteria, which is max(Delta(s)) and equivalent to 1-backup difference between V-functions

	PPI(const DiscreteMDP* mdp, real gamma, int state_decoder_const, int part_encoder_const, real baseline=0.0);
    ~PPI();
  real Priority(int s, real threshold);
  void EvaluateStateValues(int current_partition,int max_iter,real threshold);
  real getValue (int state, int action) const;


    void Reset();
    void ComputeStateValuesStandard(real threshold, int max_iter=-1);
    void ComputeStateValuesAsynchronous(int current_partition, int max_iter, real threshold);
//    void ComputeStateValuesElimination(int current_partition, real threshold, int max_iter=-1);
    FixedDiscretePolicy* getPolicy();
    void InitializePartitions();

    inline void ComputeStateValues(real threshold, int max_iter=-1)
    {
		//ComputeStateValuesElimination(threshold, max_iter);
		ComputeStateValuesStandard(threshold, max_iter);
    }
    /// Set the MDP to something else
    inline void setMDP(const DiscreteMDP* mdp_)
    {
        mdp = mdp_;
    }
    inline void setDiscount(real gamma_) {
        assert(gamma >= 0.0 && gamma <= 1.0);
        gamma = gamma_;
    }
        inline real getValue (int state)
        {
            return getValue(state);
        }
};
#endif
