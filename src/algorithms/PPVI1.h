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

#ifndef PPVI1_H
#define PPVI1_H

#include "DiscreteMDP.h"
#include "DiscretePolicy.h"
#include <unordered_map>
#include "Matrix.h"
#include "Vector.h"
#include "real.h"
#include <vector>

/** A Partitioned Prioritized VI for large discrete MDPs */
class PPVI1
{
protected:
    const DiscreteMDP* mdp; ///< pointer to the MDP
public:
    real gamma; ///< discount factor
    int n_states; ///< number of states
    int n_actions; ///< number of actions
    Vector V; ///< state values
    Vector pV; ///< previous statate value
    Matrix Q; ///< state-action value
    real Delta;
    real baseline;

	//Specific to PPVI1
	int state_decoder_const;///< The constant 'n' in which the states were originially encoded as i.e. nxn
	int part_encoder_const;///< how many states is in one block per dimesion (everything assuming square grid)
	int part_decoder_const;///<  No. of states per dimension/no. of partitions in that dimension.

  int *arr; ///< Partition assignment of a state
	std::unordered_map<int, std::vector<int>> partitions;///< gives states belonging to a partition
	int *SDP;///< defined as in paper Wingate, 2005. JMRL.

  //For recursive part of the algorithm
  Vector H;///< Priority of States
  Vector HP;///< Priority of Partition
  Vector HPP;///< Relative Priority of Partition
  Vector B;///< Bellman error, mainly used for convergence

	PPVI1(const DiscreteMDP* mdp, real gamma, int state_decoder_const, int part_encoder_const, real baseline=0.0);
    ~PPVI1();
  real Priority(int s, real threshold);

  
    void Reset();
    void ComputeStateValuesStandard(real threshold, int max_iter=-1);
    void ComputeStateValuesAsynchronous(int current_partition, real threshold, int max_iter=-1);
    FixedDiscretePolicy* getPolicy();
    void InitializePartitions();

    inline void ComputeStateValues(real threshold, int max_iter=-1)
    {
		ComputeStateValuesStandard(threshold, max_iter);
    }
    inline void ComputeStateActionValues(real threshold, int max_iter=-1)
	{
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
    inline real getValue (int state, int action)
    {
        return Q(state, action);
    }
    inline real getValue (int state)
    {
        return V(state);
    }
    inline Matrix getValues() const
    {
        return Q;
    }
    inline Vector getStateValues() const
    {
        return V;
    }
    inline Vector getValues(int s) const
    {
        assert(s >= 0 && s < n_states);
        return Q.getRow(s);
    }
};
#endif
