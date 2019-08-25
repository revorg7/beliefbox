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

#ifndef POLICY_ITERATION_H
#define POLICY_ITERATION_H

#include "PolicyEvaluation.h"
#include "DiscreteMDP.h"
#include "DiscretePolicy.h"
#include "real.h"
#include <vector>

class PolicyIteration
{
protected:
    PolicyEvaluation* _evaluation;
public:
    PolicyEvaluation* evaluation;
    const DiscreteMDP* mdp;
    FixedDiscretePolicy* policy;
    std::vector<int> a_max;
    real gamma;
    int n_states;
    int n_actions;
    real Delta;
    real baseline;
    PolicyIteration(PolicyEvaluation* evaluation_,
                    const DiscreteMDP* mdp_,
                    real gamma_,
                    real baseline_ = 0.0);
    PolicyIteration(const DiscreteMDP* mdp_,
                    real gamma_,
                    real baseline_ = 0.0);
    ~PolicyIteration();
    void Reset();
    void ComputeStateValues(int max_iter,real evaluation_threshold); //<<< Max iter should be the most imp parameter, not evaluation_threshold (which when reduced makes it look like more and more VI)
    inline real getValue (int state, int action)
    {
        return evaluation->getValue(state, action);
    }
    inline real getValue (int state)
    {
        return evaluation->getValue(state);
    }
};

#endif

