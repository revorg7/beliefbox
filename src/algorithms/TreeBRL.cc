// -*- Mode: c++ -*-
// copyright (c) 2010 by Christos Dimitrakakis <christos.dimitrakakis@gmail.com>
// $Revision$
/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ***************************************************************************/

#include "TreeBRL.h"

TreeBRL::TreeBRL(int n_states_,
                             int n_actions_,
                             real gamma_,
                             MDPModel* model_,
                             RandomNumberGenerator* rng_,
                             int horizon_,
                             bool use_upper_bound_)
    : n_states(n_states_),
      n_actions(n_actions_),
      gamma(gamma_),
      model(model_),
      rng(rng_),
      horizon(horizon_),
      T(0)
{
  
    printf("# Starting Tree-Bayes-RL with %d states, %d actions, %d horizon %d\n", n_states, n_actions, horizon);

    current_state = -1;
    
}
TreeBRL::~TreeBRL()
{


}
void TreeBRL::Reset()
{
    current_state = -1;
    next_update = T;

}
/// Full observation
real TreeBRL::Observe (int state, int action, real reward, int next_state, int next_action)
{
    if (state>=0) {
        model->AddTransition(state, action, reward, next_state);
    }
    current_state = next_state;
    current_action = next_action;
    return 0.0;
}
/// Partial observation 
real TreeBRL::Observe (real reward, int next_state, int next_action)
{
    if (current_state >= 0) {
        model->AddTransition(current_state, current_action, reward, next_state);
    }
    current_state = next_state;
    current_action = next_action;
    return 0.0;
}

void TreeBRL::RecalculateTree()
{
}

void TreeBRL::Resample()
{

  /// Get an action using the current exploration policy.
/// it calls Observe as a side-effect.
int TreeBRL::Act(real reward, int next_state)
{
    assert(next_state >= 0 && next_state < n_states);
    T++;

    // update the model
    if (current_state >= 0) {
        model->AddTransition(current_state, current_action, reward, next_state);
    }
    current_state = next_state;

    RecalculateTree();
    
    return current_action;
}

