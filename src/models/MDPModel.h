// -*- Mode: c++ -*-
// copyright (c) 2005 by Christos Dimitrakakis <christos.dimitrakakis@gmail.com>
// $Id: MDPModel.h,v 1.2 2006/10/31 16:59:39 cdimitrakakis Exp cdimitrakakis $
/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ***************************************************************************/

#ifndef MDP_MODEL_H
#define MDP_MODEL_H

#include <cmath>
#include <cassert>
#include <vector>
#include "real.h"
#include "SmartAssert.h"
#include "Distribution.h"


/**
   \ingroup MachineLearning
*/

/*! 
  \class MDPModel
  \brief A model of a Markov Decision Process

  Implements algorithms for learning the parameters of an observed MDP.
  This is not an abstract class, it used for discrete spaces.
*/
class MDPModel 
{
protected:
    int n_states; ///< number of states (or dimensionality of state space)
    int n_actions; ///< number of actions (or dimensionality of action space)
public:
    MDPModel (int n_states, int n_actions)
    {
        this->n_states = n_states;
        this->n_actions = n_actions;
    }
    inline virtual int GetNStates()
    {
        return n_states;
    }
    inline virtual int GetNActions()
    {
        return n_actions;
    }
    virtual ~MDPModel()
    {
    }
    virtual void AddTransition(int s, int a, real r, int s2) = 0;
    virtual real GenerateReward (int s, int a) {return 0.0;}
    virtual int GenerateTransition (int s, int a) = 0;
    virtual real getTransitionProbability (int s, int a, int s2) {return 0.0;}
    virtual real getExpectedReward (int s, int a) const = 0;
    virtual void Reset() = 0;

};

class GradientDescentMDPModel : public MDPModel
{
protected:
    real alpha; ///< learning rate.
    real** P;
    real* R;
    int N;
    int getID (int s, int a)
    {
        SMART_ASSERT(s>=0 && s<n_states)(s)(n_states);
        SMART_ASSERT(a>=0 && a<n_actions)(a)(n_actions);
        return s*n_actions + a;
    }
public:
    Distribution* initial_transitions;
    Distribution* initial_rewards; 
    GradientDescentMDPModel (int n_states, int n_actions, Distribution* initial_transitions, Distribution* initial_rewards);
    virtual ~GradientDescentMDPModel();
    virtual void AddTransition(int s, int a, real r, int s2);
    void ShowModel();
    virtual real GenerateReward (int s, int a);
    virtual int GenerateTransition (int s, int a);
    virtual real getTransitionProbability (int s, int a, int s2);
    virtual real getExpectedReward (int s, int a);
    //virtual int getNStates () { return N;}
    virtual void setLearningRate (real learning_rate)
    {
        alpha = learning_rate;
        assert (alpha>=0.0f && alpha <=1.0f);
    }
    virtual void Reset();
};


#endif
