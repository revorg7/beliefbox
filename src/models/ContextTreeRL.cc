/* -*- Mode: c++;  -*- */
// copyright (c) 2010 by Christos Dimitrakakis <christos.dimitrakakis@gmail.com>
/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ***************************************************************************/

#include "ContextTreeRL.h"
#include <cmath>
#include <cassert>

ContextTreeRL::Node::Node(int n_branches_,
                          int n_outcomes_)
    : n_branches(n_branches_),
      n_outcomes(n_outcomes_),
      depth(0),
      prev(NULL),
      next(n_branches),
      P(n_outcomes), alpha(n_outcomes), prior_alpha(0.5),
      w(1), log_w(0), log_w_prior(0), Q(1)
{
    for (int i=0; i<n_outcomes; ++i) {
        P(i) = 1.0 / (real) n_outcomes;
        alpha(i) = 0;
    }
}

/// Make a node for K symbols at nominal depth d
ContextTreeRL::Node::Node(ContextTreeRL::Node* prev_)
    : n_branches(prev_->n_branches),
      n_outcomes(prev_->n_outcomes),
      depth(prev_->depth + 1),
      prev(prev_),
      next(n_branches),
      P(n_outcomes),
      alpha(n_outcomes),
      prior_alpha(0.5),
      log_w(0),
      log_w_prior(prev_->log_w_prior - log(2)),
      Q(1)
      //log_w_prior( - log(10))
{
    w = exp(log_w_prior);
    for (int i=0; i<n_branches; ++i) {
        next[i] = NULL;
    }
    for (int i=0; i<n_outcomes; ++i) {
        P(i) = 1.0 / (real) n_outcomes;
        alpha(i) = 0;
    }

}

/// make sure to kill all
ContextTreeRL::Node::~Node()
{
    for (int i=0; i<n_branches; ++i) {
        delete next[i];
    }
}


/** Obtain a new observation.

    This function serves multiple purposes

    1. It predicts the next observation.
    2. It adapts the parameters of the predictors.
    3. It adds the node to the list of active context.
    4. It finds/creates the next Node and calls Observe(), if needed.
    5. It stores the current context probability.
    6. It adapts the weight of the context.
    7. It returns the prediction.
   
 */
real ContextTreeRL::Node::Observe(Ring<int>& history,
                                  Ring<int>::iterator x,
                                  int y,
                                  real r,
                                  real probability,
                                  std::list<Node*>& active_contexts)
{
    active_contexts.push_back(this);
    //printf ("contexts: %d, d:%d\n", active_contexts.size(), depth);
    real total_probability = 0;
    // calculate probabilities

    // Standard
#if 1
    real S = alpha.Sum();
    real Z = 1.0 / (prior_alpha * (real) n_outcomes + S);
    P = (alpha + prior_alpha) * Z;
    P /= P.Sum();
    //P[y] = (alpha[y] + prior_alpha) * Z;
#endif

#if 0
    // aka: I-BVMM -- best for many outcomes
    real S = alpha.Sum();
    real N = 0;
    for (int i=0; i<n_outcomes; ++i) {
        if (alpha(i)) {
            N += 1;
        }
    }
    real Z = (1 + N) * prior_alpha + S;
    P = (alpha + prior_alpha) / Z;
    real n_zero_outcomes = n_outcomes - N;
    if (n_zero_outcomes > 0) {
        real SA = 1.0 / n_zero_outcomes;
        for (int i=0; i<n_outcomes; ++i) {
            if (alpha(i)==0) {
                P(i) *= SA;
            }
        }
    }
#endif
    alpha[y]++;

    // Do it for probability too
    real p_reward = reward_prior.Observe(r);
        
    // P(y | B_k) = P(y | B_k, h_k) P(h_k | B_k) + (1 - P(h_k | B_k)) P(y | B_{k-1})
    w = exp(log_w_prior + log_w); 
    assert(w >= 0 && w <= 1);

    real p_observations = P[y] * p_reward;
    total_probability = p_observations * w + (1 - w) * probability;

    //real log_w_prev = log_w;
    //real log_w2 = log_w  + log(p_observations) - log(total_probability);    
    log_w = log(w * p_observations / total_probability) - log_w_prior;

    //log_w = log_w + log(p_observations) - log(total_probability) - log_w_prior;

    assert(log_w + log_w_prior <= 0);
    //assert(log_w2 + log_w_prior <= 0);

    assert(!isnan(log_w));

    w_prod = 1; ///< auxilliary calculation

    // Go deeper if the context is long enough and the number of
    // observations justifies it.
    real threshold = 2;
    if (x != history.end() && S >  threshold) {
        int k = *x;
        ++x;
        if (!next[k]) {
            next[k] = new Node(this);
        }
        total_probability = next[k]->Observe(history, x, y, r, total_probability, active_contexts);
        w_prod = next[k]->w_prod; ///< for post facto context probabilities
        assert(!isnan(total_probability));
        assert(!isnan(w_prod));
    }

    // Auxilliary calculation for context
    context_probability = w * w_prod;
    w_prod *= (1 - w);

    assert(!isnan(w_prod));
    assert(!isnan(context_probability));

    return total_probability;
}



/// Recursive calculation of the Q-value.
real ContextTreeRL::Node::QValue(Ring<int>& history,
                                 Ring<int>::iterator x,
                                 real Q_prev)
{
    //w = exp(log_w_prior + log_w); 
    real Q_next = Q * w + (1 - w) * Q_prev;
    int k = *x;
    if (x != history.end() && next[k]) {
        ++x;
        Q_next = next[k]->QValue(history, x, Q_next);
    }

    return Q_next;
}






void ContextTreeRL::Node::Show()
{
	
    std::cout << w << " " << depth << "# weight depth\n";
    for (int i=0; i<n_outcomes; ++i) {
        std::cout << alpha[i] << " ";
    }
    for (int k=0; k<n_branches; ++k) {
        if (next[k]) {
            std::cout << "b: " << k << std::endl;
            next[k]->Show();
        }
    }
    std::cout << "<<<<\n";
}

int ContextTreeRL::Node::NChildren()
{
    int my_children = 0;
    for (int k=0; k<n_branches; ++k) {
        if (next[k]) {
            my_children++;
            my_children += next[k]->NChildren();
        }
    }
    return my_children;
}

ContextTreeRL::ContextTreeRL(int n_branches_,
                             int n_observations_,
                             int n_actions_,
                             int n_symbols_,
                             int max_depth_)
    : n_branches(n_branches_),
      n_observations(n_observations_),
      n_actions(n_actions_),
      n_symbols(n_symbols_),
      max_depth(max_depth_),
      history(max_depth)
{
    root = new Node(n_branches, n_symbols);
    std::cout << "# Making new CTRL with depth " << max_depth << std::endl;
}

ContextTreeRL::~ContextTreeRL()
{
    delete root;
}

/// Observe complete observation x, action y, reward r
real ContextTreeRL::Observe(int x, int y, real r)
{
    active_contexts.clear();
    history.push_back(x);
    return root->Observe(history, history.begin(), y, r, 0, active_contexts);
}

void ContextTreeRL::Show()
{
    root->Show();
    std::cout << "Total contexts: " << NChildren() << std::endl;
}

int ContextTreeRL::NChildren()
{
    return root->NChildren();
}
/** Q Learning implementation.
    
 */
real ContextTreeRL::QLearning(real step_size, real gamma, int observation, real reward)
{
    //real Q_prev = root->QValue(history, history.begin(), 0);
    //assert(!isnan(Q_prev));

    real max_Q = -INF;    
    for (int a = 0; a < n_actions; ++a) {
        int x = a * n_observations + observation;
        real Q_x = QValue(x);
        if (Q_x > max_Q) {
            max_Q = Q_x;
        }
    }

    max_Q += reward;
    real td_err = 0;
    for (std::list<Node*>::iterator i = active_contexts.begin();
         i != active_contexts.end();
         ++i) {
        real p_i = (*i)->context_probability;
        real dQ_i = reward + gamma * max_Q - (*i)->Q; ///< This works even better!
        //real dQ_i = reward + gamma * max_Q - Q_prev; ///< This works OK!
        real delta = p_i * dQ_i; 
        (*i)->Q += step_size * delta;
        //printf ("%f * %f = %f ->  %f\n", p_i, dQ_i, delta, (*i)->Q);
        td_err += fabs(delta);
    }
    return td_err;
}

/// Get a Q value
///
/// x = (y, a)
real ContextTreeRL::QValue(int x)
{
    Ring<int> tmp_history(history);
    tmp_history.push_back(x);
    return root->QValue(tmp_history, tmp_history.begin(), 0);
}
