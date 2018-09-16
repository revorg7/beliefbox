/* -*- Mode: C++; -*- */
// copyright (c) 20012 by Christos Dimitrakakis <christos.dimitrakakis@gmail.com>
/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ***************************************************************************/

#ifndef DIRICHLET_FINITE_OUTCOMES_H
#define DIRICHLET_FINITE_OUTCOMES_H

#include "Distribution.h"

/** Finite outcome Dirichlet.

    This is a Dirichlet distirbution with a finite, but unknown, set
    of outcomes. It is a straightforward extension of the Dirichlet
    distribution.

	This particular version of the process assumes that all values will eventually be seen.
 */
class DirichletFiniteOutcomes : public VectorDistribution
{
protected:
//    const real pi = 3.14159265358979323846;
    int n; ///< size of multinomial distribution
    Vector alpha; ///< size of vector
    real alpha_sum; ///< sum of the vector
    int n_observations; ///< number of observations seen so far

    real C_dl;///< current C_dl
    int n_seen_symbols;
    Vector unseen_symbols; //last elements are seen_symbols, leading elements are unseen, on which random_permutations are performed according to Fisher-Yates_shuffle (//https://stackoverflow.com/questions/9345087/choose-m-elements-randomly-from-a-vector-containing-n-elements)
    real prior_alpha;
    int type_of_prior;		//0 for exponential, 1 for polynomial. refer: https://papers.nips.cc/paper/1616-efficient-bayesian-parameter-estimation-in-large-discrete-domains.pdf
    real prior_constant;
    real stirling(real x);
    Vector m_k;		//unlike paper, m_k refers to normalized values
  public:
    DirichletFiniteOutcomes();
    DirichletFiniteOutcomes(int n, real p = 1.0);
    DirichletFiniteOutcomes(const DirichletFiniteOutcomes& obj);
    virtual ~DirichletFiniteOutcomes();
    virtual void generate(Vector& x) const;
    virtual Vector generate() const;
    virtual real pdf(const Vector& x) const;
    virtual real marginal_pdf(int i) const;
    virtual real log_pdf(const Vector& x) const;
    virtual void update(Vector* x);
    virtual real Observe(int i);
    virtual Vector getMarginal() const;
    virtual void resize(int n, real p = 0.0);

    void printParams() const;
    inline Vector getParameters() const
    {
        return alpha;
    }

    inline int getCounts() const
    {
        return n_observations;
    }


};


#endif


