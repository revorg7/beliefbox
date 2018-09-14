/* -*- Mode: C++; -*- */
// copyright (c) 2006 by Christos Dimitrakakis <christos.dimitrakakis@gmail.com>
/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ***************************************************************************/


#ifndef DIRICHLET_H
#define DIRICHLET_H

#include "Distribution.h"
#include "MultinomialDistribution.h"

/** Dirichlet distribution.
	
	This distribution is conjugate to the multinomial.
 */
class DirichletDistribution : public VectorDistribution
{
protected:
    int n; ///< size of multinomial distribution
    Vector alpha; ///< size of vector
    real alpha_sum; ///< sum of the vector
	int n_observations; ///< number of observations seen so far
public:
    DirichletDistribution();
    DirichletDistribution(int n, real p = 1.0);
    DirichletDistribution(const Vector& x);
//    DirichletDistribution(const DirichletDistribution& obj);	//Dont need it, since non of internal variables are pointer-based
    virtual ~DirichletDistribution();
    virtual void generate(Vector& x) const;
    virtual Vector generate() const;
    virtual real pdf(const Vector& x) const;
    virtual real log_pdf(const Vector& x) const;
    virtual void update(Vector* x);
    virtual real Observe(int i);
    virtual Vector getMarginal() const;
    virtual real marginal_pdf(int i) const;
    Vector getParameters() const;
    real& Alpha(int i)
    {
        return alpha[i];
    }
    int size() const
    {
        return n;
    }
    virtual void resize(int n, real p = 0.0);

    inline real getMass() const
    {
        return alpha_sum;
    }

    inline int getCounts() const
    {
        return n_observations;
    }
    
};

#endif

