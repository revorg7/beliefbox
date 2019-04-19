/* -*- Mode: C++; -*- */
// copyright (c) 2004-2011 by Christos Dimitrakakis <christos.dimitrakakis@gmail.com>
/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ***************************************************************************/

#include "ranlib.h"
#include "BetaDistribution.h"
#include "SpecialFunctions.h"
#include "ExponentialDistribution.h"
#include <cstdlib>

//#define TBRL_DEBUG

/// Calculate
/// \f[
///  \exp(\log x(\alpha - 1) + \log (1-x)(\beta - 1) - B(\alpha, \beta)
///  = \frac {x^{\alpha -1}(1-x)^{\beta -1}}{B(\alpha, \beta)}
/// \f]
real BetaDistribution::pdf(real x) const
{
    if (x<0.0 || x>1.0) {
        return 0.0;
    }
    if (alpha == 1 && beta == 1) {
        return 1.0;
    }
    
    real log_pdf = 0;
    if (x == 0 && alpha == 1 && beta > 0) {
        log_pdf = -logBeta(alpha, beta);
    } else if (x == 1 && beta == 1 && alpha > 0) {
        log_pdf = -logBeta(alpha, beta);
    } else {
        log_pdf = log(x)*(alpha - 1.0) + log(1-x)*(beta - 1.0)- logBeta(alpha, beta);    
    }
    return exp(log_pdf);
}

real BetaDistribution::log_pdf(real x) const
{
    if (x<0.0 || x>1.0) {
        return log(0.0);
    }
    if (alpha == 1 && beta == 1) {
        return 0.0;
    }
    return log(x)*(alpha - 1.0) + log(1-x)*(beta - 1.0)- logBeta(alpha, beta);
}



/// Standard posterior calculation
void BetaDistribution::calculatePosterior(real x)
{
	assert (x>=0 && x <= 1);
    alpha += x;
    beta += (1.0-x);
}


void BetaDistribution::setMean(real mean)
{
    fprintf(stderr,"Warning: cannot set mean for Beta distribution\n");
} 

void BetaDistribution::setVariance(real var)
{
    fprintf(stderr, "Warning: cannot set variance for Beta distribution\n");
}

real BetaDistribution::getMean() const
{
#ifdef TBRL_DEBUG
	printf("alpa:%f beta:%f\n",alpha,beta);
#endif
    return alpha/(alpha + beta);
}

real BetaDistribution::getVariance() 
{
    real a_b = alpha + beta;
    return (alpha/a_b)*(beta/a_b)/(a_b + 1);
}


/// Generate using ranlib
real BetaDistribution::generate() const
{
	assert(alpha > 0 && (beta >= 0 || alpha >= 0) && beta > 0);
    return genbet(alpha, beta);
}

/// Generate from the marginal distribution
real BetaDistribution::generateMarginal() const
{
	//changing RNG to this from uniform() didn't change time or performance
	if (drand48() < getMean()) {
		return 1.0;
	} else {
		return 0.0;
	}
}

real BetaDistribution::marginal_pdf(real x) const
{
   return (x * (alpha) + (1 - x) * beta) / (alpha + beta);
}

real BetaDistribution::Observe(real x)
{
    real p = marginal_pdf(x);
    calculatePosterior(x);
    return p;
}

real BetaDistribution::setMaximumLikelihoodParameters(const std::vector<real>& x,
                                    int n_iterations)
{
    // First set up the mean and variance by the method of moments
    real Z = 1.0f / (real) x.size();
    real mean = Sum(x) * Z;

    real d = 0;
    for (uint i=0; i != x.size(); ++i) {
        assert(x[i] >= 0.0f && x[i] <= 1.0f);
        real delta = (x[i] - mean);
        d += delta * delta;
    }
#if 1
    // use moments for initial estimate
    real variance = d * Z;
    real S = mean * (1.0f - mean) / variance - 1.0f;
    real max_alpha = mean * S;
    real max_beta = (1.0f - mean) * S;
#else
    // just set to default values
    real max_alpha = 1;
    real max_beta = 1;
#endif
    real max_log_likelihood = Distribution::log_pdf(x);
    //printf ("%f %f %f %d # alpha, beta, LL, iter\n", max_alpha, max_beta, max_log_likelihood, n_iterations);
    ExponentialDistribution Exp;
    for (int k=0; k<n_iterations; ++k) {
        alpha = 1 + 10 * Exp.generate();
        beta = 1 + 10 * Exp.generate();
        real log_likelihood = Distribution::log_pdf(x);
        if (log_likelihood > max_log_likelihood) {
            max_alpha = alpha;
            max_beta = beta;
            max_log_likelihood = log_likelihood;
            //printf ("%f %f %f # alpha, beta\n", max_alpha, max_beta, max_log_likelihood);
        }
    }
    alpha = max_alpha;
    beta = max_beta;
    return max_log_likelihood;
}

BetaDistributionMCPrior::BetaDistributionMCPrior(real kappa_, real lambda_)
    : kappa(kappa_),
      lambda(lambda_)
{
}

BetaDistributionMCPrior::~BetaDistributionMCPrior()
{
}

real BetaDistributionMCPrior::LogLikelihood(std::vector<real>& x, int K) const
{
    //real t = (real) T;
    ExponentialDistribution ExpA(kappa);
    ExponentialDistribution ExpB(lambda);
    real log_likelihood = LOG_ZERO;
    for (int k=0; k<K; ++k) {
        real alpha = ExpA.generate();
        real beta = ExpB.generate();
        GammaDistribution beta_distribution_h(alpha, beta);
        real log_p = 0.0;
        int n = x.size();
        for (int i=0; i<n; ++i) {
            log_p += beta_distribution_h.log_pdf(x[i]);
        }
        log_likelihood = logAdd(log_likelihood, log_p);
    }
    log_likelihood -= log(K);
    return log_likelihood;
}
