/* -*- Mode: C++; -*- */
// VER: $Id: Distribution.c,v 1.3 2006/11/06 15:48:53 cdimitrakakis Exp cdimitrakakis $
// copyright (c) 2004 by Christos Dimitrakakis <christos.dimitrakakis@gmail.com>
/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ***************************************************************************/

#include "NormalDistribution.h"
#include "Random.h"

// Taken from numerical recipes in C
real NormalDistribution::generate() const
{
    if(!cache) {
        normal_x = urandom();
        normal_y = urandom();
        normal_rho = sqrt(-2.0 * log(1.0 - normal_y));
        cache = true;
    } else {
        cache = false;
    }
    
    if (cache) {
        return normal_rho * cos(2.0 * M_PI * normal_x) * s + m;
    } else {
        return normal_rho * sin(2.0 * M_PI * normal_x) * s + m; 
    }
}

real NormalDistribution::generate()
{
    if(!cache) {
        normal_x = urandom();
        normal_y = urandom();
        normal_rho = sqrt(-2.0 * log(1.0 - normal_y));
        cache = true;
    } else {
        cache = false;
    }
    
    if (cache) {
        return normal_rho * cos(2.0 * M_PI * normal_x) * s + m;
    } else {
        return normal_rho * sin(2.0 * M_PI * normal_x) * s + m; 
    }
}
/// Normal distribution pdf
real NormalDistribution::pdf(real x) const
{
    real d = (m-x)/s;
    return exp(-0.5 * d*d)/(sqrt(2.0 * M_PI) * s);
}

//------------------ Unknown mean -----------------------//
real NormalDistributionUnknownMean::generate()
{
    return prior.generate();
}

real NormalDistributionUnknownMean::generate() const
{
    return prior.generate();
}

/// TODO: Check that the marginal likelihood is correctly calculated
real NormalDistributionUnknownMean::pdf(real x) const
{

    real mean = mu_n / tau_n;
    real sigma = 1.0 / tau_n + 1.0 / tau;
    real d = (mean - x )/sigma;
    return exp(-0.5 * d*d)/(sqrt(2.0 * M_PI) * sigma);
    //return prior.pdf(x);
}

void NormalDistributionUnknownMean::calculatePosterior(real x)
{
    //    sum += x;
    n++;
    mu_n += tau * x;
    tau_n += tau;
    prior.setMean(mu_n);
    prior.setSTD(1.0 / tau_n);
    observations.setMean(mu_n);
}

real NormalDistributionUnknownMean::getMean() const
{
    return mu_n / tau_n;
}

//------------------- Unknown mean and precision -------------------------//
NormalUnknownMeanPrecision::NormalUnknownMeanPrecision()
{
    mu_0 = 0.0;
    tau_0 = 1.0;
    alpha_0 = 1;
    beta_0 = 1;
    Reset();
}

NormalUnknownMeanPrecision::NormalUnknownMeanPrecision(real mu_0_, real tau_0_)
    : mu_0(mu_0_), tau_0(tau_0_)
{
    alpha_0 = 1;
    beta_0 = 1;
    Reset();
}
void NormalUnknownMeanPrecision::Reset()
{
    p_x_mr.setMean(mu_0);
    p_x_mr.setVariance(1.0 / (tau_0 * tau_0));
    n = 0;
    sum = 0.0;
    tau_n = tau_0;
    mu_n = mu_0 * tau_0;
    alpha_n = alpha_0;
    beta_n = beta_0;
    M_2n = 0;
    bx_n = 0;
}
NormalUnknownMeanPrecision::~NormalUnknownMeanPrecision()
{
}
real NormalUnknownMeanPrecision::generate()
{
    Serror("Fix me!\n");
    return mu_n;
}
real NormalUnknownMeanPrecision::generate() const
{
    Serror("Fix me!\n");
    return mu_n;
}

real NormalUnknownMeanPrecision::Observe(real x)
{
    real p = pdf(x);
    calculatePosterior(x);
    return p;
}
/** The marginal pdf of the observations.
    
    Instead of calculating the actual marginal:
    \f[
    \xi(x) = \int f(x \mid m, r) \, d\xi(m, r),
    \f]
    we calculate:
    \f[
    \xi(x) = f(x \mid E_\xi m, E_\xi r),
    \f]
    where \f$E_\xi m = \int m d\xi(m) \f$, \f$E_\xi r = \int r d\xi(r) \f$.
*/
real NormalUnknownMeanPrecision::pdf(real x) const
{
    return p_x_mr.pdf(x);
}


real NormalUnknownMeanPrecision::getMean() const
{
    return mu_n;
}


void NormalUnknownMeanPrecision::calculatePosterior(real x)
{
    n++;
    real bxn_prev = bx_n;
    real rn = (real) n;
    real irn = 1 / rn;
    bx_n = bx_n + (x - bx_n) * irn;
    M_2n = M_2n + (x - bx_n)*(x - bxn_prev);
    real delta_mean = (bx_n - mu_0);
    
    real gamma = (rn - 1) * irn;
    mu_n = mu_n * gamma + (1 - gamma) * x;
    tau_n += 1.0;

    alpha_n += 0.5;
    beta_n = beta_0 + 0.5 * (M_2n + tau_0*n*delta_mean*delta_mean/(tau_0 + rn));

    p_x_mr.setMean(mu_n);
    p_x_mr.setVariance(beta_n / alpha_n);
}


//----------------------- Multivariate -----------------------------------//
/// Multivariate Gaussian generation
void MultivariateNormal::generate(Vector& x)
{
	x = mean;
}

/// Multivariate Gaussian generation
Vector MultivariateNormal::generate()
{	
	return mean; /// NOTE: FIX ME
}

/// Multivariate Gaussian density
real MultivariateNormal::pdf(Vector& x) const
{
	assert (x.Size()==mean.Size());
	real n = (real) x.Size();
    Matrix diff = x - mean;
	real d = (Transpose(diff) * accuracy * diff)(0,0);
	real determinant = 1.0f; // well, not really.
	real log_pdf = - 0.5 * d - 0.5*(log(2.0*M_PI)*n + log(determinant));
	return exp(log_pdf);
}


//----------------- Multivariate Unknown mean and precision -----------------------//
MultivariateNormalUnknownMeanPrecision::MultivariateNormalUnknownMeanPrecision()
{
    mu_0 = 0.0;
    tau_0 = 1.0;
    alpha_0 = 1;
    beta_0 = 1;
    Reset();
}

MultivariateNormalUnknownMeanPrecision::MultivariateNormalUnknownMeanPrecision(real mu_0_, real tau_0_)
    : mu_0(mu_0_), tau_0(tau_0_)
{
    alpha_0 = 1;
    beta_0 = 1;
    Reset();
}
void MultivariateNormalUnknownMeanPrecision::Reset()
{
    p_x_mr.setMean(mu_0);
    p_x_mr.setVariance(1.0 / (tau_0 * tau_0));
    n = 0;
    sum = 0.0;
    tau_n = tau_0;
    mu_n = mu_0 * tau_0;
    alpha_n = alpha_0;
    beta_n = beta_0;
    M_2n = 0;
    bx_n = 0;
}
MultivariateNormalUnknownMeanPrecision::~MultivariateNormalUnknownMeanPrecision()
{
}
real MultivariateNormalUnknownMeanPrecision::generate()
{
    Serror("Fix me!\n");
    return mu_n;
}
real MultivariateNormalUnknownMeanPrecision::generate() const
{
    Serror("Fix me!\n");
    return mu_n;
}

real MultivariateNormalUnknownMeanPrecision::Observe(real x)
{
    real p = pdf(x);
    calculatePosterior(x);
    return p;
}
/** The marginal pdf of the observations.
    
    Instead of calculating the actual marginal:
    \f[
    \xi(x) = \int f(x \mid m, r) \, d\xi(m, r),
    \f]
    we calculate:
    \f[
    \xi(x) = f(x \mid E_\xi m, E_\xi r),
    \f]
    where \f$E_\xi m = \int m d\xi(m) \f$, \f$E_\xi r = \int r d\xi(r) \f$.
*/
real MultivariateNormalUnknownMeanPrecision::pdf(real x) const
{
    return p_x_mr.pdf(x);
}


real MultivariateNormalUnknownMeanPrecision::getMean() const
{
    return mu_n;
}


void MultivariateNormalUnknownMeanPrecision::calculatePosterior(real x)
{
    n++;
    real bxn_prev = bx_n;
    real rn = (real) n;
    real irn = 1 / rn;
    bx_n = bx_n + (x - bx_n) * irn;
    M_2n = M_2n + (x - bx_n)*(x - bxn_prev);
    real delta_mean = (bx_n - mu_0);
    
    real gamma = (rn - 1) * irn;
    mu_n = mu_n * gamma + (1 - gamma) * x;
    tau_n += 1.0;

    alpha_n += 0.5;
    beta_n = beta_0 + 0.5 * (M_2n + tau_0*n*delta_mean*delta_mean/(tau_0 + rn));

    p_x_mr.setMean(mu_n);
    p_x_mr.setVariance(beta_n / alpha_n);
}


