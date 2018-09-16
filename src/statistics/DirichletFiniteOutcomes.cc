/* -*- Mode: C++; -*- */
/* VER: $Id: Distribution.h,v 1.3 2006/11/06 15:48:53 cdimitrakakis Exp cdimitrakakis $*/
// copyright (c) 2006 by Christos Dimitrakakis <christos.dimitrakakis@gmail.com>
/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ***************************************************************************/

#include "DirichletFiniteOutcomes.h"
#include "ranlib.h"
#include "SpecialFunctions.h"
#include <cmath>
#include <random>
#include <algorithm>


/// Create a placeholder Dirichlet
DirichletFiniteOutcomes::DirichletFiniteOutcomes()
    : n_observations(0),C_dl(-1),n_seen_symbols(0), prior_alpha(0.25),type_of_prior(0),prior_constant(0.8)
{
    Swarning("Invalid Constructor, Alphabet Size needed\n");
    n = 0;
    alpha_sum = 1.0;
    m_k = Vector(n);
    unseen_symbols.Resize(n);
    for (int i=0;i<n;i++) unseen_symbols[i] = i;
}

/// Create a Dirichlet with uniform parameters
DirichletFiniteOutcomes::DirichletFiniteOutcomes(int n, real p)
    : n_observations(0),C_dl(-1),n_seen_symbols(0),prior_alpha(p),type_of_prior(0),prior_constant(0.8)
{
//    alpha_sum = p;
    resize(n,p);
    for (int i=0; i<n; ++i) {
        alpha(i) = p;
    }
    alpha_sum = (real) n * p;

    m_k = Vector(n);
    unseen_symbols.Resize(n);
    for (int i=0;i<n;i++) unseen_symbols[i] = i;
}


/// Initialise parameters from an object
DirichletFiniteOutcomes::DirichletFiniteOutcomes(const DirichletFiniteOutcomes& obj) 
    : n(obj.n), alpha(obj.alpha), alpha_sum(obj.alpha_sum), n_observations(obj.n_observations), C_dl(obj.C_dl), n_seen_symbols(obj.n_seen_symbols), prior_alpha(obj.prior_alpha), type_of_prior(obj.type_of_prior),
	prior_constant(obj.prior_constant), m_k(obj.m_k), unseen_symbols(obj.unseen_symbols)
{
}


/// Destructor
DirichletFiniteOutcomes::~DirichletFiniteOutcomes()
{
}

/// Striling approx. used
real DirichletFiniteOutcomes::stirling(real x)
{
//    if (x == 0) return 0.0;
//    else return x*log(x) - x + 1.0/6.0 * log(4*pow(x,2)*(2*x+1) + x + 1.0/30) + 0.5*log(pi);
    return logGamma(x+1);
}


/// Generate a multinomial vector
Vector DirichletFiniteOutcomes::generate() const
{
    Vector x(n);
    generate(x);
    return x;
}


/// Generate a multinomial vector in-place
void DirichletFiniteOutcomes::generate(Vector& y) const
{

    //Generating the seen symbols with alpha(i)
    real sum = 0.0;
    for (int i=0;i<n;i++) {
        if (alpha(i) > prior_alpha) {
            y(i) = gengam(1.0, alpha(i));
            sum += y(i);
	}
    }

    //Generating random length k for k dimensional dirichlet-distr
    real number = ranf();
    real sum1 = 0.0;
    int k = n;
    for (int i=0; i<n; ++i) {
        sum1 += m_k[i];
	if (number < sum1){ k = i+1; break;}
    }
    //Randomly selecting k-k0 new symbols and assigning probabilities
    if (k-n_seen_symbols > 0) {
	int len = n - n_seen_symbols;
	Vector vals(unseen_symbols);
	for (int i=0; i<k-n_seen_symbols; i++) {
            int r = i + (rand() % (len-i)); // Random remaining position.
            int temp = vals[i]; vals[i] = vals[r]; vals[r] = temp;
        }

	for (int i=0;i<k-n_seen_symbols;i++) {
	    y(int(vals[i])) = gengam(1.0, prior_alpha); 
            sum += y(int(vals[i]));
	}
    }

    real invsum = 1.0 / sum;
    y *= invsum;

    assert(fabs(y.Sum() - 1.0) < 1e-6);

}

/** Dirichlet distribution
    Gets the parameters of a multinomial distribution as input.
*/
real DirichletFiniteOutcomes::pdf(const Vector& x) const
{
    Swarning("Not correctly implemented\n");
	return exp(log_pdf(x));
}

/** Dirichlet distribution

    Gets the parameters \f$x\f$ of a multinomial distribution as
	input.
	
	
	Returns the logarithm of the pdf.

*/
real DirichletFiniteOutcomes::log_pdf(const Vector& x) const
{
    assert(x.Size() == n);

    real log_prod = 0.0;
    real sum = 0.0;
    for (int i=0; i<n; i++) {
        real xi = x(i);
        if (xi<=0) {
            Swarning ("Got a negative value for x[%d]:%f\n", i, xi);
            return 0.0;
        }
        sum += xi;
        log_prod += log(xi) * alpha(i);
    }
    if (fabs(sum-1.0f)>0.001) {
        Swarning ("Vector x not a distribution apparently: sum=%f.  Returning 0.\n", sum);
        return 0.0;
    }
    Swarning("Not correctly implemented\n");
    return log_prod - logBeta(alpha);
}

void DirichletFiniteOutcomes::update(Vector* x)
{

    real tmp = 0;
    int j,temp;
    for (int i=0; i<n; ++i) {
        real xi = (*x)(i);
        if (xi > 0 && alpha(i) == prior_alpha) {
	    for (j=0; j<n-n_seen_symbols; j++) {
	        if (int(unseen_symbols[j]) == i) {
		    temp = unseen_symbols[j]; unseen_symbols[j] = unseen_symbols[n-n_seen_symbols-1]; unseen_symbols[n-n_seen_symbols-1] = temp;
		    break;
	        }
	    }
            n_seen_symbols++;
        }
        alpha(i) += xi;
        alpha_sum += xi;
	tmp += xi;
    }
    n_observations += (int) tmp;

    //Recalculating m_k, this can be avoided if N >> n_seen and n_seen doesnt changes (not implemented but see paper)
    Vector logs(n);
    for (int i=n_seen_symbols-1; i<logs.Size(); i++) {
	real k_dash = i+1;
	logs(i) = k_dash*log(prior_constant) + stirling(k_dash) - stirling(k_dash-n_seen_symbols) + stirling(k_dash*prior_alpha - 1) - stirling(k_dash*prior_alpha + n_observations - 1);
    }
    real base_value = logs(n_seen_symbols-1);
    real sum = 1.0;
    for (int i=n_seen_symbols; i < logs.Size(); i++) {
	if (base_value - logs(i) < 16.13) sum += exp(logs(i) - base_value);
    }
    real factor = 1.0/sum;
    for (int i=0; i < n; i++) {
	if (i < n_seen_symbols-1) m_k(i) = 0;
	else {
	    if (base_value - logs(i) < 16.13) {
	        m_k(i) = factor / exp(base_value - logs(i)); 
	    }
	    else m_k(i) = 0;
	}
    }

    //Recalculating C_dl
    C_dl = 0;
    real const_num = n_seen_symbols*prior_alpha + n_observations;
    for (int i=n_seen_symbols-1; i<n; ++i) {
	C_dl += ( const_num / ( (i+1)*prior_alpha + n_observations) )*m_k[i];
    }

    assert(fabs(m_k.Sum() - 1.0) < 1e-6);

}

/// When there is only one observation, give it directly.
real DirichletFiniteOutcomes::Observe(int i)
{
//    real p = getMarginal()(i);
    if (alpha(i) == prior_alpha) {
	for (int j=0; j<n-n_seen_symbols; j++) {
	    if (int(unseen_symbols[j]) == i) {
		int temp = unseen_symbols[j]; unseen_symbols[j] = unseen_symbols[n-n_seen_symbols-1]; unseen_symbols[n-n_seen_symbols-1] = temp;
		break;
	    }
	}
        n_seen_symbols++;
    }
    alpha(i) += 1.0;
    alpha_sum += 1.0;
    n_observations+=1;

    //Recalculating m_k, this can be avoided if N >> n_seen and n_seen doesnt changes (not implemented but see paper)
    Vector logs(n);
    for (int i=n_seen_symbols-1; i<logs.Size(); i++) {
	real k_dash = i+1;
	logs(i) = k_dash*log(prior_constant) + stirling(k_dash) - stirling(k_dash-n_seen_symbols) + stirling(k_dash*prior_alpha - 1) - stirling(k_dash*prior_alpha + n_observations - 1);
    }
    real base_value = logs(n_seen_symbols-1);
    real sum = 1.0;
    for (int i=n_seen_symbols; i < logs.Size(); i++) {
	if (base_value - logs(i) < 16.13) sum += exp(logs(i) - base_value);
    }
    real factor = 1.0/sum;
    for (int i=0; i < n; i++) {
	if (i < n_seen_symbols-1) m_k(i) = 0;
	else {
	    if (base_value - logs(i) < 16.13) {
	        m_k(i) = factor / exp(base_value - logs(i)); 
	    }
	    else m_k(i) = 0;
	}
    }

    //Recalculating C_dl
    C_dl = 0;
    real const_num = n_seen_symbols*prior_alpha + n_observations;
    for (int i=n_seen_symbols-1; i<n; ++i) {
	C_dl += ( const_num / ( (i+1)*prior_alpha + n_observations) )*m_k[i];
    }


    assert(fabs(m_k.Sum() - 1.0) < 1e-6);

    return 0.0;		//This is wrong, but getting marginal is expensive as well
}

/// Return the marginal probabilities
Vector DirichletFiniteOutcomes::getMarginal() const
{
    Vector P(n);
    real const_num = n_seen_symbols*prior_alpha + n_observations;
    for (int i=0; i<n; ++i) {
	if (alpha(i)>prior_alpha) {
	    P(i) = ( alpha(i)/ const_num ) *C_dl;
	}
    }
    real remaining_probability = (1.0 - P.Sum())/ (n-n_seen_symbols);
    for (int i=0; i<n; ++i) {
	if (alpha(i)<=prior_alpha) {
	    P(i) = remaining_probability;
	}
    }

    assert(fabs(P.Sum() - 1.0) < 1e-6);

    return P;
}

void DirichletFiniteOutcomes::resize(int n, real p)
{
    this->n = n;
    prior_alpha = p;
    alpha.Resize(n);
    alpha.Clear();
}

/// Return the marginal probabilities
real DirichletFiniteOutcomes::marginal_pdf(int i) const
{
	return getMarginal()(i);
}

void DirichletFiniteOutcomes::printParams() const
{
	printf("\nn_seen: %d,prior_alpha: %f,prior_constant:%f ",n_seen_symbols,prior_alpha,prior_constant);
	printf("\nalpha ");
	for (int i=0;i<alpha.Size();i++) printf("%f ",alpha[i]);
	printf("\n");
	printf("m_k ");
	for (int i=0;i<n;i++) printf("%f ",m_k[i]);
	printf("\n");
	printf("unseen_symbols ");
	for (int i=0;i<n-n_seen_symbols;i++) printf("%f ",unseen_symbols[i]);
	printf("\nseen_symbols ");
	for (int i=n-n_seen_symbols;i<n;i++) printf("%f ",unseen_symbols[i]);
	printf("\n");

}
