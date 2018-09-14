// -*- Mode: c++ -*-
// copyright (c) 2013 by Christos Dimitrakakis <christos.dimitrakakis@gmail.com>
/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ***************************************************************************/

#include "DirichletTransitionsSparse.h"
#include "Distribution.h"

//#define TBRL_DEBUG2
 
DirichletTransitionsSparse::DirichletTransitionsSparse(int n_states_,
										   int n_actions_,
										   real prior_mass_,
										   bool uniform_unknown_)
		: n_states(n_states_),
		  n_actions(n_actions_),
		  prior_mass(prior_mass_),
		  uniform_unknown(uniform_unknown_)
{

#ifdef TBRL_DEBUG2
	std::printf("Size of dirichlet transitions map is: %d \n",P.size());
#endif
//	NOT A GOOD IDEA, SINCE IT CAN BE VERY LARGE
//	if (uniform_unknown) {
//	    for (int s=0; s<n_states; ++s) {
//		for (int a=0; a<n_actions; ++a) {
//
//		    DiscreteStateAction SA(s, a);
//		    P.insert(std::make_pair(SA, DirichletFiniteOutcomes(n_states, prior_mass)));
//		}
//	    }
//	} else {
//	    for (int s=0; s<n_states; ++s) {
//		for (int a=0; a<n_actions; ++a) {
//		    real zero = 0.0;
//		    Vector v(n_states,&zero);
//		    v[s] = 1.0;
//		    DiscreteStateAction SA(s, a);
//		    P.insert(std::make_pair(SA, v));
//		  }
//	    }
//	}
}

DirichletTransitionsSparse::~DirichletTransitionsSparse()
{
#if 0
	for (int i=0; i<n_states; ++i) {
		for (int a=0; a<n_actions; ++a) {
			int c = getCounts(i, a);
			if (c > 0) {
				printf ("# visits: (%d %d) %d\n", i, a, c);
			}
		}
	}
#endif
}

real DirichletTransitionsSparse::Observe(int state, int action, int next_state)
{
	DiscreteStateAction SA(state, action);
	auto got = P.find(SA);
	if (got == P.end()) {
		// arrgh C++
		return P.insert(std::make_pair(SA, DirichletFiniteOutcomes(n_states, prior_mass))).first->second.Observe(next_state);
	} else {
		return got->second.Observe(next_state);
	}
}


int DirichletTransitionsSparse::marginal_generate(int state, int action) const
{
	return DiscreteDistribution::generate(getMarginal(state, action));
}


Vector DirichletTransitionsSparse::generate(int state, int action) const
{
	auto got = P.find(DiscreteStateAction(state, action));
	if (got == P.end()) {
#ifdef TBRL_DEBUG2
std::printf("inside dirichlet transitions - 1\n");
#endif
		Vector p(n_states);
		if (uniform_unknown) {
			real z = 1.0 / (real) n_states;
			for (int j=0; j<n_states; j++) {
				p(j) = z;
			}
		} else {
			p(state) = 1;
		}
		return p;
	} 
#ifdef TBRL_DEBUG2
std::printf("inside dirichlet transitions - 2\n");
#endif
	return got->second.generate();
}

Vector DirichletTransitionsSparse::getMarginal(int state, int action) const
{
	auto got = P.find(DiscreteStateAction(state, action));
	if (got == P.end()) {
		Vector p(n_states);
		if (uniform_unknown) {
			real z = 1.0 / (real) n_states;
			for (int j=0; j<n_states; j++) {
				p(j) = z;
			}
		} else {
			p(state) = 1;
		}
		return p;
	} 
	return got->second.getMarginal();
}

Vector DirichletTransitionsSparse::getParameters(int state, int action) const
{
	auto got = P.find(DiscreteStateAction(state, action));
	if (got == P.end()) {
		Vector p(n_states);
		if (uniform_unknown) {
			real z = prior_mass;
			for (int j=0; j<n_states; j++) {
				p(j) = z;
			}
		} else {
			p(state) = prior_mass;
		}
		return p;
	} 
	return got->second.getParameters();
}


/// Get the marginal probability of the next state
real DirichletTransitionsSparse::marginal_pdf(int state, int action, int next_state) const
{
	auto got = P.find(DiscreteStateAction(state, action));
	if (got == P.end()) {
		if (uniform_unknown) {
printf("\nreturning this %d\n",n_states); disco
			return 1.0 / n_states;
		} else {
			if (next_state == state) {
				return 1.0;
			} else {
				return 0.0;
			}
		}
	} 
printf("\nreturning that\n");what ever
	return got->second.marginal_pdf(next_state);
}

int DirichletTransitionsSparse::getCounts(int state, int action) const
{
	DiscreteStateAction SA(state, action);
	auto got = P.find(SA);
	if (got == P.end()) {
		return 0;
	} else {
		return got->second.getCounts();
	}
}
