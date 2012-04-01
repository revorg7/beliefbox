// -*- Mode: C++; -*-
// copyright (c) 2012 by Christos Dimitrakakis <christos.dimitrakakis@gmail.com>
/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ***************************************************************************/

#ifdef MAKE_MAIN

#include "Matrix.h"
#include "Dirichlet.h"
#include "Sarsa.h"
#include "MersenneTwister.h"
#include "Bounds.h"

class ObjectiveBanditPolicy
{
public:
	const int n_actions;
	const int n_outcomes;
	RandomNumberGenerator& rng;
	ObjectiveBanditPolicy(int n_actions_, 
						  int n_outcomes_, 
						  RandomNumberGenerator& rng_)
		: n_actions(n_actions_), n_outcomes(n_outcomes_), rng(rng_)
	{
	}
	virtual ~ObjectiveBanditPolicy()
	{}
	/// get the greedy action
	virtual int getGreedyAction(const Vector& payoff) const = 0;
	/// act
	virtual int Act(const Vector& payoff, int outcome) = 0;
};



class EpsilonGreedyObjectiveBandit : public ObjectiveBanditPolicy
{
protected:
	int action;
public:
	Matrix N; ///< probability matrix
	Matrix P; ///< probability matrix
	real epsilon; ///< randomness
	EpsilonGreedyObjectiveBandit(int n_actions_,
								 int n_outcomes_,
								 RandomNumberGenerator& rng_,
								 real epsilon_)
		: ObjectiveBanditPolicy(n_actions_, n_outcomes_, rng_),
		  action(-1),
		  N(n_actions, n_outcomes),
		  P(n_actions, n_outcomes),
		  epsilon(epsilon_)
	{
		real p = 1.0 / (real) n_outcomes;
		for (int i=0; i<n_actions; ++i) {
			for (int j=0; j<n_outcomes; ++j) {
				N(i,j) = 0.5;
				P(i,j) = p;
			}
		}
		assert(epsilon >= 0.0 && epsilon <= 1.0);
	}

	virtual ~EpsilonGreedyObjectiveBandit()
	{
	}

	/// get the greedy action
	virtual int getGreedyAction(const Vector& payoff) const
	{
		const Matrix& rP = P;
		return ArgMax(rP * payoff);
	}

	/// act
	virtual int Act(const Vector& payoff, int outcome)
	{
		// use previous action
		if (action >= 0 && outcome >= 0) {
			N(action, outcome)++;
			Vector p = N.getRow(action);
			P.setRow(action, p / p.Sum());
		}

		if (rng.uniform() < epsilon) {
			action = rng.discrete_uniform(n_actions);
		} else {
			action = getGreedyAction(payoff);
		}
		return action;
	}
};


class WeissmanObjectiveBandit : public ObjectiveBanditPolicy
{
protected:
	int action;
public:
	Matrix N; ///< incidence matrix
	Matrix P; ///< probability matrix
	real delta; ///< randomness
	int T; ///< number of time-steps
	std::vector<int> plays; ///< number of plays
	WeissmanObjectiveBandit(int n_actions_,
							int n_outcomes_,
							RandomNumberGenerator& rng_)
		: ObjectiveBanditPolicy(n_actions_, n_outcomes_, rng_),
		  action(-1),
		  N(n_actions, n_outcomes),
		  P(n_actions, n_outcomes),
		  delta(1.0),
		  T(0),
		  plays(n_actions)
	{
		real p = 1.0 / (real) n_outcomes;
		for (int i=0; i<n_actions; ++i) {
			plays[i] = 0;
			for (int j=0; j<n_outcomes; ++j) {
				N(i,j) = 0.5;
				P(i,j) = p;
			}
		}
	}

	virtual ~WeissmanObjectiveBandit()
	{
	}

	Vector OptimisticTransition(const Vector& p, const Vector& r, real epsilon)
	{
		int m = p.Size();
		assert(m == r.Size());
		real best_gain = 0.0;
		Vector best_vector = p;
		//printf("P: "); p.print(stdout);
		for (int i=0; i<m; ++i) {
			real max_gap = 1.0 - p(i);
			if (max_gap > 0.5 * epsilon) {
				max_gap = 0.5 * epsilon;
			}
			real min_gap = 1.0;
			for (int j=0; j<m; ++j) {
				if (i == j) continue;
				if (min_gap > p(j)) {
					min_gap = p(j);
				}
			}
			real u = 0.5 * epsilon;
			real s = u / (real) (m - 1);
			if (min_gap > s) {
				min_gap = s;
			}
			Vector hp = p;
			hp(i) += max_gap;
			assert(hp(i) <= 1.0);
			for (int j=0; j<m; ++j) {
				if (i==j) continue;
				hp(j) -= min_gap;
				assert(hp(j) >= 0.0);
			}
			//			printf ("%f %f %f %f\n", max_gap, min_gap, max_gap + (real) (m-1)*min_gap, epsilon);
			//printf("H: "); hp.print(stdout);
			hp /= hp.Sum();
			real gain = Product(hp, r);
			if (gain > best_gain) {
				best_gain = gain;
				best_vector = hp;
			}
		}
		return best_vector;
	}

	/// get the greedy action
	virtual int getGreedyAction(const Vector& payoff) const
	{
		const Matrix& rP = P;
		return ArgMax(rP * payoff);
	}

	/// act
	virtual int Act(const Vector& payoff, int outcome)
	{
		// use previous action to update probabilities
		if (action >= 0 && outcome >= 0) {
			N(action, outcome)++;
			Vector p = N.getRow(action);
			P.setRow(action, p / p.Sum());
		}

		// choose next action
		real u_max = - 1;
		T++;
		delta = 1.0 / (real) T;
		for (int i=0; i<n_actions; ++i) {
			real epsilon = WeissmanBound(n_outcomes, plays[i], delta);
			Vector B = OptimisticTransition(P.getRow(i), payoff, epsilon);
			real u = Product(B, payoff);
			if (u > u_max) {
				u_max = u;
				action = i;
			}
		}

		return action;
	}
};

struct RegretPair
{
	real	average;
	real worst;
};

RegretPair getRegret(const Matrix& P, const ObjectiveBanditPolicy* policy)
{
	RegretPair regret;
	regret.average = 0.0;
	regret.worst = 0.0;
	//int n_actions = policy.n_actions;
	int n_outcomes = policy->n_outcomes;

	for (int i=0; i<n_outcomes; ++i) {
		Vector payoff(n_outcomes);
		payoff(i) = 1.0;
		int action = policy->getGreedyAction(payoff);
		const Vector& r_payoff = payoff;
		Vector U = P * r_payoff;
		real delta = Max(U) - U(action);
		regret.average += delta;
		regret.worst = std::max(regret.worst, delta);
	}
	regret.average /= (real) n_outcomes;
	return regret;
}


enum Method {
	UNDEFINED = 0x0,
	EPSILON_GREEDY,
	WEISSMAN_UCB
};

int main (int argc, char** argv)
{

    srand48(1228517343);
	MersenneTwisterRNG rng;
	rng.manualSeed(1228517343);
	setRandomSeed(1228517343);
    //RandomNumberFile rng("./dat/r1e7.bin");
	int n_actions = 4;
	int n_outcomes = 4;
	int horizon = 1000;
	int n_experiments = 10;
	real epsilon = 0.0;
	Method method = UNDEFINED;

	if (argc <= 6) {
		fprintf(stderr, "arguments: n_actions n_outcomes horizon n_runs method epsilon \n method 1: epsilon-greedy \n method 2: weissman_ucb");
		exit(-1);
	}
	n_actions = atoi(argv[1]);
	n_outcomes = atoi(argv[2]);
	horizon = atoi(argv[3]);
	n_experiments = atoi(argv[4]);
	method = (Method) atoi(argv[5]);
	epsilon = atof(argv[6]);
	
	Vector worst_case(horizon);
	Vector average_case(horizon);

	

    for (int experiment=0; experiment<n_experiments; experiment++) {

		Matrix P_sa(n_actions, n_outcomes);
		DirichletDistribution prior(n_outcomes);
		for (int i=0; i<n_actions; ++i) {
			P_sa.setRow(i, prior.generate());
		}
		ObjectiveBanditPolicy* policy = NULL;
		switch (method) {
		case EPSILON_GREEDY:
			policy = new EpsilonGreedyObjectiveBandit(n_actions, n_outcomes, rng, epsilon);
			break;
		case WEISSMAN_UCB:
			policy = new WeissmanObjectiveBandit(n_actions, n_outcomes, rng);
			break;
		default:
			fprintf(stderr, "Unknown method %d\n", method);
			exit(-1);
		}	
		real reward = 0.0;
		int outcome = -1;
        for (int t=0; t<horizon; t++) {
			Vector payoff = prior.generate();
            int action = policy->Act(payoff, outcome);
			//printf ("%d ", action);	payoff.print(stdout);
			outcome = DiscreteDistribution::generate(P_sa.getRow(action));
			reward = payoff(outcome);
			RegretPair regret =  getRegret(P_sa, policy);
			worst_case(t) += regret.worst;
			average_case(t) += regret.average;
		}
    }

	worst_case /= (real) n_experiments;
	average_case /= (real) n_experiments;
	for (int t=0; t<horizon; ++t) {
		printf ("%f %f\n", worst_case(t), average_case(t));
	}

    return 0;
}



#endif
