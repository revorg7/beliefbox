// -*- Mode: c++ -*-
// copyright (c) 2007 by Christos Dimitrakakis <christos.dimitrakakis@gmail.com>
// $Revision$
/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ***************************************************************************/

# include "GuezEnv.h"

GuezEnv::GuezEnv(uint S_,real discount_)
	: env(S_,discount_)
{
	n_actions = 4;
	n_states = S_*S_;	
//	env = Grid(S_,discount_);
	Reset();
}

GuezEnv::~GuezEnv()
{
//	delete env;
}

void GuezEnv::Reset()
{
	state = 0;
	reward = 0.0;
}

bool GuezEnv::Act(const int& action)
{
	uint observation;
	double reward1;
	env.Step(state,action,observation,reward1);
//printf("s,a,r,s: %d,%d,%f,%d\n",state,action,reward1,observation);
	reward = reward1;
	state = static_cast<int>(observation);
	return true;
}

real GuezEnv::getExpectedReward(const int& state, const int& action) const
{
	return env.R[state*4+action];
}


