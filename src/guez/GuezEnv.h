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

#ifndef GUEZENV_H
#define GUEZENV_H


#include "grid.h"
#include "Environment.h"

class GuezEnv : public Environment<int, int>
{
public:

	GuezEnv(uint S_,real discount_);
	virtual ~GuezEnv();
	virtual void Reset();
    //int getState(int x, int y) const; << wont be overridden, so need to set state manually
    //real getReward() const;	<< wont be overridden, so need to set reward manually
    virtual bool Act(const int& action);
	//virtual real getTransitionProbability(const int& state, const int& action, const int& next_state) const; << Not implemented
    virtual real getExpectedReward(const int& state, const int& action) const;

protected:
	
	Grid env;

};
#endif
