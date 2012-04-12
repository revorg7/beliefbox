/* -*- Mode: C++; -*- */
// copyright (c) 2011 by Christos Dimitrakakis <christos.dimitrakakis@gmail.com>
/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ***************************************************************************/

#ifdef MAKE_MAIN
#include "Random.h"
#include <vector>
#include "EasyClock.h"
#include "NormalDistribution.h"
#include "SingularDistribution.h"
#include "BetaDistribution.h"


void TestData(std::vector<real>& data,
                    ConjugatePrior& prior);

int main (int argc, char** argv)
{
    BetaDistribution beta;
    NormalUnknownMeanPrecision beta;
    
};

#endif
