/* -*- Mode: C++; -*- */
/* VER: $Id: Distribution.h,v 1.3 2006/11/06 15:48:53 cdimitrakakis Exp cdimitrakakis $*/
// copyright (c) 2010 by Christos Dimitrakakis <christos.dimitrakakis@gmail.com>
/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ***************************************************************************/

#ifdef MAKE_MAIN
#include "ContextTree.h"
#include "Random.h"
#include "RandomNumberGenerator.h"
#include "MersenneTwister.h"
#include "ReadFile.h"
#include <ctime>

int main(int argc, char** argv)
{
	int depth = 16;
	int n_symbols = 2;
	MersenneTwisterRNG mt;
	RandomNumberGenerator* rng = &mt;
	rng->manualSeed(123456791);
	int T = 10000;
	std::vector<int> data(T);
	if (argc==1) {
		for (int t=0; t<T; ++t) {
			data[t] = rng->discrete_uniform(n_symbols);
		} 
	} else if (argc>=2) {
		if (argc==3) {
			T = atoi(argv[2]);
		} else {
			T = 0;
		}
		n_symbols = FileToIntVector(data, argv[1], T);
		T = data.size();
	} 
	
	ContextTree tree(n_symbols, n_symbols, depth);
	std::cout << std::endl;
	for (int t=0; t<T; ++t) {
		int x = data[t];
		tree.Observe(x, x);
	}
	tree.Show();
	

return 0;
}

#endif
