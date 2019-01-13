#pragma once

#include "simulator.h"

class Chain : public SIMULATOR
{
public:
	const uint S = 5;
	const uint A = 2;
	const uint SA = S*A;

	Chain(double discount);
	inline int encode(uint s, uint a){
		return (s*A + a);
	}

	inline int encode(uint s, uint a, uint S_next){
		if (a==1) return S*((s*A) + 1) + S_next;
		else return S*(s*A) + S_next;
	}

	void setMDP(double* R, double* T);

    virtual uint CreateStartState() const;
    virtual bool Step(uint state, uint action, 
        uint& observation, double& reward) const;

};
