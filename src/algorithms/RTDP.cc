// -*- Mode: c++ -*-
// copyright (c) 2006 by Christos Dimitrakakis <christos.dimitrakakis@gmail.com>
// $Revision$
/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 *   Created by - Divya Grover
 ***************************************************************************/

#include "RTDP.h"

RTDP::RTDP(DiscreteMDP* mdp, real gamma, int init_state_, real baseline)
{
    assert (mdp);
    assert (gamma>=0 && gamma <=1);
    this->mdp = mdp;
    this->gamma = gamma;
    this->baseline = baseline;
	init_state = init_state_;
    n_actions = mdp->getNActions();
    n_states = mdp->getNStates();
    Reset();
}

RTDP::~RTDP()
{
	delete mdp;
}

void RTDP::Reset()
{
    V.Resize(n_states);
    pV.Resize(n_states);

    Q.Resize(n_states, n_actions);
    pQ.Resize(n_states, n_actions);

    
    for (int s=0; s<n_states; s++) {
        pV(s) = 0.0;
		real max = mdp->getExpectedReward(s, 0);
        for (int a=0; a<n_actions; a++) {
			real val = mdp->getExpectedReward(s, a);
            Q(s, a) = 0.0;
            pQ(s, a) = 0.0;
			if (val > max) max = val;
        }
        V(s) = max;
    }

}

void RTDP::ComputeStateValues(real threshold, int max_iter)
{
	ComputeStateValues(20, n_states);
}

void RTDP::ComputeStateValues(int n_runs, int max_depth)
{
	for(int run=0; run<n_runs; run++){

		real ran = rand() / double(RAND_MAX);
		int s = init_state;
		if(ran<0.5){
			 mdp->Reset(init_state);
		}
		else{
			s = rand() % n_states;
			mdp->Reset(s);
		}

		for (int depth=0;depth<max_depth;depth++){
//printf("depth,state %d %d\n",depth,s);

			std::set<int> all_next_states;
			//Selecting best-action
            for (int a=0; a<n_actions; a++) {
                real V_next_sa = 0.0;
                const DiscreteStateSet& next = mdp->getNextStates(s, a);
                for (DiscreteStateSet::iterator i=next.begin();
                     i!=next.end();
                     ++i) {
                    int s2 = *i;
					all_next_states.insert(s2);
                    real P = mdp->getTransitionProbability(s, a, s2);
                    V_next_sa += P * V(s2);
//	printf("s2,P,V_next %d %f %f\n",s2,P,V_next_sa);
                }
                Q(s, a) = mdp->getExpectedReward(s, a) - baseline + gamma * V_next_sa;
//	printf("expR,Q,a %f %f %d\n",mdp->getExpectedReward(s, a),Q(s, a),a);
            }
			Vector max_vec = Q.getRow(s);
//for (int i = 0; i < max_vec.Size(); ++i) printf(" %f ",max_vec[i]); printf("\n");
//ola			std::vector<int> actions = ArgMaxs(max_vec);
//for (int i = 0; i < actions.size(); ++i) printf(" %d ",actions[i]); printf("\n");
//ola			int index = rand() % actions.size();
//ola            V(s) = Q(s,actions[index]);
//printf("V(s) new is ");
//for (int i = 0; i < n_states; ++i) printf(" %f ",V[i]); printf("\n");
//            Delta += fabs(V(s) - pV(s));

			//Acting in MDP
//printf(" action taken is:%d ran is:%f\n",actions[index],ran);
//			mdp->Reset((s+1)%5);
			if (run > int(0.7*n_runs)){
//				real ran = rand() / double(RAND_MAX);
//				if(ran<0.3) mdp->Act(rand()%n_actions);
//				else mdp->Act(actions[index]);

				std::vector<int> actions = ArgMaxs(max_vec);
				int index = rand() % actions.size();
				mdp->Act(actions[index]);
			}
			else {
				int next_state;
//				std::vector<int> indices = ArgMins(pQ.getRow(s));
//				if (indices.size()>1){
//					std::vector<int> indices_2;
//					real min = pV[indices[0]];
//					indices_2.push_back(indices[0]);
//					for(int i=1;i<indices.size();i++){
//						if (pV[indices[i]] < min ){
//							indices_2.clear();
//							min = pV[indices[i]];
//							indices_2.push_back(indices[i]);
//						}
//						else if (pV[indices[i]] == min) indices_2.push_back(indices[i]);
//					}
//					int index = rand() % indices_2.size();
//					next_state = indices_2[index];
//				}
//				else{ 
//					next_state = indices[0];
//				}
//				pQ(s,actions[index]) += 1;
//				pV(s)+=1;
//				mdp->Reset(next_state);

				//Copying set to vector
				std::vector<int> output(all_next_states.size());
				std::copy(all_next_states.begin(), all_next_states.end(), output.begin());
				
				std::vector<int> next_states;
				next_states.push_back(output[0]);
				real min = pV[output[0]];
					for(int i=1;i<output.size();i++){
						if (pV[output[i]] < min ){
							next_states.clear();
							min = pV[output[i]];
							next_states.push_back(output[i]);
						}
						else if (pV[output[i]] == min) next_states.push_back(output[i]);
					}
				int index = rand() % next_states.size();
				next_state = next_states[index];
				mdp->Reset(next_state);
			}

			V(s) = Max(max_vec);
			pV(s)+=1;
			s = mdp->getState();
		}
	}
}
