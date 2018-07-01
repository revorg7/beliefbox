/// The main thing to test
#include "DiscreteMDPCounts.h"
#include <stdio.h>
//#include "MDPModel.h"


int main(int argc, char** argv) {

	enum DiscreteMDPCounts::RewardFamily reward_prior = DiscreteMDPCounts::BETA;

	int n_states = 2;
	int n_actions = 	1;
	DiscreteMDPCounts* belief1 =  new DiscreteMDPCounts(n_states, n_actions, 0.2, reward_prior);
	belief1->AddTransition(0,0,.1,0);
//	belief1->ShowModelStatistics();

        DiscreteMDPCounts* belief2 =  new DiscreteMDPCounts(n_states, n_actions, 0.5, reward_prior);
//	belief2->ShowModelStatistics();        

	std::vector<DiscreteMDPCounts*> beliefs;
	beliefs.push_back(belief1);
	beliefs.push_back(belief2);


        DiscreteMDPCounts* mean_belief =  new DiscreteMDPCounts(n_states, n_actions, beliefs);
//	mean_belief->ShowModelStatistics();        

        MDPModel* model1 = (MDPModel*) belief1;
	MDPModel* model2 = (MDPModel*) belief2;

	std::vector<MDPModel*> models;
	models.push_back(model1);
	//beliefs.push_back(belief2);
        DiscreteMDPCounts* mean_belief1 =  new DiscreteMDPCounts(n_states, n_actions, models);

    return 0;
}

