/// Compilation test.
#include "Gridworld.h"
#include "Mazeworld.h"
#include "DeardenMaze.h"
/// STD
#include <iostream>
#include <memory>
#include <algorithm> //for shuffle
using namespace std;


int main(void)
{
    
    //shared_ptr<DiscreteEnvironment> environment;
    //environment = make_shared<DiscreteChain>(n_states);
    //environment = make_shared<DoubleLoop>();
    //environment = make_shared<OptimisticTask>(0.1,0.7); //2nd argument is success probablity of transition
    //environment = make_shared<Gridworld>("../../../dat/maze01",0.2,0,1.0,0); //For GRID5,GRID10
    Mazeworld environment = Mazeworld("../../../dat/maze02",0.1,0,0.0,0); //For Maze

	printf("checking flags %d %d %d are at correct position, should be no. 3\n",environment.whatIs(0,5),environment.whatIs(6,4),environment.whatIs(2,0));
	printf("checking flags state-value %d %d %d\n",environment.getState(0,5),environment.getState(6,4),environment.getState(2,0));

	std::vector<int> flags_position;
	flags_position.push_back(2);
	flags_position.push_back(34);
	flags_position.push_back(35);

	DeardenMaze helper = DeardenMaze();
	printf("flag status is %d\n",helper.compress_flag(flags_position));

    return 0;
}
