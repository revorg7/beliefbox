#include "DeardenMaze.h"
#include <algorithm>

DeardenMaze::DeardenMaze()
{
	FreeY = new int[7];
	FreeY[0] = 5;
	FreeY[1] = 3;
	FreeY[2] = 6;
	FreeY[3] = 6;
	FreeY[4] = 4;
	FreeY[5] = 5;
	FreeY[6] = 4;
		
	grid.resize(7);
	grid[0] = {1, 1, 1, 0, 1, 1};
	grid[1] = {0, 0, 1, 0, 1, 1};
	grid[2] = {1, 1, 1, 1, 1, 1};
	grid[3] = {1, 1, 1, 1, 1, 1};
	grid[4] = {0, 0, 1, 1, 1, 1};
	grid[5] = {1, 1, 1, 0, 1, 1};
	grid[6] = {1, 1, 1, 0, 1, 0};

}

DeardenMaze::~DeardenMaze()
{
	delete [] FreeY;
}

int DeardenMaze::compress_flag(const std::vector<int>& flags_position) const{
	int size = flags_position.size();
	if (size==0) return 0;
	else if (size==1){
		if (flags_position[0]==2) return 1;
		else if (flags_position[0]==34) return 2;
		else if (flags_position[0]==35) return 3;
	}
	else if (size==2){
		if (std::find(flags_position.begin(), flags_position.end(), 35) == flags_position.end()) return 4;
		else if (std::find(flags_position.begin(), flags_position.end(), 34) == flags_position.end()) return 5;
		else if (std::find(flags_position.begin(), flags_position.end(), 2) == flags_position.end()) return 6;
	}
	else if (size==3){
		return 7;
	}
	else{
		throw("Wrong input to compress_flag in DeardenMaze\n");
	}
	return 0;
}
int DeardenMaze::compress(int flag, int x, int y) const{		
	int l = 0;
	for(int i=0;i<x;++i){
		l += FreeY[i];
	}
	for(int i=0;i<y;++i){
		l += grid[x][i];
	}
	int state = l*8+flag; //8 possible flag
	return state;
}
void DeardenMaze::uncompress(int state, int& flag, int& x, int& y) const{
	int l = state/8;
	flag = state - l*8;
	int sum = 0;
	for(int i=0;i<7;++i){
		for(int j=0;j<6;++j){
			if(sum == l && grid[i][j] != 0){
				x = i;
				y = j;
				return;
			}
			sum += grid[i][j];
		}
	}
}
