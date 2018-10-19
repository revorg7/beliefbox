#ifndef DEARDENMAZE_H
#define DEARDENMAZE_H

#include <vector>

class DeardenMaze
{
public:
	DeardenMaze();
	virtual ~DeardenMaze();
	int *FreeY;
	std::vector<std::vector<int> > grid;

	int compress_flag(const std::vector<int>& flags_position) const;
	int compress(int flag, int x, int y) const;
	void uncompress(int state, int& flag, int& x, int& y) const;
};

#endif
