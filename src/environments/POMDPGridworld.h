// -*- Mode: c++ -*-
// copyright (c) 2009 by Christos Dimitrakakis <christos.dimitrakakis@gmail.com>
/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ***************************************************************************/

#ifndef POMDP_GRIDWORLD_H
#define POMDP_GRIDWORLD_H

#include "Environment.h"
#include <string>
#include <vector>



class POMDPGridworld : public Environment<int, int> {
public:
    uint ox, oy;
    int total_time;
    enum MapDirection {
        NORTH=0, SOUTH, EAST, WEST
    };
    enum MapElement {
        INVALID=-1, GRID, WALL, GOAL, PIT
    };
    DiscreteMDP* mdp;
    uint terminal_state;
    POMDPGridworld(const char* fname,
              uint height_,
              uint width_,
              uint n_actions_ = 4,
              real random_ = 0.0,
              real pit_ = -1.0,
              real goal_ = 0.0,
              real step_ = -0.1);
    virtual ~POMDPGridworld();


    virtual DiscreteMDP* getMDP() const
    {
        return mdp;
    }

    MapElement whatIs(int x, int y)
    {
        if (x>=0 && y >=0 && x< (int) width && y < (int) height) {
            return grid[x][y];
        } else {
            return INVALID;
        }
    }
    virtual void Reset();
    virtual bool Act(int action);
    void Show();
    int getState(int x, int y)
    {
        if (x>=0 && y >=0 && x< (int) width && y < (int) height) {
            return x + y*width;
        }
        return -1;
    };

    uint getWidth() const
    {
        return width;
    }

    uint getHeight() const
    {
        return height;
    }
    virtual const char* Name()
    {
        return "POMDPGridworld";
    }

protected:
    uint height;
    uint width;
    //uint n_aactions;
    real random;
    real pit_value;
    real goal_value;
    real step_value;
    std::vector< std::vector<MapElement> > grid;
    real** transitions;
    real* P_data;
    std::vector<Distribution*> rewards;
};

#endif
