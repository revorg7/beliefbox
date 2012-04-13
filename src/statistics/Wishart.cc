/* -*- Mode: C++; -*- */
// copyright (c) 2012 by Christos Dimitrakakis <christos.dimitrakakis@gmail.com>
/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ***************************************************************************/

#include "Wishart.h"
#include "Matrix.h"

Wishart::Wishart()
    : k(1),
      n(1),
      V(Matrix::Unity(1,1))
{
    
}

Wishart::Wishart(real n_, const Matrix& V_)
    : k(V_.Rows()),
      n(n_),
      V(V_)
{
    assert(V.Rows() == V.Columns());
}

Wishart::~Wishart()
{
    Serror("Not implemented\n");
}
void Wishart::generate(Matrix& x) const
{
    Serror("Not implemented\n");
}

Matrix Wishart::generate() const
{
    Serror("Not implemented\n");
    return Vector(1);
}

real Wishart::pdf(const Matrix& x) const
{
    Serror("Not implemented\n");
    return 0.0;
}
