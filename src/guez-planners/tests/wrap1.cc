#include <pybind11/pybind11.h>
#include<real.h>
#include<Matrix.h>
#include "OnlineAlgorithm.h"
//#include "TreeBRLPolicyPython.h"


class TestClass : public OnlineAlgorithm<int, int>
{
public:
//	TestClass();
	virtual int Act(real reward, int next_state)
	{
		return 7;
	}

    virtual real Observe (real reward, int next_state, int next_action) {return 2;}
    virtual real getValue (int state, int action) {return 3;}

};

namespace py = pybind11;


PYBIND11_MODULE(wrap1, m) {

    py::class_<OnlineAlgorithm<int, int>> base(m, "base");
		base.def("Observe",&OnlineAlgorithm<int, int>::Observe);

/*
Note that I couldn't do this:
    py::class_<OnlineAlgorithm<int, int>> base(m, "base");
		base.def("Observe",&OnlineAlgorithm<int, int>::Observe);
Because giving a name of class is seperate act, so I followed the suggestion in "Misecelleneous" section of documentation
*/

 	py::class_<TestClass>(m, "derived", base)
		.def(py::init<>());
//		.def("Act",&TestClass::Act)
//		.def("Observe",&TestClass::Observe)		//This is easier than what I did in wrap.cc (exposing Qlearning diretly), since all that remains now to do is define a init<>() for each OnlineAlgorithm 
//		.def("getValue",&TestClass::getValue);
}

