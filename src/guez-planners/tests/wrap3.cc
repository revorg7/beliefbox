#include <pybind11/pybind11.h>

#include "DiscreteMDPCountsSparse.h"
#include "MDPModel.h"

namespace py = pybind11;

class TestTreeBRL
{
public:
	TestTreeBRL() { belief = new DiscreteMDPCountsSparse(5,2);belief->AddTransition(0,1,0.5,0);belief->ShowModel();}

	//Nested class
    class BeliefState
    {
    protected:
        TestTreeBRL& tree; ///< link to the base tree
		MDPModel* belief; ///< current belief (simulated belief at each node)
    public:
        BeliefState(TestTreeBRL& tree_,
                    const MDPModel* belief_) : tree(tree_)
			{	belief = belief_->Clone();belief->AddTransition(0,0,1,4);belief->ShowModel();	}
	};
	//////

	void Act() {BeliefState belief_state(*this, belief);}
protected:
MDPModel* belief;	///< Tree belief linked to real-world

};

PYBIND11_MODULE(wrap3, m) {

 	py::class_<TestTreeBRL>(m, "testTree")
		.def(py::init<>())
		.def("act",&TestTreeBRL::Act);


//	py::class_<MDPModel>(m, "MDP");
// 	py::class_<DiscreteMDPCountsSparse>(m, "sparseMDP")
//		.def(py::init<int,int>())
//		.def("show",&DiscreteMDPCountsSparse::ShowModel)
//		.def("add",&DiscreteMDPCountsSparse::AddTransition)
//		.def("pclone",&DiscreteMDPCountsSparse::Clone);

}
