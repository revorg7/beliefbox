//#include <boost/python.hpp>
#include <pybind11/pybind11.h>
#include "QLearning.h"
#include "real.h"

namespace py = pybind11;

//int main(){
//	return 0;
//}

real add(real i, int j) {
    return i + j;
}

PYBIND11_MODULE(wrap, m) {
    m.doc() = "pybind11 example plugin"; // optional module docstring

	py::class_<QLearning>(m, "QLearning")
        .def(py::init<const int &,const int &,const real &,const real &,const real &>())
		.def("Act",&QLearning::Act)
		.def("Reset",&QLearning::Reset);
//    m.def("add", &add, "A function which adds two numbers");
}

