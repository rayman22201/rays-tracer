//defines a point Light object
//    The most basic form of light. It just exists as a point (derived from the base class), and some color and intesity info.
//    Inheritance is cool. Lets take some lessons from Java :-P
//
//By: Ray Imber a.k.a Rayman22201

#ifndef __POINTLIGHT__
#define __POINTLIGHT__

#include <cuda_runtime.h>
#include "lightObject.cuh"
#include "vector.cu"
#include "ArrayObject.cuh"

namespace wildDoughnut {

	//It is pretty much just a struct. but classes allow me to do inheritance, which is a more elegant way to have permutations of a data type.
	class pointLight : public lightObject {
	public:
		Vector color;
		float intensity;

		//constructor
		__host__ __device__ pointLight() { color = Vector(); intensity = 0.0f; }

	};

}

#endif