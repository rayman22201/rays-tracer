//defines a generic Light object
//    all other types of Lights should inherit from this class.
//    Inheritance is cool. Lets take some lessons from Java :-P
//
//By: Ray Imber a.k.a Rayman22201

#ifndef __LIGHTOBJECT__
#define __LIGHTOBJECT__

#include <cuda_runtime.h>
#include "vector.cu"
#include "ArrayObject.cuh"

namespace wildDoughnut {

	//It is pretty much just a struct. but classes allow me to do inheritance, which is a more elegant way to have permutations of a data type.
	class lightObject {
	public:
		Vector position;
		int castShadows;

		//constructor
		__host__ __device__ lightObject() { position = Vector(); castShadows = 0; }
	};

}

#endif