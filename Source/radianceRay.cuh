//defines a Radiance Ray object
//    Defines a Ray object that handles color information.
//    Inheritance is cool. Lets take some lessons from Java :-P
//
//By: Ray Imber a.k.a Rayman22201

#ifndef __RADIANCERAY__
#define __RADIANCERAY__

#include <cuda_runtime.h>
#include "rayObject.cuh"
#include "vector.cu"
#include "ArrayObject.cuh"

namespace wildDoughnut {

	//It is pretty much just a struct. but classes allow me to do inheritance, which is a more elegant way to have permutations of a data type.
	class radianceRay : public rayObject {
	public:
		Vector color; //the resulting color of the radiance ray.
		float importance; //the reflective importance. Used for scaling when the ray is reflected.
		float depth; //keeps track of the recursion depth of the ray.

		//constructor
		__host__ __device__ radianceRay() { color = Vector(); importance = 0.0f; depth = 0.0f; }
	};

}

#endif