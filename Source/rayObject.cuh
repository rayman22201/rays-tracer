//defines a generic Ray object
//    all other types of Rays should inherit from this class.
//    Inheritance is cool. Lets take some lessons from Java :-P
//
//By: Ray Imber a.k.a Rayman22201

#ifndef __RAYOBJECT__
#define __RAYOBJECT__

#include <cuda_runtime.h>
#include "vector.cu"
#include "ArrayObject.cuh"

namespace wildDoughnut {

	//It is pretty much just a struct. but classes allow me to do inheritance, which is a more elegant way to have permutations of a data type.
	class rayObject {
	public:
		Vector origin;
		Vector direction;
		float tmin; //min clipping plane
		float tmax; //far clipping plane

		//constructor
		__host__ __device__ rayObject() { origin = Vector(); direction = Vector(); tmin = 0.0f; tmax = 0.0f; }

		__host__ __device__ Vector paramToVector( float t ){ return ( this->origin + (this->direction * t) ); } 
	};
}

#endif