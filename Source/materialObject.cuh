//defines a generic material object
//    all other types of matrials should inherit from this class.
//    Inheritance is cool. Lets take some lessons from Java :-P
//
//By: Ray Imber a.k.a Rayman22201

#ifndef __MATERIALOBJECT__
#define __MATERIALOBJECT__

#include <cuda_runtime.h>
#include "rayObject.cuh"
#include "lightObject.cuh"
#include "vector.cu"
#include "ArrayObject.cuh"

namespace wildDoughnut {
	
	template <typename T>
	class materialObject {
	public:
		//constructor
		__host__ __device__ materialObject() { return; }

		//A Material basically has one function. Given a ray, an intersection point, and a surface normal
		//the function modifies the ray data in some way. Typically this means changing color data, but it could be anything.
		//Returns: void.
		//Side Effects: some modification to a public data member of ray.
		__host__ __device__ void valueAtIntersection( rayObject* ray, float t_intersect, Vector* surfaceNormal, lightObject* light ){ T* ptr = static_cast<T*>(this); ptr->valueAtIntersection( ray, t_intersect, surfaceNormal, light); } //  --virual
	};

}

#endif