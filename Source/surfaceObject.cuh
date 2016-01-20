//defines a generic Surface object
//    all other types of Surfaces should inherit from this class.
//    Inheritance is cool. Lets take some lessons from Java :-P
//
//By: Ray Imber a.k.a Rayman22201

#ifndef __SURFACEOBJECT__
#define __SURFACEOBJECT__

#include <cuda_runtime.h>
#include "rayObject.cuh"
#include "materialObject.cuh"
#include "vector.cu"
#include "ArrayObject.cuh"

namespace wildDoughnut {

	//This class actually has some methods associated with it. Surfaces must have an intersection checking method.
	template <typename T>
	class surfaceObject {
	public:
		Vector position; //all objects have a position, for more abstract or unbounded surfaces this is simply the origin point
		
		int material; //index into the material object array which contains a Pointer to a material object. All surfaces must have a material or shader associated with them. 

		//constructor
		__host__ __device__ surfaceObject() { position = Vector(); material = NULL; }

		//Returns: 1 if the an intersection was found between ray.tmin and ray.tmax, 0 otherwise.
		//Note: CheckIntersection ALWAYS returns the lowest intersection point found in terms of t, the ray parameter.
		//
		//Side Effects: if 1 was returned: t_intersect will contain the t value where the interesection occured, t_intersect will be undefined otherwise.
		//              if 1 was returned: normal will contain the direction vector of the normal of the surface at the point of intersection, normal is undefined otherwise.
		__host__ __device__ int checkIntersection( rayObject* ray, float* t_intersect, Vector* normal){ T* ptr = static_cast<T*>(this); return ptr->checkIntersection( ray, t_intersect, normal ); }
		
	};

}

#endif