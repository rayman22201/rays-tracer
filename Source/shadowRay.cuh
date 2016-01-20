//defines a Shadow Ray object
//    Defines a Ray object that handles shadow attenuation information.
//    Inheritance is cool. Lets take some lessons from Java :-P
//
//By: Ray Imber a.k.a Rayman22201

#ifndef __SHADOWRAY__
#define __SHADOWRAY__

#include <cuda_runtime.h>
#include "rayObject.cuh"
#include "vector.cu"
#include "ArrayObject.cuh"

namespace wildDoughnut{

	//It is pretty much just a struct. but classes allow me to do inheritance, which is a more elegant way to have permutations of a data type.
	class shadowRay : public rayObject {
	public:
		Vector attenuation; //How much the ray has lost intensity. Making it a float3 allows for refraction and caustic effects. (0,0,0) means the Ray is in complete shadow.
		
		//constructor
		__host__ __device__ shadowRay() { attenuation = Vector(); }
	};

}

#endif