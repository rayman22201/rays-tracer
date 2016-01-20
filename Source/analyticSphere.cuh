//defines a Sphere through purely Analytic root finding.
//    This is used for setting the baseline for benchmarking
//    Inheritance is cool. Lets take some lessons from Java :-P
//
//By: Ray Imber a.k.a Rayman22201

#ifndef __ANALYTICSPHERE__
#define __ANALYTICSPHERE__

#include <cuda_runtime.h>
#include "rayObject.cuh"
#include "surfaceObject.cuh"
#include "vector.cu"
#include "ArrayObject.cuh"

namespace wildDoughnut {

	//This class actually has some methods associated with it. Surfaces must have an intersection checking method.
	class analyticSphere : public surfaceObject<analyticSphere> {
	public:
		float radius; //radius of the sphere
		
		//constructor
		__host__ __device__ analyticSphere() { position = Vector(0,0,0); radius = 0; }

		//Returns: 1 if the an intersection was found between ray.tmin and ray.tmax, 0 otherwise.
		//Note: CheckIntersection ALWAYS returns the lowest intersection point found in terms of t, the ray parameter.
		//
		//Side Effects: if 1 was returned: t_intersect will contain the t value where the interesection occured, t_intersect will be undefined otherwise.
		//              if 1 was returned: normal will contain the direction vector of the normal of the surface at the point of intersection, normal is undefined otherwise.
		__host__ __device__ int checkIntersection( rayObject* ray, float* t_intersect, Vector* normal)
		{
			float rSquared = radius * radius;

			float a = ray->direction.dot( ray->direction );

			Vector b2 = (ray->origin - this->position);
			float b = 2.0f * b2.dot( ray->direction );

			float c = ( (b2.dot( b2 ) ) - rSquared );

			float discriminant = ((b * b) - (4.0f * a * c));

			if(discriminant > 0) //positive root, there actually is a hit
			{
				//solve the roots
				float t0 = (((-1.0f * b) + sqrtf(discriminant)) / (2.0f * a));
				float t1 = (((-1.0f * b) - sqrtf(discriminant)) / (2.0f * a));

				Vector intersection;

				//check if the intersection is within the range of the Ray scope
				if( (t0 > ray->tmin) && (t0 < ray->tmax) && (t0 < t1) )
				{
					(*t_intersect) = t0;
					intersection = ray->paramToVector( t0 );
					(*normal) = ((intersection - this->position) * 2.0f);
					(*normal) = normal->normalize();
					//(*normal) = (*normal) * -1.0f;
					return 1;
				}
				else if( (t1 > ray->tmin) && (t1 < ray->tmax) && (t1 < t0) )
				{
					(*t_intersect) = t1;
					intersection = ray->paramToVector( t1 );
					(*normal) = ((intersection - this->position) * 2.0f);
					(*normal) = normal->normalize();
					//(*normal) = (*normal) * -1.0f;
					return 1;
				}
			}
			return 0; //no intersection
		}
	};

}

#endif