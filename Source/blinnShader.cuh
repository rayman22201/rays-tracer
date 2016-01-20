//Blinn Shader is a basic material with all the standard components: lambert diffuse, Blinn-Phong Specular, Mirror Reflection, and Shadows
//
//By Ray Imber a.k.a Rayman22201

#ifndef __BLINNSHADER__
#define __BLINNSHADER__

#include <cuda_runtime.h>
#include "rayObject.cuh"
#include "lightObject.cuh"
#include "materialObject.cuh"
#include "vector.cu"
#include "ArrayObject.cuh"

#include "radianceRay.cuh"
#include "pointLight.cuh"

namespace wildDoughnut {

	class blinnShader : public materialObject<blinnShader> {
	public:
		float lambertCoeff;
		float specPower;
		float reflectance;
		Vector diffuseColor;
		Vector specColor;

		//constructor
		__host__ __device__ blinnShader()
		{ 
			lambertCoeff = 1.0f; 
			reflectance = 0.5f; 
			diffuseColor = Vector(0.5f,0.5f,0.5f); 
			specColor = Vector(1.0f,1.0f,1.0f); 
			specPower = 500.0f; 
			return;
		}

		//A Material basically has one function. Given a ray, an intersection point, and a surface normal
		//the function modifies the ray data in some way. Typically this means changing color data, but it could be anything.
		//Returns: void.
		//Side Effects: some modification to a public data member of ray.
		__host__ __device__ void valueAtIntersection( rayObject* ray, float t_intersect, Vector* surfaceNormal, lightObject* light )
		{
			radianceRay* rdRay = (radianceRay*)ray;
			pointLight* ptLight = (pointLight*)light;

			Vector finalColor;
			finalColor = Vector(0,0,0);
			//finalColor = finalColor + (this->diffuseColor * 0.1f);

			Vector intersection = rdRay->paramToVector( t_intersect );
			Vector lightDir = ( ptLight->position - intersection );
			lightDir = lightDir.normalize();

			finalColor = Vector(0.5f,0.2f,0.3f) + ( this->diffuseColor * (-0.8f * (ray->direction.dot( (*surfaceNormal) ))) );

			/*if( ( surfaceNormal->dot( lightDir ) < 0 ) )
			{
				rdRay->color = finalColor;
				return;
			}
			//lambert component
			float lambert = ( lightDir.dot( (*surfaceNormal) ) * (rdRay->importance) ); //cosine weighting
			Vector lightColor = (ptLight->color * ptLight->intensity);
			finalColor = finalColor + ( (this->diffuseColor * lightColor) * lambert );

			//specular component
			Vector blinnVector = (lightDir - rdRay->direction);
			float blinnSqrt = sqrtf( (blinnVector.dot( blinnVector )) );
			if(blinnSqrt != 0.0f)
			{
				blinnVector = blinnVector.normalize();

				float blinnTerm = 0.0f;
				float blinnCos = blinnVector.dot( (*surfaceNormal) );
				if( blinnCos > 0.0f )
				{
					blinnTerm = blinnCos;
				}
				
				if( this->specPower == 0.0f ) { blinnTerm = 0.0f; }
				else { blinnTerm = (powf(blinnTerm, this->specPower) * rdRay->importance); }
				//printf("blinnTerm %f\n",blinnTerm);
				finalColor = finalColor + (this->specColor * (blinnTerm) * ptLight->intensity );
			}
			
			//reduce the light energy by the reflection factor for the next reflection bounce
			rdRay->importance = rdRay->importance * this->reflectance;
			*/
			//apply the final color
			rdRay->color = finalColor;
			//rdRay->color = (((*surfaceNormal) * 0.5f) + 0.5f);
			return;
		}
	};

}

#endif