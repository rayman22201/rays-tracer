//Implements a Barth Decic Surface using Ray Marching
//Allows for both bisection and Ridders method
//Ray Imber a.k.a Rayman22201

#ifndef __MARCHINGBLOBBY__
#define __MARCHINGBLOBBY__

#include <cuda_runtime.h>
#include "rayObject.cuh"
#include "surfaceObject.cuh"
#include "vector.cu"
#include "ArrayObject.cuh"
//#include <math.h>

#define MIN(x,y) (!(y<x)?x:y)
#define MAX(x,y) (!(y>x)?x:y)
#define SIGN(x) (!(x>0)?-1:1)

namespace wildDoughnut {

	class MarchingBlobby : public surfaceObject<MarchingBlobby> {
	protected:
		//--------------------Surface Evaluation Functions-------------------------//
		//Note: I forgot to include the parameter, so w is Hard coded to be 1.0, Oh well...
		__host__ __device__ inline float surfaceAtPoint( Vector point )
		{
			//Vector rotated = Vector( (point.values.x), (point.values.y * cosf(45) + point.values.z * -1.0f * sinf(45)), (point.values.y * sinf(45) + point.values.z * cosf(45)));
			Vector pt = (point - this->position);
			Vector paramPt = (pt * this->param1);
			return (pt.dot(pt) + sinf(paramPt.values.x) - cosf(paramPt.values.y) + sinf(paramPt.values.z) - this->param2);
		}
		
		__host__ __device__ inline Vector gradientAtPoint( Vector point )
		{
			//Vector rotated = Vector( (point.values.x), (point.values.y * cosf(45) + point.values.z * -1.0f * sinf(45)), (point.values.y * sinf(45) + point.values.z * cosf(45)));
			Vector pt = (point - this->position);
			Vector paramPt = (pt * this->param1);
			return Vector((this->param1 * cosf(paramPt.values.x) + (2.0f * pt.values.x)),(this->param1 * cosf(paramPt.values.y) + (2.0f * pt.values.y)),(this->param1 * cosf(paramPt.values.z) + (2.0f * pt.values.z)));
		}

		__host__ __device__ inline float derivativeAtPoint( rayObject* ray, Vector point )
		{
			Vector gradient = this->gradientAtPoint( point );
			return ( gradient.dot( ray->direction ) );
		}
		//------------------------------------------------------------------------//

		//--------------------Interval Extension Tests----------------------------//
		__host__ __device__ inline int signTest( Vector intervalStart, Vector intervalEnd)
		{
			float rangeStart, rangeEnd;
			
			rangeStart = this->surfaceAtPoint( intervalStart );
			rangeEnd = this->surfaceAtPoint( intervalEnd );
			
			if( (rangeStart * rangeEnd) < 0 ) { return 1; }
			return 0;
		}

		__host__ __device__ inline int taylorTest( rayObject* ray, float t_start, float t_end)
		{
			float rangeStart, rangeEnd, taylorStart, taylorEnd, intervalExtensionStart, intervalExtensionEnd;
			Vector vectorStart, vectorEnd;

			vectorStart = ray->paramToVector( t_start );
			vectorEnd = ray->paramToVector( t_end );

			rangeStart = this->surfaceAtPoint( vectorStart );
			rangeEnd = this->surfaceAtPoint( vectorEnd );

			taylorStart = ( rangeStart + ( this->derivativeAtPoint( ray, vectorStart ) * ( (t_end - t_start) / 2.0f ) ) );
			taylorEnd = ( rangeEnd + ( this->derivativeAtPoint( ray, vectorEnd ) * ( (t_end - t_start) / 2.0f ) ) );

			intervalExtensionStart = MIN( MIN(rangeEnd,taylorEnd), MIN(rangeStart, taylorStart) );
			intervalExtensionEnd = MAX( MAX( rangeStart, taylorStart), MAX(rangeEnd, taylorEnd) );

			if( (intervalExtensionStart * intervalExtensionEnd) < 0 ){ return 1; }
			return 0;
		}
		//-----------------------------------------------------------------------//

		//--------------------Numerical Approximation Methods--------------------//
#define TOLERANCE 0.00001

		__host__ __device__ inline float bisectionMethod( rayObject* ray, float intervalStart, float intervalEnd)
		{
			float currentStart, currentMid, currentEnd, valueAtStart, rootApprox;
			Vector vectorStart, vectorMid, vectorEnd;

			currentStart = intervalStart;
			currentEnd = intervalEnd;
			for(int i = 0; i < this->numIterations; i++)
			{
				currentMid = ( (currentStart + currentEnd) / 2.0f );
				vectorStart = ray->paramToVector( currentStart );
				vectorMid = ray->paramToVector( currentMid );
				vectorEnd = ray->paramToVector( currentEnd );
				rootApprox = this->surfaceAtPoint( vectorMid );

				//check if we found the root
				if( (rootApprox == 0.0f) || ( ( abs(currentEnd - currentStart) / 2.0f ) < TOLERANCE ) )
				{
					//double check that it is not a false root
					//**Note to self: This takes a lot of extra computation and may not be strictly necessary. Though it helps maintain accuracy.
					if( signTest( vectorStart, vectorEnd ) )
					{
						return currentMid;
					}
					else
					{
						return -1;
					}
				}

				//compute the next iteration
				valueAtStart = surfaceAtPoint( vectorStart );

				if( (rootApprox * valueAtStart) > 0 ) //Start and Mid have the same sign
				{
					currentStart = currentMid;
				}
				else
				{
					currentEnd = currentMid;
				}
			}
			return currentMid; //No root was found within the Tolerance
		}

		__host__ __device__ inline float riddersMethod( rayObject* ray, float intervalStart, float intervalEnd)
		{
			float currentMid, rootApprox, valueAtStart, valueAtEnd, valueAtMid, riddersMid, riddersSign;
			float currentStart = intervalStart;
			float currentEnd = intervalEnd;
			Vector vectorStart, vectorEnd, vectorMid, vectorRidders;

			for(int i = 0; i < this->numIterations; i++)
			{
				currentMid = ( (currentStart + currentEnd) / 2.0f );
				vectorStart = ray->paramToVector( currentStart );
				vectorMid = ray->paramToVector( currentMid );
				vectorEnd = ray->paramToVector( currentEnd );

				valueAtStart = this->surfaceAtPoint( vectorStart );
				valueAtMid = this->surfaceAtPoint( vectorMid );
				valueAtEnd = this->surfaceAtPoint( vectorEnd );

				riddersSign = SIGN( (valueAtStart - valueAtEnd) );
		
				//x4 in Ridders' Method
				//    This is the crux of the method. This formula comes from the e^ax * f(x) which produces a straight line.
				//
				//    x4 = x3 + (x3 - x1) *  sign[ f(x1) - f(x2) ] * f(x3)
				//                          --------------------------------
				//                           sqrt( f(x3)^2 - f(x1) * f(x2) )
				//
				riddersMid = ( currentMid + ( (currentMid - currentStart) * ( (riddersSign * valueAtMid) / sqrtf( ((valueAtMid * valueAtMid) - (valueAtStart * valueAtEnd)) ) ) ) );
				vectorRidders = ray->paramToVector( riddersMid );

				rootApprox = this->surfaceAtPoint( vectorRidders );

				//check if we found the root
				if( (rootApprox == 0.0f) || ( (abs(currentEnd - currentStart) / 2.0f) < TOLERANCE ) )
				{
					//check for a false positive
					if( signTest( vectorStart, vectorEnd ) )
					{
						return riddersMid;
					}
					else
					{
						return -1;
					}
				}

				//compute the next iteration
				if( (rootApprox * valueAtMid) < 0 ) //riddersMid and Mid have opposite sign
				{
					currentStart = MIN(riddersMid,currentMid);
					currentEnd = MAX(riddersMid,currentMid);
				}
				else
				{
					if( (rootApprox * valueAtStart) < 0 )//riddersMid and Start have opposite sign
					{
						currentStart = MIN(riddersMid, currentStart);
						currentEnd = MAX(riddersMid, currentStart);
					}
					else
					{
						currentStart = MIN(riddersMid, currentEnd);
						currentEnd = MAX(riddersMid, currentEnd);
					}
				}
			}
			return -1; //No root found within the tolerance
		}
		//-----------------------------------------------------------------------//

		//--------------------Ray Marching Functions-----------------------------//
		__host__ __device__ inline int naiveMarchA( rayObject* ray, float* t_intersect, Vector* normal)
		{
			float currentStart, currentEnd, rootApprox;
			Vector vectorStart, vectorEnd, vectorApprox;
			//Vector vectorApprox;

			currentStart = ray->tmin;
			currentEnd = currentStart + this->marchInterval;
			rootApprox = -1;

			while( currentEnd < ray->tmax )
			{
				vectorStart = ray->paramToVector( currentStart );
				vectorEnd = ray->paramToVector( currentEnd );
				//test the interval
				if( this->signTest( vectorStart, vectorEnd ) )
				{
					rootApprox = this->bisectionMethod( ray, currentStart, currentEnd);
					if( rootApprox != -1 )
					{
						vectorApprox = ray->paramToVector( rootApprox );
						(*normal) = this->gradientAtPoint( vectorApprox );
						(*t_intersect) = rootApprox;
						(*normal) = normal->normalize();
						return 1;
					}
				}
				currentStart += this->marchInterval;
				currentEnd += this->marchInterval;
			}
			return 0;			
		}

		__host__ __device__ inline int naiveMarchB( rayObject* ray, float* t_intersect, Vector* normal)
		{
			float currentStart, currentEnd, rootApprox;
			//Vector vectorStart, vectorEnd, vectorApprox;
			Vector vectorApprox;

			currentStart = ray->tmin;
			currentEnd = currentStart + this->marchInterval;
			rootApprox = -1;

			while( currentEnd < ray->tmax )
			{
				//vectorStart = ray->paramToVector( currentStart );
				//vectorEnd = ray->paramToVector( currentEnd );
				//test the interval
				if( this->taylorTest( ray, currentStart, currentEnd ) )
				{
					rootApprox = this->bisectionMethod( ray, currentStart, currentEnd);
					if( rootApprox != -1 )
					{
						vectorApprox = ray->paramToVector( rootApprox );
						(*normal) = this->gradientAtPoint( vectorApprox );
						(*t_intersect) = rootApprox;
						(*normal) = normal->normalize();
						return 1;
					}
				}
				currentStart += this->marchInterval;
				currentEnd += this->marchInterval;
			}
			return 0;			
		}

		__host__ __device__ inline int naiveMarchC( rayObject* ray, float* t_intersect, Vector* normal)
		{
			float currentStart, currentEnd, rootApprox;
			Vector vectorStart, vectorEnd, vectorApprox;
			//Vector vectorApprox;

			currentStart = ray->tmin;
			currentEnd = currentStart + this->marchInterval;
			rootApprox = -1;

			while( currentEnd < ray->tmax )
			{
				vectorStart = ray->paramToVector( currentStart );
				vectorEnd = ray->paramToVector( currentEnd );
				//test the interval
				if( this->signTest( vectorStart, vectorEnd ) )
				{
					rootApprox = this->riddersMethod( ray, currentStart, currentEnd);
					if( rootApprox != -1 )
					{
						vectorApprox = ray->paramToVector( rootApprox );
						(*normal) = this->gradientAtPoint( vectorApprox );
						(*t_intersect) = rootApprox;
						(*normal) = normal->normalize();
						return 1;
					}
				}
				currentStart += this->marchInterval;
				currentEnd += this->marchInterval;
			}
			return 0;			
		}

		__host__ __device__ inline int naiveMarchD( rayObject* ray, float* t_intersect, Vector* normal)
		{
			float currentStart, currentEnd, rootApprox;
			//Vector vectorStart, vectorEnd, vectorApprox;
			Vector vectorApprox;

			currentStart = ray->tmin;
			currentEnd = currentStart + this->marchInterval;
			rootApprox = -1;

			while( currentEnd < ray->tmax )
			{
				//vectorStart = ray->paramToVector( currentStart );
				//vectorEnd = ray->paramToVector( currentEnd );
				//test the interval
				if( this->taylorTest( ray, currentStart, currentEnd ) )
				{
					rootApprox = this->riddersMethod( ray, currentStart, currentEnd);
					if( rootApprox != -1 )
					{
						vectorApprox = ray->paramToVector( rootApprox );
						(*normal) = this->gradientAtPoint( vectorApprox );
						(*t_intersect) = rootApprox;
						(*normal) = normal->normalize();
						return 1;
					}
				}
				currentStart += this->marchInterval;
				currentEnd += this->marchInterval;
			}
			return 0;			
		}
		//-----------------------------------------------------------------------//

	public:
		float param1;
		float param2;
		float marchInterval;
		unsigned int numIterations;
		
		//constructor
		__host__ __device__ MarchingBlobby() { position = Vector(0,0,0); marchInterval = 0.0f; numIterations = 10; param1 = 4.0f; param2 = 1.0f; }

		//Returns: 1 if the an intersection was found between ray.tmin and ray.tmax, 0 otherwise.
		//Note: CheckIntersection ALWAYS returns the lowest intersection point found in terms of t, the ray parameter.
		//
		//Side Effects: if 1 was returned: t_intersect will contain the t value where the interesection occured, t_intersect will be undefined otherwise.
		//              if 1 was returned: normal will contain the direction vector of the normal of the surface at the point of intersection, normal is undefined otherwise.
		__host__ __device__ int checkIntersection( rayObject* ray, float* t_intersect, Vector* normal)
		{
			return this->naiveMarchD( ray, t_intersect, normal );
		}
	};
}

#endif