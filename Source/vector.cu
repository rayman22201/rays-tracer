//Defines a 3D Vector class with the barebones vector operations that I need. 
//    It is basically a wrapper around the built in float3 struct, with some vector operations associated with it.
//    I wish there was a better optimized library for this, but I can't find one atm :-(
//
//By Ray Imber a.k.a Rayman22201

#ifndef __VECTOROBJECT__
#define __VECTOROBJECT__

#include <cuda_runtime.h>

namespace wildDoughnut {
	
	class Vector {
	public:
		float3 values;

		//constructor
		__host__ __device__ Vector( float newX, float newY, float newZ ) { values.x = newX; values.y = newY; values.z = newZ; }

		__host__ __device__ Vector( float3 newValues ) { values = newValues; }

		__host__ __device__ Vector() { values.x = 0; values.y = 0; values.z = 0; }

		//math operations
		__host__ __device__ Vector operator+ ( Vector b ) const
		{
			return Vector( (*this).values.x + b.values.x , (*this).values.y + b.values.y, (*this).values.z + b.values.z );
		}

		__host__ __device__ Vector operator+ ( float b ) const
		{
			return Vector( (*this).values.x + b , (*this).values.y + b, (*this).values.z + b );
		}

		__host__ __device__ Vector operator- ( Vector b ) const
		{
			return Vector( (*this).values.x - b.values.x , (*this).values.y - b.values.y, (*this).values.z - b.values.z );
		}

		__host__ __device__ Vector operator* ( Vector b ) const
		{
			return Vector( (this->values.x * b.values.x) , (this->values.y * b.values.y), (this->values.z * b.values.z) );
		}

		__host__ __device__ Vector operator* ( float b ) const
		{
			return Vector( (*this).values.x * b , (*this).values.y * b , (*this).values.z * b );
		}

		__host__ __device__ Vector operator/ ( float b ) const
		{
			return Vector( (*this).values.x / b , (*this).values.y / b , (*this).values.z / b );
		}

		__host__ __device__ float dot( Vector b ) const
		{
			return ( ((*this).values.x * b.values.x) + ((*this).values.y * b.values.y) + ((*this).values.z * b.values.z) ); 
		}

		__host__ __device__ float magnitude( void )
		{
			return sqrtf( ((*this).values.x * (*this).values.x) + ((*this).values.y * (*this).values.y) + ((*this).values.z * (*this).values.z) );
		}

		__host__ __device__ Vector normalize( void )
		{
			float mag = this->magnitude();
			return Vector( ((*this).values.x / mag), ((*this).values.y / mag), ((*this).values.z / mag) ); 
		}
	};

}

#endif