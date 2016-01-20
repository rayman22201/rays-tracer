//The Ray Tracing Kernel.
//    I was inspired by Nvidia Optix, but is much simpler, and optimized for implicit surfaces.
//
//    The kernal acts as the virtual camera, sending out Rays and computing the color for each pixel
//    The screen is represented as a simple Buffer in Device Memory of size (Width * Height * Float4). 
//    Colors are stored as floating points
//
//    Included is an externally exposed entry function so that the kernal can be called from an arbitrary C++ file.
//
//By Ray Imber a.k.a rayman22201

#include "Windows.h"
#include <stdlib.h>
#include <iostream>
#include <cuda_runtime.h>
#include "vector.cu"

#include "glew.h"
#include "glut.h"
#include <cuda_gl_interop.h>

#include "rayObject.cuh"
#include "radianceRay.cuh"
#include "shadowRay.cuh"

#include "lightObject.cuh"
#include "pointLight.cuh"

#include "surfaceObject.cuh"
#include "analyticSphere.cuh"
#include "marchingSphere.cuh"
#include "marchingTorus.cuh"
#include "marchingBarthDecic.cuh"
#include "marchingBlobby.cuh"

#include "materialObject.cuh"
#include "blinnShader.cuh"

using namespace wildDoughnut;

//-----------------------------Global Vars------------------------------------//

//set up imagebuffer
float4* myImageBuffer; //kept for legacy purpose. GPU interop mode doesn't need this
int myWidth;
int myHeight;

float4* dev_imageBuffer;
int* dev_width;
int* dev_height;

//declare scene arrays
ArrayObject<pointLight> lightBuffer;
ArrayObject<blinnShader> materialBuffer;
ArrayObject<MarchingTorus> sceneBuffer;

//Used for animation test
float sphere0Angle;
float myMarch;

//---------------------------------------------------------------------------//

//------------------------Device Utility functions--------------------------//

//Basically a Macro to encapsulate the intersection checking loop. Loops through each object in SceneBuffer until it finds an intersection, or hits the end of the array.
// If an intersection is found, returns 1, else returns 0.
template <typename U>
__device__ inline int checkForIntersection( rayObject* checkRay, float* intersection, Vector* normal, ArrayObject<U>* sceneBuffer, int* intersectedObject )
{
	int anyIntersection = 0;
	int intersectionFound[20];
	float intersections[20];
	Vector normals[20];

	surfaceObject<U>* currObject;
	int smallestIndex = 0;
	
	//naively check for an intersection for every object in the scene
	//I am assuming I will only ever have a small amount of objects, so I can get away with this.
	for(int index = 0; index < sceneBuffer->numElements; index++)
	{
		currObject = &(sceneBuffer->deviceData[index]);
		intersectionFound[index] = currObject->checkIntersection( checkRay, &(intersections[index]), &(normals[index]) );
		if(intersectionFound[index]){ smallestIndex = index; anyIntersection = 1; }
	}
	if( anyIntersection )
	{
		for(int index = 0; index < sceneBuffer->numElements; index++)
		{
			if( intersectionFound[index] && (intersections[index] < intersections[smallestIndex]) )
			{
				smallestIndex = index;
			}
		}
		(*intersectedObject) = smallestIndex; 
		(*intersection) = intersections[smallestIndex];
		(*normal) = normals[smallestIndex];
	}
	return anyIntersection;
}

template<typename T, typename U, typename V>
__device__ inline Vector traceRay( radianceRay* pixelRay, ArrayObject<T>* lightBuffer, ArrayObject<U>* sceneBuffer, ArrayObject<V>* materialBuffer )
{
	Vector finalColor;
	finalColor = Vector(0.0f,0.0f,0.0f);
	int maxDepth = 0; //max reflection depth

	//reflection ray bounces
	do
	{
		//----------------------------------one level of direct light bounce--------------------------------------//
		//check for an intersection
		int intersectionFound = 0;
		float intersection = 0.0f;
		Vector normal;
		int objectIndex;
		normal = Vector(0,0,0);

		intersectionFound = checkForIntersection( pixelRay, &intersection, &normal, sceneBuffer, &objectIndex );
		if( intersectionFound )
		{
			//for each light in the scene, check if in shadow, and apply color information
			for(int j = 0; j < lightBuffer->numElements; j++)
			{
				lightObject* light = &(lightBuffer->deviceData[j]);
				Vector hit_point = pixelRay->paramToVector( intersection );
				//create a Shadow Ray
				shadowRay shadowChecker;
				shadowChecker.tmin = 0.1f;
				Vector lightDistance = (light->position - hit_point); 
				shadowChecker.tmax = lightDistance.magnitude();
				shadowChecker.origin = pixelRay->paramToVector(intersection);
				lightDistance = lightDistance.normalize();
				shadowChecker.direction = lightDistance;
			
				//check for an intersection
				int inShadow = 0;
				if( light->castShadows )
				{
					float shadow_t = 0.0f;
					Vector shadowNormal;
					int shadowIndex;
					inShadow = checkForIntersection( &shadowChecker, &shadow_t, &shadowNormal, sceneBuffer, &shadowIndex );
				}

				//add color information
				surfaceObject<U>* object = &(sceneBuffer->deviceData[objectIndex]);
				materialObject<V>* shader = &(materialBuffer->deviceData[(object->material)]);
				shader->valueAtIntersection( pixelRay, intersection, &normal, light);	
				if( inShadow )
				{
					//multiply the shadow on top
					//finalColor = Vector(0.1f,0.1f,0.1f);
				}
				else
				{
					finalColor = finalColor + pixelRay->color;
				}
			}
		}
		//----------------------------------------------------------------------------------------------------//
		pixelRay->depth ++;
		//calculate Reflected Ray Direction
		float refAngle = (2.0f * ( normal.dot( pixelRay->direction ) ) );
		pixelRay->origin = pixelRay->paramToVector(intersection);
		pixelRay->direction = (pixelRay->direction - (normal * refAngle));
		pixelRay->direction = pixelRay->direction.normalize();
		pixelRay->color = Vector(0,0,0);
	}
	while( (pixelRay->importance > 0.0f) && (pixelRay->depth < maxDepth) );
	
	return finalColor;
}

//---------------------------------------------------------------------------//

//------------------------Kernel function------------------------------------//
template <typename T, typename U, typename V>
__global__ void rayTracerKernel( float4* imageBuffer, int width, int height, ArrayObject<T> lightBuffer, ArrayObject<U> sceneBuffer, ArrayObject<V> materialBuffer )
{
    // map from threadIdx/BlockIdx to pixel position
    int threadX = threadIdx.x + blockIdx.x * blockDim.x;
    int threadY = threadIdx.y + blockIdx.y * blockDim.y;
    int offset = threadX + threadY * blockDim.x * gridDim.x;

	//center the viewing plane to (0,0), with a viewing window of [(-1,-1) to (1,1)]
	float x = ( (((float)threadX) - (float)width/2.0f) / ((float)width / 2.0f) );
	float y = ( (((float)threadY) - (float)height/2.0f) / ((float)height / 2.0f) );
	Vector finalColor = make_float3(0.0f,0.0f,0.5f);

	//pinhole camera
	Vector origin = Vector( x, y, 2.0f );
	Vector direction = Vector( 0, 0, -1.0f); //face the negative Z direction
	direction = direction.normalize();
	
	//create a Radiance Ray
	radianceRay pixelRay;
	pixelRay.tmin = 0.1f;
	pixelRay.tmax = 3.0f;
	pixelRay.importance = 1.0f; //0 = no reflection
	pixelRay.depth = 0;
	pixelRay.origin = origin;
	pixelRay.direction = direction;

	//Trace the Ray
	finalColor = traceRay( &pixelRay, &lightBuffer, &sceneBuffer, &materialBuffer );
	//if( (finalColor.values.x == 0.0f) && (finalColor.values.y == 0.0f) && (finalColor.values.z == 0.0f) ) { finalColor = make_float3(0.0f,0.0f,0.5f); }

	//write output back to imageBuffer
	imageBuffer[offset] = make_float4(finalColor.values.x, finalColor.values.y, finalColor.values.z, 1.0f);
	//printf("imageBuffer %f,%f,%f,%f\n",imageBuffer[offset].x,imageBuffer[offset].y,imageBuffer[offset].z,imageBuffer[offset].w);
}
//-----------------------------------------------------------------------------------//

//--------------------Data Transfer Utility Functions-------------------------------//

template <typename T>
__host__ inline void copyArrayObjectToDevice( ArrayObject<T>* object )
{
	cudaError_t error[2];
	T* host_data;
	T* device_data;

	error[0] = cudaMalloc( &device_data, (sizeof(T) * object->numElements) );
	
	host_data = object->hostData;
	object->deviceData = device_data;
	error[1] = cudaMemcpy( device_data, host_data, (sizeof(T) * object->numElements), cudaMemcpyHostToDevice );
}

template <typename T>
__host__ inline void updateDeviceArrayObject( ArrayObject<T>* object )
{
	cudaError_t error;
	T* host_data;
	T* device_data;

	host_data = object->hostData;
	device_data = object->deviceData;

	error = cudaMemcpy( device_data, host_data, (sizeof(T) * object->numElements), cudaMemcpyHostToDevice );
}

template <typename T>
__host__ inline void deleteDeviceArrayObject( ArrayObject<T>* object )
{
	cudaError_t error;
	T* device_data;

	device_data = object->deviceData;
	error = cudaFree( device_data );
}

//Copies an array of float4 to the device
//   Side Effects: dev_imageBuffer becomes a pointer IN DEVICE SPACE of the array
//                 dev_width and dev_height become pointers to the width and height values IN DEVICE SPACE
//                 If Copy = 1, copies from the Host to the Device
//                 If Copy = 0, just Malloc an array of size width * height on the Device
__host__ inline void copyImageBufferToDevice( float4* imageBuffer, int width, int height, float4** dev_imageBuffer, unsigned int copy )
{
	cudaError_t error[2];
	error[0] = cudaMalloc( dev_imageBuffer, (sizeof(float4) * width * height) );

	if(copy){ error[1] = cudaMemcpy( (*dev_imageBuffer), imageBuffer, (sizeof(float4) * width * height), cudaMemcpyHostToDevice ); }
}

//Copies an array of float4 from the Device to the Host
//   Side Effects: host_imageBuffer becomes a pointer IN HOST SPACE to the copied float4 data
__host__ inline void copyImageBufferToHost( float4* dev_imageBuffer, float4* host_imageBuffer, int width, int height )
{
	cudaError_t error = cudaMemcpy( host_imageBuffer, dev_imageBuffer, (sizeof(float4) * width * height), cudaMemcpyDeviceToHost );
}
//-----------------------------------------------------------------------------------//

//-----------------scene Setup Macros-----------------------------------------------//
//Really just for convenience, to make it easy to switch to different set ups

//1 Yellow Sphere
//1 point light
//1 blinn shader: Yellow
__host__ inline void setupScene1(void)
{
	pointLight* lightData;
	blinnShader* shaderData;
	MarchingSphere* spheres;

	lightData = new pointLight[1];
	shaderData = new blinnShader[1];
	spheres = new MarchingSphere[1];

	spheres[0] = MarchingSphere();
	spheres[0].material = 0; //index 0 of the material Buffer is the shader for this sphere
	spheres[0].radius = 0.25f;
	spheres[0].position = Vector(0.5f,0.0f,0.0f);
	spheres[0].marchInterval = 0.03f;
	sphere0Angle = 0.0f;

	lightData[0] = pointLight();
	lightData[0].castShadows = 1;
	lightData[0].color = Vector( 1.0f, 1.0f, 1.0f );
	lightData[0].intensity = 1.0f;
	lightData[0].position = Vector( 0.0f, 1.0f, 20.0f );

	shaderData[0] = blinnShader();
	shaderData[0].lambertCoeff = 1.0f;
	shaderData[0].diffuseColor = Vector(1.0f,0.9f,0.1f);

	lightBuffer.hostData = lightData;
	lightBuffer.numElements = 1;

	materialBuffer.hostData = shaderData;
	materialBuffer.numElements = 1;

	//sceneBuffer.hostData = spheres;
	sceneBuffer.numElements = 1;
}

__host__ inline void setupScene1Torus(void)
{
	pointLight* lightData;
	blinnShader* shaderData;
	MarchingTorus* tori;

	lightData = new pointLight[1];
	shaderData = new blinnShader[1];
	tori = new MarchingTorus[1];

	tori[0] = MarchingTorus();
	tori[0].material = 0; //index 0 of the material Buffer is the shader for this sphere
	tori[0].radius1 = 0.10f;
	tori[0].radius2 = 0.25f;
	tori[0].position = Vector(0.0f,0.5f,0.0f);
	tori[0].marchInterval = myMarch;
	sphere0Angle = 0.0f;

	lightData[0] = pointLight();
	lightData[0].castShadows = 1;
	lightData[0].color = Vector( 1.0f, 1.0f, 1.0f );
	lightData[0].intensity = 1.0f;
	lightData[0].position = Vector( 0.0f, 1.0f, 20.0f );

	shaderData[0] = blinnShader();
	shaderData[0].lambertCoeff = 1.0f;
	shaderData[0].diffuseColor = Vector(1.0f,0.9f,0.1f);

	lightBuffer.hostData = lightData;
	lightBuffer.numElements = 1;

	materialBuffer.hostData = shaderData;
	materialBuffer.numElements = 1;

	sceneBuffer.hostData = tori;
	sceneBuffer.numElements = 1;
}

__host__ inline void setupScene1BarthDecic(void)
{
	pointLight* lightData;
	blinnShader* shaderData;
	MarchingBarthDecic* barths;

	lightData = new pointLight[1];
	shaderData = new blinnShader[1];
	barths = new MarchingBarthDecic[1];

	barths[0] = MarchingBarthDecic();
	barths[0].material = 0; //index 0 of the material Buffer is the shader for this sphere
	barths[0].position = Vector(0.0f,0.0f,0.0f);
	barths[0].marchInterval = myMarch;
	sphere0Angle = 0.0f;

	lightData[0] = pointLight();
	lightData[0].castShadows = 1;
	lightData[0].color = Vector( 1.0f, 1.0f, 1.0f );
	lightData[0].intensity = 1.0f;
	lightData[0].position = Vector( 0.0f, 1.0f, 20.0f );

	shaderData[0] = blinnShader();
	shaderData[0].lambertCoeff = 1.0f;
	shaderData[0].diffuseColor = Vector(1.0f,0.9f,0.1f);

	lightBuffer.hostData = lightData;
	lightBuffer.numElements = 1;

	materialBuffer.hostData = shaderData;
	materialBuffer.numElements = 1;

	//sceneBuffer.hostData = barths;
	sceneBuffer.numElements = 1;
}

__host__ inline void setupScene1Blobby(void)
{
	pointLight* lightData;
	blinnShader* shaderData;
	MarchingBlobby* blobbies;

	lightData = new pointLight[1];
	shaderData = new blinnShader[1];
	blobbies = new MarchingBlobby[1];

	blobbies[0] = MarchingBlobby();
	blobbies[0].material = 0; //index 0 of the material Buffer is the shader for this sphere
	blobbies[0].position = Vector(0.0f,0.0f,-0.5f);
	blobbies[0].param1 = 4.0f;
	blobbies[0].param2 = 0.1f;
	blobbies[0].marchInterval = myMarch;
	sphere0Angle = 0.0f;

	lightData[0] = pointLight();
	lightData[0].castShadows = 1;
	lightData[0].color = Vector( 1.0f, 1.0f, 1.0f );
	lightData[0].intensity = 1.0f;
	lightData[0].position = Vector( 0.0f, 1.0f, 20.0f );

	shaderData[0] = blinnShader();
	shaderData[0].lambertCoeff = 1.0f;
	shaderData[0].diffuseColor = Vector(1.0f,0.9f,0.1f);

	lightBuffer.hostData = lightData;
	lightBuffer.numElements = 1;

	materialBuffer.hostData = shaderData;
	materialBuffer.numElements = 1;

	//sceneBuffer.hostData = blobbies;
	sceneBuffer.numElements = 1;
}

//3 Spheres, 2 grey, 1 yellow
//2 point lights
//2 blinn shaders: grey, yellow
__host__ inline void setupScene2(void)
{
	pointLight* lightData;
	blinnShader* shaderData;
	MarchingSphere* spheres;

	lightData = new pointLight[2];
	shaderData = new blinnShader[2];
	spheres = new MarchingSphere[3];

	spheres[0] = MarchingSphere();
	spheres[0].material = 0; //index 0 of the material Buffer is the shader for this sphere
	spheres[0].radius = 0.25f;
	spheres[0].position = Vector(0.5f,0.0f,0.0f);
	spheres[0].marchInterval = 0.01f;
	sphere0Angle = 0.0f;

	spheres[1] = MarchingSphere();
	spheres[1].material = 1;
	spheres[1].radius = 0.25f;
	spheres[1].position = Vector( 0.0f, 0.0f, 0.0f );
	spheres[1].marchInterval = 0.01f;

	spheres[2] = MarchingSphere();
	spheres[2].material = 1; 
	spheres[2].radius = 0.25f;
	spheres[2].position = Vector( -0.5f, 0.5f, 0.0f );
	spheres[2].marchInterval = 0.01f;

	lightData[0] = pointLight();
	lightData[0].castShadows = 1;
	lightData[0].color = Vector( 1.0f, 1.0f, 1.0f );
	lightData[0].intensity = 1.0f;
	lightData[0].position = Vector( 0.0f, 1.0f, 20.0f );

	lightData[1] = pointLight();
	lightData[1].castShadows = 1;
	lightData[1].color = Vector( 1.0f, 1.0f, 1.0f );
	lightData[1].intensity = 1.0f;
	lightData[1].position = Vector( 0.0f, 20.0f, 0.0f );

	shaderData[1] = blinnShader();
	shaderData[1].lambertCoeff = 1.0f;
	shaderData[1].diffuseColor = Vector(0.5f,0.5f,0.5f);

	shaderData[0] = blinnShader();
	shaderData[0].lambertCoeff = 1.0f;
	shaderData[0].diffuseColor = Vector(1.0f,0.9f,0.1f);

	lightBuffer.hostData = lightData;
	lightBuffer.numElements = 2;

	materialBuffer.hostData = shaderData;
	materialBuffer.numElements = 2;

	//sceneBuffer.hostData = spheres;
	sceneBuffer.numElements = 3;
}

//5 spheres
__host__ inline void setupScene3(void)
{
	pointLight* lightData;
	blinnShader* shaderData;
	analyticSphere* spheres;

	lightData = new pointLight[2];
	shaderData = new blinnShader[2];
	spheres = new analyticSphere[7];

	spheres[0] = analyticSphere();
	spheres[0].material = 0; //index 0 of the material Buffer is the shader for this sphere
	spheres[0].radius = 0.25f;
	spheres[0].position = Vector(0.5f,0.0f,0.0f);
	sphere0Angle = 0.0f;

	spheres[1] = analyticSphere();
	spheres[1].material = 1;
	spheres[1].radius = 0.25f;
	spheres[1].position = Vector( 0.0f, 0.0f, 0.0f );

	spheres[2] = analyticSphere();
	spheres[2].material = 1; 
	spheres[2].radius = 0.25f;
	spheres[2].position = Vector( -0.5f, 0.5f, 0.0f );

	spheres[3] = analyticSphere();
	spheres[3].material = 1; 
	spheres[3].radius = 0.25f;
	spheres[3].position = Vector( -0.5f, -0.5f, 0.5f );

	spheres[4] = analyticSphere();
	spheres[4].material = 1; 
	spheres[4].radius = 0.25f;
	spheres[4].position = Vector( 0.5f, 0.5f, -0.5f );

	spheres[5] = analyticSphere();
	spheres[5].material = 1; 
	spheres[5].radius = 0.25f;
	spheres[5].position = Vector( -0.25f, 1.0f, -0.5f );

	spheres[6] = analyticSphere();
	spheres[6].material = 1; 
	spheres[6].radius = 0.25f;
	spheres[6].position = Vector( 0.25f, -1.0f, -0.5f );

	lightData[0] = pointLight();
	lightData[0].castShadows = 1;
	lightData[0].color = Vector( 1.0f, 1.0f, 1.0f );
	lightData[0].intensity = 1.0f;
	lightData[0].position = Vector( 0.0f, 1.0f, 20.0f );

	lightData[1] = pointLight();
	lightData[1].castShadows = 1;
	lightData[1].color = Vector( 1.0f, 1.0f, 1.0f );
	lightData[1].intensity = 1.0f;
	lightData[1].position = Vector( 0.0f, 20.0f, 0.0f );

	shaderData[1] = blinnShader();
	shaderData[1].lambertCoeff = 1.0f;
	shaderData[1].diffuseColor = Vector(0.5f,0.5f,0.5f);

	shaderData[0] = blinnShader();
	shaderData[0].lambertCoeff = 1.0f;
	shaderData[0].diffuseColor = Vector(1.0f,0.9f,0.1f);

	lightBuffer.hostData = lightData;
	lightBuffer.numElements = 2;

	materialBuffer.hostData = shaderData;
	materialBuffer.numElements = 2;

	//sceneBuffer.hostData = spheres;
	sceneBuffer.numElements = 7;
}
//------------------------------------------------------------------------------------//

//------------------------external entry functions------------------------------------//
__host__ void initRayTracer ( float4* imageBuffer, int width, int height, float marchInterval )
{
	//increase the CUDA stack size
	cudaError_t error;
	error = cudaDeviceSetLimit( cudaLimitStackSize, 8192 ); //8kb stack size

	//initalization of global vars
	myImageBuffer = imageBuffer;
	myWidth = width;
	myHeight = height;
	myMarch = marchInterval;

	//set up scene
	setupScene1Torus();

	//copy data to the GPU
	copyArrayObjectToDevice( &lightBuffer );
	copyArrayObjectToDevice( &materialBuffer );
	copyArrayObjectToDevice( &sceneBuffer );

	copyImageBufferToDevice( imageBuffer, width, height, &dev_imageBuffer, 0 );
}

__host__ void cleanUpRayTracer()
{
	deleteDeviceArrayObject( &lightBuffer );
	deleteDeviceArrayObject( &materialBuffer );
	deleteDeviceArrayObject( &sceneBuffer );
}

__host__ void animateRayTracer()
{
	//animation modifications go here
	//1.)modify
	//MarchingBlobby* sphere0 = &(sceneBuffer.hostData[0]);
	sphere0Angle += 0.035f;
	float radius = 0.5f;
	//sphere0->position = Vector( (radius * cosf(sphere0Angle)), (radius * cosf((-1.0f * sphere0Angle))), (radius * sinf(sphere0Angle)));
	//2.)copy to GPU
	updateDeviceArrayObject( &sceneBuffer );

}

__host__ void runRayTracer()
{
	//run the kernel
	dim3 blocks((myWidth+15)/16,(myHeight+15)/16); //cheap rounding up to the nearest int
	dim3 threads(16,16);//256 threads per block
	rayTracerKernel<<<blocks,threads>>>( dev_imageBuffer, myWidth, myHeight, lightBuffer, sceneBuffer, materialBuffer );
	//copy the output back
	copyImageBufferToHost( dev_imageBuffer, myImageBuffer, myWidth, myHeight);	
}

__host__ double benchmarkRayTracer()
{
	DWORD threshold = 1000;
	double fps = 0.0f;
	unsigned long frameCount = 0;
	DWORD startTime, endTime, interval;
	interval = 0;
	startTime = timeGetTime();

	//get a pointer to the openGL buffer and give it to CUDA
	//float4* devPtr;
	//size_t size;
	//cudaGraphicsMapResources( 1, &cudaResource, NULL );
	//cudaGraphicsResourceGetMappedPointer( (void**)&devPtr, &size, cudaResource );
	while( interval < threshold )
	{
		//run the kernel
		runRayTracer();
		//update timing info
		endTime = timeGetTime();
		frameCount++;
		interval = endTime - startTime;	
	}
	//unmap the pointer so openGL can render the buffer
	//cudaGraphicsUnmapResources( 1, &cudaResource, NULL );

	fps = ( ((double)frameCount * 1000.0f) / ((double)interval) );
	return fps;
}
//-----------------------------------------------------------------------------------//