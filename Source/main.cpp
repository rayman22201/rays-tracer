//Project Wild Doughnut
//A simple Ray Tracer Designed Explicitly for Ray Tracing Implicit Surfaces using CUDA
//By: Ray Imber a.k.a rayman22201

#include "wtypes.h"
#include "windows.h"
#include <stdlib.h>
#include <iostream>
#include <cuda_runtime.h>
#include "vector.cu"

#include "glew.h"
#include "glut.h"
#include <cuda_gl_interop.h>

using namespace wildDoughnut;

//Define the rayTracer wrapper function externally. Remember to link with rayTracer.cu!
extern void initRayTracer ( float4* imageBuffer, int width, int height, float marchInterval );
extern void animateRayTracer();
extern void runRayTracer();
extern void cleanUpRayTracer();
extern double benchmarkRayTracer();

//global vars
float4* imageBuffer;

//openGL buffer; 
GLuint glBufferID;
//cudaGraphicsResource* cudaResource;

int width;
int height;
int pause = 1;
int benchmarkDone = 0;
int runRT = 1;

//glut callbacks
static void displayCallback(void){

	if( benchmarkDone )
	{
		printf("benchmark Done\n");
		benchmarkDone = 0;
	}

	glClear(GL_COLOR_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);

	void* buffer = glMapBuffer( GL_PIXEL_UNPACK_BUFFER_ARB, GL_READ_WRITE);
	memcpy( buffer, imageBuffer, (sizeof(float4) * width * height) );
	glUnmapBuffer(GL_PIXEL_UNPACK_BUFFER_ARB);

	glDrawPixels( width, height, GL_RGBA, GL_FLOAT, 0 ); //draw from the pixel unpack buffer
	glFlush();
	GLenum err = glGetError();
	const GLubyte* errStr =  gluErrorString(err);
}

static inline void mapAndrunRayTracer()
{
	//get a pointer to the openGL buffer and give it to CUDA
	//float4* devPtr;
	//size_t size;
	//cudaGraphicsMapResources( 1, &cudaResource, NULL );
	//cudaGraphicsResourceGetMappedPointer( (void**)&devPtr, &size, cudaResource );
	
	//run the kernel
	runRayTracer();

	//unmap the pointer so openGL can render the buffer
	//cudaGraphicsUnmapResources( 1, &cudaResource, NULL );
}

static void idleCallback(void){
	//update animation if not paused
	if(!pause){ animateRayTracer(); }

	//run the Ray Tracer
	if( runRT == 1 )
	{
		//runRayTracer();
		//runRT = 0;
	}

	//call the display callback
	glutPostRedisplay();
}

static void keyboardCallback(unsigned char key, int x, int y)
{
    printf("key: %d\n", key);
	double fps;
	switch (key) {
		case 112:
			//pause button P
			if(pause == 0) { pause = 1; }
			else { pause = 0; }
			break;
		case 98:
			//b for benchmark
			benchmarkDone = 0;
			fps = benchmarkRayTracer();
			printf("FPS: %f\n\n", fps);
			Sleep( 1000 );
			benchmarkDone = 1;
			break;
		case 114:
			if(runRT == 0){runRT = 1;}
			break;
        case 27:
			//esc key
			exit(0);
			break;
	}
}

// Get the horizontal and vertical screen sizes in pixel
void GetDesktopResolution(int& horizontal, int& vertical)
{
   RECT desktop;
   // Get a handle to the desktop window
   const HWND hDesktop = GetDesktopWindow();
   // Get the size of screen to the variable desktop
   GetWindowRect(hDesktop, &desktop);
   // The top left corner will have coordinates (0,0)
   // and the bottom right corner will have coordinates
   // (horizontal, vertical)
   horizontal = desktop.right;
   vertical = desktop.bottom;
}

//main function
int main(int argc, char* argv[]) 
{
	//set up CUDA to work with openGL
	cudaDeviceProp prop;
	int device;
	memset( &prop, 0, sizeof( cudaDeviceProp ) );
	prop.major = 1;
	prop.minor = 0;
	cudaChooseDevice( &device, &prop);

	cudaGLSetGLDevice( device );

	width = 1024;
	height = 1024;
	imageBuffer = new float4[ (width * height) ];

	printf("Project Wild Doughnut Reporting for Duty!\n");

	//get the center of the screen
	int horizontal, vertical;
	GetDesktopResolution(horizontal, vertical);
	horizontal = (horizontal / 2) - (width / 2);
	vertical = (vertical / 2) - (height / 2);

	//set up GLUT
	glutInit( &argc, argv );
	glutInitDisplayMode( GLUT_SINGLE | GLUT_RGBA );
	glutInitWindowSize( width, height );
	glutInitWindowPosition( horizontal, vertical);
	glutCreateWindow( "Wild Doughnut 0.2.5" );
	glewInit();//set up GLEW

	//set up the openGL buffer
	glGenBuffers( 1 , &glBufferID );
	glBindBuffer( GL_PIXEL_UNPACK_BUFFER_ARB , glBufferID );
	glBufferData( GL_PIXEL_UNPACK_BUFFER_ARB , (sizeof(float4) * width * height), NULL, GL_DYNAMIC_DRAW);

	//Register the Buffer with CUDA
	//cudaGraphicsGLRegisterBuffer( &cudaResource, glBufferID, cudaGraphicsMapFlagsNone );

	//bind openGL callbacks
	glutDisplayFunc( displayCallback );
	glutIdleFunc( idleCallback );
	glutKeyboardFunc( keyboardCallback );
	atexit(cleanUpRayTracer);

	//set up the Wild Doughnut Ray Tracer
	initRayTracer( imageBuffer, width, height, 0.09 );
	runRayTracer();

	//start up GLUT
	glutMainLoop();

	return 0;
}
