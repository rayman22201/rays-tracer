//defines an array class template


#ifndef __ARRAYOBJECT__
#define __ARRAYOBJECT__

namespace wildDoughnut {

	template <typename T>
	class ArrayObject {
	public:
		T* hostData;
		T* deviceData;
		unsigned int numElements;		
	};
}

#endif