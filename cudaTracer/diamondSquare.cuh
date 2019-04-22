#pragma once
#include <cuda_runtime.h>

extern "C" {
	void cu_diamondSquare( const void *surface, int width, int height,
		size_t pitch, unsigned char* p_rngStates );
	unsigned char* cu_initCurand( int width, int height );
	void cu_cleanCurand( unsigned char* p_rngStates );
}