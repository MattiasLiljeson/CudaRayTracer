#pragma once

#include <cuda_runtime_api.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <thrust/system_error.h>
#include <thrust/system/cuda/error.h>
#include <sstream>

// http://stackoverflow.com/questions/14038589/what-is-the-canonical-way-to-check-for-errors-using-the-cuda-runtime-api
#define gpuErrchk(ans) { throw_on_cuda_error((ans), __FILE__, __LINE__); }
//#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }

inline void gpuAssert(cudaError_t code, char *file, int line, bool abort=true){
	if (code != cudaSuccess){ 
		//fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}

inline void throw_on_cuda_error(cudaError_t code, const char *file, int line)
{
	if(code != cudaSuccess)
	{
		std::stringstream ss;
		ss << file << "(" << line << ")";
		std::string file_and_line;
		ss >> file_and_line;
		throw thrust::system_error(code, thrust::cuda_category(), file_and_line);
	}
}

//
// Below stuph is shamelessly stolen from helper_cuda.h
//

#ifdef __DRIVER_TYPES_H__
// This will output the proper CUDA error strings in the event that a CUDA host call returns an error
#define checkCudaErrors(val)           check ( (val), #val, __FILE__, __LINE__ )
template< typename T >
bool check(T result, char const *const func, const char *const file, int const line)
{
	if (result)
	{
		fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\" \n",
				file, line, static_cast<unsigned int>(result), _cudaGetErrorEnum(result), func);
				std::stringstream ss;
				std::string msg("CUDA error at ");
				msg += file;
				msg += ":";
				ss << line;
				msg += ss.str();
				msg += " code=";
				ss << static_cast<unsigned int>(result) << cudaGetErrorString(result);
				msg += ss.str();
				msg += " (";
				msg += _cudaGetErrorEnum(result);
				msg += ") \"";
				msg += func;
				msg += "\"";
				std::cerr  << msg <<"\n";
				//Exception e(msg);
				//throw e;
		return true;
	}
	else
	{
		return false;
	}
}

// This will output the proper error string when calling cudaGetLastError
#define getLastCudaError(msg)      __getLastCudaError (msg, __FILE__, __LINE__)

inline void __getLastCudaError(const char *errorMessage, const char *file, const int line)
{
	cudaError_t err = cudaGetLastError();

	if (cudaSuccess != err)
	{
		fprintf(stderr, "%s(%i) : getLastCudaError() CUDA error : %s : (%d) %s.\n",
			file, line, errorMessage, (int)err, cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
}
#endif