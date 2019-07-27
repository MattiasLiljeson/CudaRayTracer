#ifndef UTILS_H
#define UTILS_H

#include <string>
#include <cassert>
#include "Popup.h"

// Object naming:
#if defined(DEBUG) | defined(_DEBUG)
//#define ERR_HR( hr ) Util::errHr( hr );
#define SET_D3D_OBJECT_NAME( resource, name ) SetD3dObjectName( resource, name );
#else
//#define ERR_HR( hr ) ;
#define SET_D3D_OBJECT_NAME( resource, name ) ;
#endif

template< int TNameLength >
inline void SetD3dObjectName( 
_In_ ID3D11DeviceChild* p_resource, 
_In_z_ const char ( &name )[ TNameLength ] ) {
		p_resource->SetPrivateData( WKPDID_D3DDebugObjectName, TNameLength - 1, name );
}


//*************************************************************************
// Simple d3d error checker for book demos.
//*************************************************************************

#if defined(DEBUG) | defined(_DEBUG)
	#ifndef HR
	#define HR(x)                                              \
	{                                                          \
		HRESULT hr = (x);                                      \
		if(FAILED(hr))                                         \
		{                                                      \
			Popup::error(__FILE__, __FUNCTION__, __LINE__, "HR error" );\
		}                                                      \
	}
	#endif

#else
	#ifndef HR
	#define HR(x) (x)
	#endif
#endif

// Release COM objects if not NULL and set them to NULL
#define SAFE_RELEASE(x)											\
	if( x )														\
	{															\
		x->Release();											\
		(x) = NULL; 											\
	}

struct IdxOutOfRange : std::exception {
	const char* what() const throw() {return "Index out of range\n";}
};

struct VecIdx
{
	enum Idx 
	{
		Idx_NA = -1, Idx_FIRST,

		X = Idx_FIRST, Y, Z, W,

		Idx_LAST = W, Idx_CNT
	};
};

struct Vec2 
{
	int x;
	int y;

	int& operator[]( unsigned int idx )
	{
		switch( idx )
		{
		case VecIdx::X:
			return x;
		case VecIdx::Y:
			return y;
		default:
			IdxOutOfRange e;
			throw e;
		}
		return x; // THIS SHOULD NEVER HAPPEN!
	}

	const int& operator[](int idx) const
	{

	}
};

#endif //UTILS_H