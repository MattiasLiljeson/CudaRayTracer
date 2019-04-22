#pragma once

#include <d3d11.h>
#include "preProc.h"

#if (defined( DEBUG ) || defined( _DEBUG ))  && !defined(SKIP_D3D_DEBUG)
#include <DXGI.h> 
#include <Initguid.h> 
#include <DXGIDebug.h>
#include <D3D11SDKLayers.h>
#endif //DEBUG || _DEBUG

//! Wraps d3d-debug COM-objects that allow detailed printing of live objects at run-time.
/*!
If DEBUG- or _DEBUG-flags are set, D3DDebug will be empty, and have no functionality whatsoever.
Warning: these members will be reported as live objects, and may influence other COM-objects to do the same. Use with caution.
\ingroup xkill-renderer
\sa DEBUG
\sa _DEBUG
*/
class D3DDebugger
{
public:
	//! Sets object to default state and nulls all members.
	D3DDebugger( ID3D11Device* p_device );
	//! Clears memory and resets to default state.
	~D3DDebugger();

	//! Clears memory and resets to default state.
	void reset();

	//! Prints all live COM-objects.
	/*!
	\return Any error encountered.
	*/
	HRESULT reportLiveDeviceObjects();
	//! Prints all live objects.
	/*!
	\return Any error encountered.
	*/
	HRESULT reportLiveObjects();

private:
	//! Initializes all members. Must be called before usage of report-functionality.
	/*!
	\param device DirectX Device pointer.
	\return Any eror encountered.
	*/
	HRESULT init(ID3D11Device* p_device);

private:
	#if (defined( DEBUG ) || defined( _DEBUG ))  && !defined(SKIP_D3D_DEBUG)
	IDXGIFactory1*	m_dxgiFactory;		//!< Will not be created if DEBUG- or _DEBUG-flags are not set.
	ID3D11Debug*	m_d3d11Debug;		//!< Will not be created if DEBUG- or _DEBUG-flags are not set.
	IDXGIDebug*		m_dxgiDebug;		//!< Will not be created if DEBUG- or _DEBUG-flags are not set.
	IDXGIInfoQueue* m_dxgiInfoQueue;	//!< Will not be created if DEBUG- or _DEBUG-flags are not set.
	#endif //DEBUG || _DEBUG
};