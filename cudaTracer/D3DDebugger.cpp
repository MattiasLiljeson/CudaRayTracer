#include "D3DDebugger.h"
#include "Utils.h"

#if (defined( DEBUG ) || defined( _DEBUG )) && !defined(SKIP_D3D_DEBUG)

D3DDebugger::D3DDebugger( ID3D11Device* p_device ){
	m_dxgiFactory	= nullptr;
	m_d3d11Debug	= nullptr;
	m_dxgiDebug		= nullptr;
	m_dxgiInfoQueue	= nullptr;
	init( p_device );
}

D3DDebugger::~D3DDebugger(){
	reset();
}

void D3DDebugger::reset(){
	SAFE_RELEASE( m_dxgiFactory );
	SAFE_RELEASE( m_d3d11Debug );
	SAFE_RELEASE( m_dxgiDebug );
	SAFE_RELEASE( m_dxgiInfoQueue );
}

HRESULT D3DDebugger::reportLiveDeviceObjects(){
	return m_d3d11Debug->ReportLiveDeviceObjects( D3D11_RLDO_DETAIL );
}

HRESULT D3DDebugger::reportLiveObjects(){
	return m_dxgiDebug->ReportLiveObjects( DXGI_DEBUG_ALL, DXGI_DEBUG_RLO_DETAIL );
}

HRESULT D3DDebugger::init(ID3D11Device* device){
	HRESULT hr = CreateDXGIFactory1( __uuidof(IDXGIFactory1), (void**)(&m_dxgiFactory) );
	if( hr == S_OK ) {
		typedef HRESULT( __stdcall *fPtr )( const IID&, void** ); 
		HMODULE hDll = GetModuleHandleW( L"dxgidebug.dll" ); 
		fPtr DXGIGetDebugInterface = (fPtr)GetProcAddress( hDll,
			"DXGIGetDebugInterface" );

		DXGIGetDebugInterface( __uuidof(IDXGIDebug), (void**)&m_dxgiDebug );
		DXGIGetDebugInterface( __uuidof(IDXGIInfoQueue),
			(void**)&m_dxgiInfoQueue );

		hr = device->QueryInterface( __uuidof(ID3D11Debug),
			(void**)(&m_d3d11Debug) );
	}
	return hr;
}

#else

D3DDebugger::D3DDebugger( ID3D11Device* p_device ){
}

D3DDebugger::~D3DDebugger(){
	reset();
}

void D3DDebugger::reset(){
}

HRESULT D3DDebugger::reportLiveDeviceObjects(){
	return S_OK;
}

HRESULT D3DDebugger::reportLiveObjects(){
	return S_OK;
}

HRESULT D3DDebugger::init(ID3D11Device* device){
	return S_OK;
}

#endif //DEBUG || _DEBUG