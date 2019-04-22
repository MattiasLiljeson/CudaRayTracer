#include "DeviceHandler.h"
#include <sstream>
#include "DebugGUI.h"
#include "cudaUtils.h"

// FIXME: FUGLY continues
bool DeviceHandler::g_spacePressed = true;
bool DeviceHandler::g_returnPressed = false;

//=========================================================================
// Global callback function used by windows
//=========================================================================
LRESULT CALLBACK wndProc(HWND hWnd, UINT message, WPARAM wParam, LPARAM lParam)
{
	// See if DebugGUI (AntTweakbar) catches the msg first
	if( DebugGUI::getInstance()->updateMsgProc(hWnd, message, wParam, lParam) ){
		return 0;
	}

	// Otherwise handle the msg
	switch(message)
	{
	case WM_DESTROY:
		PostQuitMessage(0);
		break;

	case WM_KEYDOWN:
		if( wParam == VK_ESCAPE ){
			PostQuitMessage(0);
		} else if( wParam == VK_SPACE ) {
			DeviceHandler::g_spacePressed = true;
		} else if( wParam == VK_RETURN ) {
			DeviceHandler::g_returnPressed = true;
		}
		break;
	}

	// Handle any messages the switch statement didn't
	return DefWindowProc( hWnd, message, wParam, lParam );
}


//=========================================================================
// Public functions
//=========================================================================
DeviceHandler::DeviceHandler(HINSTANCE p_hInstance, int p_wndWidth, int p_wndHeight)
{
	m_hInstance = p_hInstance;
	m_hWnd = nullptr;
	m_usedAdapter = nullptr;
	m_wndHeight = p_wndHeight;
	m_wndWidth = p_wndWidth;
	m_usedCudaDevice = -1;

	m_dsv = nullptr;

	initWindow();   
	findCudaAdapter();
	initD3D();
}

DeviceHandler::~DeviceHandler()
{
	//Some COMs aren't released now, must be fixed
	SAFE_RELEASE(m_rtv);
	SAFE_RELEASE(m_dsv);
	SAFE_RELEASE(m_swapchain);
	SAFE_RELEASE(m_rtv);
	SAFE_RELEASE(m_device);    // close and release the 3D m_device
	SAFE_RELEASE(m_devContext);    // close and release the 3D m_device
	SAFE_RELEASE(m_usedAdapter);
}

ID3D11Device* DeviceHandler::getDevice()
{
	return m_device;
}

ID3D11DeviceContext* DeviceHandler::getContext()
{
	return m_devContext;
}

HWND* DeviceHandler::getHWnd()
{
	return &m_hWnd;
}

int DeviceHandler::getWindowWidth()
{
	return m_wndWidth;
}

int DeviceHandler::getWindowHeight()
{
	return m_wndHeight;
}

void DeviceHandler::setWindowTitle( string p_text )
{
	SetWindowTextA(m_hWnd, p_text.c_str());
}

void DeviceHandler::beginDrawing()
{
	//m_devContext->RSSetViewports(1, &m_viewport);    //Set the viewport
	//#1F7116
	// clear the window to a deep blue
	//Set the render target as the back buffer
	m_devContext->OMSetRenderTargets(1, &m_rtv, m_dsv);

	//m_devContext->ClearRenderTargetView( m_rtv, D3DXCOLOR(0.0f, 0.2f, 0.4f, 1.0f) );
	float clearCol[4] = {0.0f, 0.2f, 0.4f, 1.0f};
	m_devContext->ClearRenderTargetView( m_rtv, &clearCol[0] );
	m_devContext->ClearDepthStencilView( m_dsv, D3D10_CLEAR_DEPTH, 1.0f, 0 );

	// reset states
	m_devContext->OMSetDepthStencilState( 0, 0 );
	float blendFactors[] = { 0.0f, 0.0f, 0.0f, 0.0f };
	m_devContext->OMSetBlendState( 0, blendFactors, 0xffffffff );
}
void DeviceHandler::presentFrame()
{
	// display the rendered frame
	HR( m_swapchain->Present(0, 0) );
}

//=========================================================================
// Private functions
//=========================================================================
void DeviceHandler::initWindow()
{
	WNDCLASSEX wc;
	ZeroMemory(&wc, sizeof(WNDCLASSEX));
	wc.cbSize = sizeof(WNDCLASSEX);
	wc.style = CS_HREDRAW | CS_VREDRAW;
	wc.lpfnWndProc = wndProc;
	wc.hInstance = m_hInstance;
	wc.hCursor = LoadCursor(NULL, IDC_ARROW);
	wc.hbrBackground = (HBRUSH)COLOR_WINDOW;
	wc.lpszClassName = L"WindowClass";

	RegisterClassEx(&wc);

	//create window, save result
	int xPos = 100;
	int yPos = 100;
	m_hWnd = CreateWindowEx(NULL, L"WindowClass", L"Window", 
		WS_OVERLAPPEDWINDOW, xPos, yPos, m_wndWidth, m_wndHeight, NULL,
		NULL, m_hInstance, NULL);

	ShowWindow(m_hWnd, SW_SHOW);
}

std::string DeviceHandler::shaderModel()
{
	if(m_featureLevel == D3D_FEATURE_LEVEL_11_0)
		return "5_0";
	if(m_featureLevel == D3D_FEATURE_LEVEL_10_1)
		return "4_1";
	if(m_featureLevel == D3D_FEATURE_LEVEL_10_0)
		return "4_0";
	else
		return "4_0"; // no support for older shader models than 4.0
}

char* DeviceHandler::featureLevelToCString(D3D_FEATURE_LEVEL featureLevel)
{
	if(featureLevel == D3D_FEATURE_LEVEL_11_0)
		return "11.0";
	if(featureLevel == D3D_FEATURE_LEVEL_10_1)
		return "10.1";
	if(featureLevel == D3D_FEATURE_LEVEL_10_0)
		return "10.0";

	return "Unknown";
}

wchar_t* DeviceHandler::featureLevelToWCString(D3D_FEATURE_LEVEL featureLevel)
{
	if(featureLevel == D3D_FEATURE_LEVEL_11_0)
		return L"11.0";
	if(featureLevel == D3D_FEATURE_LEVEL_10_1)
		return L"10.1";
	if(featureLevel == D3D_FEATURE_LEVEL_10_0)
		return L"10.0";

	return L"Unknown";
}

void DeviceHandler::findCudaAdapter()
{
	cudaError cudaStatus;

	IDXGIFactory1 *factory;
	HR( CreateDXGIFactory1( __uuidof(IDXGIFactory1), (void **)(&factory)) );

	// iterate through the candidate adapters
	for( UINT adapterIdx = 0; !m_usedAdapter; ++adapterIdx )
	{
		// get a candidate DXGI adapterIdx
		IDXGIAdapter1 *adapter = NULL;
		if (FAILED( factory->EnumAdapters1(adapterIdx, &adapter) )) {
			break;
		}

		// query to see if there exists a corresponding compute device
		int cudaDevice;
		cudaStatus = cudaD3D11GetDevice(&cudaDevice, adapter);

		if (cudaSuccess == cudaStatus)
		{
			// if so, mark it as the one against which to create our device
			m_usedAdapter = adapter;
			m_usedAdapter->AddRef();
			m_usedCudaDevice = cudaDevice;
		}

		adapter->Release();
	}

	factory->Release();
	// clear any errors we got while querying invalid compute devices
	cudaStatus = cudaGetLastError();
}

void DeviceHandler::setTitle()
{
	cudaDeviceProp cudaDeviceProps;
	cudaError_t cudaErr = cudaGetDeviceProperties( &cudaDeviceProps, m_usedCudaDevice);

	wstringstream ss;
	//wstring devDesc = L"";
	if( m_usedAdapter != NULL )
	{
		DXGI_ADAPTER_DESC1 desc;
		m_usedAdapter->GetDesc1(&desc);
		ss<<wstring(desc.Description);
	}

	ss << L"with Feature level: ";
	ss << featureLevelToWCString(m_featureLevel);

	ss << L". Cuda Device: ";
	ss << m_usedCudaDevice;
	ss << L" with compute capability: ";
	ss << cudaDeviceProps.major;
	ss << L".";
	ss << cudaDeviceProps.minor;

	SetWindowText( m_hWnd, ss.str().c_str() );
}

void DeviceHandler::initD3D()
{
	RECT rc;
	GetClientRect( m_hWnd, &rc );
	int screenWidth = rc.right - rc.left;
	int screenHeight = rc.bottom - rc.top;

	UINT createDeviceFlags = 0;
#if  defined(_DEBUG) && !defined(SKIP_D3D_DEBUG)
	createDeviceFlags |= D3D11_CREATE_DEVICE_DEBUG;
#endif

	D3D_DRIVER_TYPE driverType;

	D3D_DRIVER_TYPE driverTypes[] =
	{
		D3D_DRIVER_TYPE_HARDWARE,
		D3D_DRIVER_TYPE_REFERENCE,
	};
	UINT driverTypeCnt = sizeof(driverTypes) / sizeof(driverTypes[0]);

	DXGI_SWAP_CHAIN_DESC sd;
	ZeroMemory( &sd, sizeof(sd) );
	sd.BufferCount = 1;
	sd.BufferDesc.Width = screenWidth;
	sd.BufferDesc.Height = screenHeight;
	sd.BufferDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
	sd.BufferDesc.RefreshRate.Numerator = 60;
	sd.BufferDesc.RefreshRate.Denominator = 1;
	sd.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
	sd.OutputWindow = m_hWnd;
	sd.SampleDesc.Count = 1;
	sd.SampleDesc.Quality = 0;
	sd.Windowed = TRUE;

	D3D_FEATURE_LEVEL featureLevelsToTry[] = {
		D3D_FEATURE_LEVEL_11_0,
		D3D_FEATURE_LEVEL_10_1,
		D3D_FEATURE_LEVEL_10_0
	};
	D3D_FEATURE_LEVEL initiatedFeatureLevel;

	HRESULT hr = S_OK;

	for( unsigned int driverTypeIdx = 0; driverTypeIdx < driverTypeCnt; driverTypeIdx++ )
	{
		driverType = driverTypes[driverTypeIdx];
		hr = D3D11CreateDeviceAndSwapChain(
			m_usedAdapter,
			D3D_DRIVER_TYPE_UNKNOWN,
			NULL,
			createDeviceFlags,
			featureLevelsToTry,
			ARRAYSIZE(featureLevelsToTry),
			D3D11_SDK_VERSION,
			&sd,
			&m_swapchain,
			&m_device,
			&initiatedFeatureLevel,
			&m_devContext
			);

		if( SUCCEEDED( hr ) )
		{
			m_featureLevel = initiatedFeatureLevel;
			setTitle();

			SET_D3D_OBJECT_NAME(m_devContext, "context")
			/* SET_D3D_OBJECT_NAME(m_device, "device")
			SET_D3D_OBJECT_NAME(m_swapchain, "swapchain") */

			break;
		}
	}

	// begin interop
	gpuErrchk( cudaD3D11SetDirect3DDevice( m_device ) );
	gpuErrchk( cudaPeekAtLastError() );
	getLastCudaError("cudaD3D11SetDirect3DDevice failed");

	// Create a render target view
	ID3D11Texture2D* pBackBuffer;
	HR(m_swapchain->GetBuffer( 0, __uuidof( ID3D11Texture2D ), (LPVOID*)&pBackBuffer ));

	HR(hr = m_device->CreateRenderTargetView( pBackBuffer, NULL, &m_rtv ));
	pBackBuffer->Release();

	SET_D3D_OBJECT_NAME(m_rtv, "rtv")


	// Create depth stencil texture
	D3D11_TEXTURE2D_DESC descDepth;
	descDepth.Width = screenWidth;
	descDepth.Height = screenHeight;
	descDepth.MipLevels = 1;
	descDepth.ArraySize = 1;
	descDepth.Format = DXGI_FORMAT_D32_FLOAT;
	descDepth.SampleDesc.Count = 1;
	descDepth.SampleDesc.Quality = 0;
	descDepth.Usage = D3D11_USAGE_DEFAULT;
	descDepth.BindFlags = D3D11_BIND_DEPTH_STENCIL;
	descDepth.CPUAccessFlags = 0;
	descDepth.MiscFlags = 0;
	HR(m_device->CreateTexture2D( &descDepth, NULL, &m_depthStencil ));

	// Create the depth stencil view
	D3D11_DEPTH_STENCIL_VIEW_DESC descDSV;
	ZeroMemory(&descDSV, sizeof(descDSV));
	descDSV.Format = descDepth.Format;
	descDSV.ViewDimension = D3D11_DSV_DIMENSION_TEXTURE2D;
	descDSV.Texture2D.MipSlice = 0;
	HR(m_device->CreateDepthStencilView( m_depthStencil, &descDSV, &m_dsv ));

	m_devContext->OMSetRenderTargets(1, &m_rtv, m_dsv);
	//m_devContext->OMSetRenderTargets( 1, &m_rtv, NULL );

	// Setup the viewport
	D3D11_VIEWPORT vp;
	vp.Width = (float)screenWidth;
	vp.Height = (float)screenHeight;
	vp.MinDepth = 0.0f;
	vp.MaxDepth = 1.0f;
	vp.TopLeftX = 0;
	vp.TopLeftY = 0;
	m_devContext->RSSetViewports( 1, &vp );
}