#pragma once

#include "preProc.h"

#include "cudaUtils.h"

#include <string>
#include <windows.h>
#include <windowsx.h>
#include <d3d11.h>
#include <dxgi.h>

#include <cuda_d3d11_interop.h>
#include <cuda_runtime_api.h>

using namespace std;

//#include "DebugGUI.h"

#include "Utils.h"

// Pre def
class DebugGUI;
class ShadowMap;


class DeviceHandler
{
public:
	// FIXME: FUGLY HACK
	static bool g_spacePressed;
	static bool g_returnPressed;

	DeviceHandler(HINSTANCE p_hInstance, int p_wndWidth, int p_wndHeight);
	~DeviceHandler();
	ID3D11Device* getDevice();
	ID3D11DeviceContext* getContext();
	HWND* getHWnd();
	int getWindowWidth();
	int getWindowHeight();
	void setWindowTitle( string p_text );

	void beginDrawing();
	void presentFrame();

private:
	void initWindow();
	string shaderModel();
	char* featureLevelToCString(D3D_FEATURE_LEVEL featureLevel);
	wchar_t* featureLevelToWCString(D3D_FEATURE_LEVEL featureLevel);
	void findCudaAdapter();
	void setTitle();
	void initD3D();

private:
	//window settings
	int m_wndWidth;
	int m_wndHeight;
	HINSTANCE m_hInstance;
	HWND m_hWnd;
	IDXGIAdapter1* m_usedAdapter;
	int m_usedCudaDevice;

	//Direct3D
	D3D_FEATURE_LEVEL m_featureLevel;
	ID3D11Device* m_device;
	ID3D11DeviceContext* m_devContext;
	ID3D11RenderTargetView* m_rtv;
	ID3D11Texture2D* m_depthStencil;
	ID3D11DepthStencilView* m_dsv;    //depth stencil view - z/depth buffer
	IDXGISwapChain* m_swapchain;
	D3D11_VIEWPORT m_viewport;
};