#include "preProc.h"

#include "DeviceHandler.h"
#include "cudaUtils.h"

#include <cuda_runtime_api.h>
#include <cuda_d3d11_interop.h>
#include "TextureRenderer.h"
#include "globals.h"

#include<D3D11SDKLayers.h>
#include "D3DDebugger.h"

#include "thrust\system\system_error.h"
#include "Profiler.h"
#include "DebugGUI.h"

#include <lodepng.h>

void doView( HINSTANCE hInstance, HINSTANCE hPrevInstance,
			LPSTR lpCmdLine, int nCmdShow );
void doExperiment( HINSTANCE hInstance, HINSTANCE hPrevInstance,
				  LPSTR lpCmdLine, int nCmdShow );
void copyFloatsToCharVector( vector<float>& p_arr, vector<unsigned char>& out_img );

int WINAPI WinMain( HINSTANCE hInstance, HINSTANCE hPrevInstance,
				   LPSTR lpCmdLine, int nCmdShow )
{
#ifdef EXPERIMENT
	doExperiment( hInstance, hPrevInstance, lpCmdLine, nCmdShow );
#else
	doView( hInstance, hPrevInstance, lpCmdLine, nCmdShow );
#endif
}


void doView( HINSTANCE hInstance, HINSTANCE hPrevInstance,
			LPSTR lpCmdLine, int nCmdShow ){
	Profiler* prof = Profiler::getInstance();

	int wndWidth = PIC_WIDTH+16; // HACK: add space for borders
	int wndHeight = PIC_HEIGHT+39; // HACK: add space for borders and header
	DeviceHandler* deviceHandler = new DeviceHandler( hInstance, wndWidth, wndHeight);
	D3DDebugger d3dDbg(deviceHandler->getDevice());
	DebugGUI::getInstance()->init( deviceHandler->getDevice(),
		PIC_WIDTH, PIC_HEIGHT );

	TextureRenderer* texRender = nullptr;
	try{
			texRender = new TextureRenderer(deviceHandler, PIC_WIDTH, PIC_HEIGHT );
	} catch( thrust::system_error e ){
		Utils::error( e.what() );
	}

	if( texRender ){
		float dt = 1.0f/60.0f; // Hard coded for now
		MSG msg = {0};
		while( msg.message != WM_QUIT ){
			if( PeekMessage( &msg, NULL, 0, 0, PM_REMOVE ) ){
				TranslateMessage( &msg );
				DispatchMessage( &msg );
			} else {
				deviceHandler->beginDrawing();
				try{
					if( DeviceHandler::g_spacePressed ){
						DeviceHandler::g_spacePressed = false;
						texRender->update( dt );
					} else if( DeviceHandler::g_returnPressed ){
						DeviceHandler::g_returnPressed = false;
						static int cnt = 0;
						cnt++;
						char buf[128];
						sprintf( buf, "R%.3d-%ix%i.png", cnt, PIC_WIDTH, PIC_HEIGHT );
						string timerName( buf );

						unsigned int picSize = PIC_WIDTH*4*PIC_HEIGHT;
						vector<float> arr;
						arr.resize(picSize);
						texRender->copyToHostArray( &arr[0] );
						vector<unsigned char> img;
						img.resize(picSize);
						copyFloatsToCharVector( arr, img );
						unsigned error = lodepng::encode( buf, img, PIC_WIDTH, PIC_HEIGHT );
					}

					texRender->draw();
				} catch( thrust::system_error e ){
					Utils::error( e.what() );
				}

				DebugGUI::getInstance()->draw();
				deviceHandler->presentFrame();
			}
		}
	}
	delete texRender;
	delete deviceHandler;
	d3dDbg.reportLiveDeviceObjects();
}

void doExperiment( HINSTANCE hInstance, HINSTANCE hPrevInstance,
				  LPSTR lpCmdLine, int nCmdShow ){

	const int RES_TO_TRY[] = { 3, 5, 9, 17, 33, 65, 129, 257, 513, 1025, 2049, 4097 };
	const int TESTS_TO_RUN = 128;
	const int STD_RES_IDX = sizeof(RES_TO_TRY)/4-1;
	int testCnt = 0;

	Profiler* prof = Profiler::getInstance();

	int wndWidth = RES_TO_TRY[STD_RES_IDX]+16; // HACK: add space for borders
	int wndHeight = RES_TO_TRY[STD_RES_IDX]+39; // HACK: add space for borders and header
	DeviceHandler* deviceHandler = new DeviceHandler( hInstance, wndWidth, wndHeight);
	D3DDebugger d3dDbg(deviceHandler->getDevice());
	DebugGUI::getInstance()->init( deviceHandler->getDevice(),
		RES_TO_TRY[STD_RES_IDX], RES_TO_TRY[STD_RES_IDX] );
	DebugGUI::getInstance()->addVar("Performance", DebugGUI::DG_INT,
		DebugGUI::READ_ONLY, "Step", &testCnt, "" );

	for( int resIdx=0; resIdx<sizeof(RES_TO_TRY)/4; resIdx++ ){
		char buf[128];
		sprintf( buf, "R%.2d-%ix%i", resIdx, RES_TO_TRY[resIdx], RES_TO_TRY[resIdx] );
		string timerName( buf );
		prof->addPerfTimer( timerName, "", true );

		TextureRenderer* texRender = nullptr;
		try{
			 texRender = new TextureRenderer(deviceHandler, RES_TO_TRY[resIdx], RES_TO_TRY[resIdx] );
		} catch( thrust::system_error e ){
			Utils::error( e.what() );
		}

		if( texRender ){
			float dt = 1.0f/60.0f; // Hard coded for now
			MSG msg = {0};
			testCnt = 0;
			while( msg.message != WM_QUIT && testCnt < TESTS_TO_RUN ){
				if( PeekMessage( &msg, NULL, 0, 0, PM_REMOVE ) ){
					TranslateMessage( &msg );
					DispatchMessage( &msg );
				} else {
					deviceHandler->beginDrawing();
					try{
						prof->start( timerName, "" );
						texRender->update( dt );
						prof->stop( timerName, "" );
						testCnt++;
						texRender->draw();
					} catch( exception e ){
						Utils::error( e.what() );
					}

					DebugGUI::getInstance()->draw();
					deviceHandler->presentFrame();
				}
			}
		delete texRender;
		}
	}

	delete deviceHandler;
	d3dDbg.reportLiveDeviceObjects();
	bool timeStamp = true;
	bool statistics = true;
# ifdef __DEBUG
	prof->logTimersToFile( timeStamp, statistics, "testData-DEBUG" );
# else
	prof->logTimersToFile( timeStamp, statistics, "testData-RELEASE" );
#endif
}

void copyFloatsToCharVector( vector<float>& p_arr, vector<unsigned char>& out_img )
{
	for( unsigned int i=0; i<p_arr.size(); i++ ){
		int tmp = (int)(p_arr[i]*256.0f);
		if( tmp>255 ){
			out_img[i] = 255;
		} else if( tmp<0 ){
			out_img[i] = 0;
		} else {
			out_img[i] = (unsigned char)tmp;
		}
	}
}
