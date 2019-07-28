#include "preProc.h"

#include "DeviceHandler.h"
#include "cudaUtils.h"

#include <cuda_d3d11_interop.h>
#include <cuda_runtime_api.h>
#include "TextureRenderer.h"
#include "globals.h"

#include <D3D11SDKLayers.h>
#include "D3DDebugger.h"

#include "DebugGUI.h"
#include "Profiler.h"
#include "thrust\system\system_error.h"

#include <lodepng.h>
#include "Camera.h"

#include "PicDumper.h"

#include "Popup.h"
#include "RayTracer.h"
#include "Statistics.h"

#define PIC_WIDTH 640
#define PIC_HEIGHT 480

void prepareServices(HINSTANCE hInstance, DeviceHandler* deviceHandler) {

    static InputHandler input(&hInstance, deviceHandler->getHWnd());
    static TextureRenderer texRender(deviceHandler, PIC_WIDTH, PIC_HEIGHT);
    static RayTracer tracer(texRender.getTextureSet(), PIC_WIDTH, PIC_HEIGHT,
                            &input);
    static Statistics stats;
    static PicDumper dumper(PIC_WIDTH, PIC_HEIGHT);
    static DebugGUI dg(deviceHandler->getDevice(),
                       deviceHandler->getWindowHeight(),
                       deviceHandler->getWindowHeight());

    ServiceRegistry::getInstance().add<InputHandler>(&input);
    ServiceRegistry::getInstance().add<RayTracer>(&tracer);
    ServiceRegistry::getInstance().add<TextureRenderer>(&texRender);
    ServiceRegistry::getInstance().add<Statistics>(&stats);
    ServiceRegistry::getInstance().add<PicDumper>(&dumper);
    ServiceRegistry::getInstance().add<DebugGUI>(&dg);
}

#ifndef _TEST
int WINAPI WinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance,
                   LPSTR lpCmdLine, int nCmdShow) {
    int wndWidth = PIC_WIDTH + 16;    // HACK: add space for borders
    int wndHeight = PIC_HEIGHT + 39;  // HACK: add space for borders and header
    DeviceHandler deviceHandler(hInstance, wndWidth, wndHeight);
    D3DDebugger d3dDbg(deviceHandler.getDevice());
    prepareServices(hInstance, &deviceHandler);

    Timer timer;
    timer.reset();
    MSG msg = {0};
    while (msg.message != WM_QUIT) {
        if (PeekMessage(&msg, NULL, 0, 0, PM_REMOVE)) {
            TranslateMessage(&msg);
            DispatchMessage(&msg);
        } else {
            deviceHandler.beginDrawing();
            timer.tick();
            ServiceRegistry::getInstance().updateAll(timer.getDt());
            deviceHandler.presentFrame();
        }
    }
    d3dDbg.reportLiveDeviceObjects();
}

#else

#define CATCH_CONFIG_RUNNER
//#define CATCH_CONFIG_MAIN  // This tells Catch to provide a main() - only do
// this in one cpp file
#include <iostream>
#include "../Catch2/single_include/catch2/catch.hpp"
int main(int argc, char** argv) {
    // If the last argument is "-p", then pause after the tests are run.
    // This allows us to run "leaks" on Mac OS X to check for memory leaks.
    bool pause_after_test = true;
    if (argc && std::string(argv[argc - 1]) == "-p") {
        pause_after_test = true;
        argc--;
    }

    int result = Catch::Session().run(argc, argv);

    if (pause_after_test) {
        printf("Press enter to continue.");
        std::string s;
        std::cin >> s;
    }

    return result;
}

#endif
