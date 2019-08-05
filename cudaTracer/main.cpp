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

void prepareServices(HINSTANCE hInstance, DeviceHandler* deviceHandler,
                     Options options) {
    static DebugGUI dg(deviceHandler->getDevice(),
                       deviceHandler->getWindowWidth(),
                       deviceHandler->getWindowHeight());
    static InputHandler input(&hInstance, deviceHandler->getHWnd());
    static TextureRenderer texRender(deviceHandler, options.width,
                                     options.height);
    static RayTracer tracer(texRender.getTextureSet(), options.width,
                            options.height, options);
    static Statistics stats;
    static PicDumper dumper(options.height, options.height);

    ServiceRegistry::instance().add<InputHandler>(&input);
    ServiceRegistry::instance().add<RayTracer>(&tracer);
    ServiceRegistry::instance().add<TextureRenderer>(&texRender);
    ServiceRegistry::instance().add<Statistics>(&stats);
    ServiceRegistry::instance().add<PicDumper>(&dumper);
    ServiceRegistry::instance().add<DebugGUI>(&dg);
}

#ifndef _TEST
int WINAPI WinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance,
                   LPSTR lpCmdLine, int nCmdShow) {
    Options options;
    options.width = 640;
    options.height = 480;
    options.fov = 90;
    options.maxDepth = 5;
    options.samples = 1;
    options.backgroundColor = Vec<float, 3>(0.2f, 0.7f, 0.8f);
    options.shadowBias = 0.0001f;
    options.scale = tan(deg2rad(options.fov * 0.5f));
    options.imageAspectRatio = options.width / (float)options.height;

    // HACK: add space for borders
    int wndWidth = options.width + 16;
    // HACK: add space for borders and header
    int wndHeight = options.height + 39;

    DeviceHandler deviceHandler(hInstance, wndWidth, wndHeight);
    D3DDebugger d3dDbg(deviceHandler.getDevice());
    prepareServices(hInstance, &deviceHandler, options);

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
            ServiceRegistry::instance().updateAll(timer.getDt());
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
