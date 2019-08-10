#include "preProc.h"

#include <D3D11SDKLayers.h>
#include <cuda_d3d11_interop.h>
#include <cuda_runtime_api.h>
#include <lodepng.h>
#include "Camera.h"
#include "D3DDebugger.h"
#include "DebugGUI.h"
#include "DeviceHandler.h"
#include "ExperimentReader.h"
#include "PicDumper.h"
#include "Popup.h"
#include "Profiler.h"
#include "RayTracer.h"
#include "Statistics.h"
#include "TextureRenderer.h"
#include "cudaUtils.h"
#include "globals.h"
#include "guicon.h"
#include "thrust\system\system_error.h"

void prepareServices(HINSTANCE hInstance, DeviceHandler* deviceHandler,
                     Options options) {
    static DebugGUI dg(deviceHandler->getDevice(),
                       deviceHandler->getWindowWidth(),
                       deviceHandler->getWindowHeight());
    static InputHandler input(&hInstance, deviceHandler->getHWnd());
    static TextureRenderer texRender(deviceHandler, options.device.width,
                                     options.device.height);
    static RayTracer tracer(texRender.getTextureSet(), options.device.width,
                            options.device.height, options);
    static Statistics stats;
    static PicDumper dumper(options.device.width, options.device.height);

    ServiceRegistry::instance().add<InputHandler>(&input);
    ServiceRegistry::instance().add<RayTracer>(&tracer);
    ServiceRegistry::instance().add<TextureRenderer>(&texRender);
    ServiceRegistry::instance().add<Statistics>(&stats);
    ServiceRegistry::instance().add<PicDumper>(&dumper);
    ServiceRegistry::instance().add<DebugGUI>(&dg);
}

Options loadDefaults() {
    Options options;
    options.host.lightCnt = 10;
    options.host.fov = 90;
    options.device.width = 1024;
    options.device.height = 768;
    options.device.maxDepth = 5;
    options.device.samples = 1;
    options.device.backgroundColor = Vec3f(0.2f, 0.7f, 0.8f);
    options.device.shadowBias = 0.0001f;
    options.device.blockSize = 4;
    options.device.useMeshBvhs = true;
    options.device.useSceneBvh = true;


    options.device.scale = tan(deg2rad(options.host.fov * 0.5f));
    options.device.imageAspectRatio =
        options.device.width / (float)options.device.height;
    options.device.gammaCorrection = true;
    return options;
}

#ifndef _TEST
INT WinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, LPSTR lpCmdLine,
            int nCmdShow) {
    RedirectIOToConsole();
    Options options = loadDefaults();

    bool doExperiment = !string(lpCmdLine).empty();
    if (!doExperiment) {
        // HACK: add space for borders
        int wndWidth = options.device.width + 16;
        // HACK: add space for borders and header
        int wndHeight = options.device.height + 39;

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
    } else {
        std::stringstream stream;
        stream << lpCmdLine;
        std::string experimentFname;
        int phaseIdx = 0;
        if (stream.rdbuf()->in_avail() != 0) {
            stream >> experimentFname;
            stream >> phaseIdx;
        }
        ExperimentReader reader(experimentFname, options);
        ExperimentParameters exp = reader.experiment;
        cout << reader;
        Profiler* profiler = Profiler::instance();
        const int experimentRounds = 1;

        std::cout << "running experiment: " << exp.experimentName << std::endl;
        ExperimentPhase phase = exp.phases[phaseIdx];
        std::cout << "\tPhase: " << phase.phaseName << std::endl;
        profiler->addPerfTimer(phase.phaseName, exp.experimentName, true);
        // HACK: add space for borders
        int wndWidth = options.device.width + 16;
        // HACK: add space for borders and header
        int wndHeight = options.device.height + 39;

        DeviceHandler deviceHandler(hInstance, wndWidth, wndHeight);
        D3DDebugger d3dDbg(deviceHandler.getDevice());
        prepareServices(hInstance, &deviceHandler, phase.opts);
        for (int j = 0; j < experimentRounds; ++j) {
            profiler->start(phase.phaseName, exp.experimentName, true);
            deviceHandler.beginDrawing();
            ServiceRegistry::instance().updateAll(0.016f);
            ServiceRegistry::instance().get<PicDumper>()->toDisk(
                ServiceRegistry::instance().get<TextureRenderer>());
            deviceHandler.presentFrame();
            profiler->stop(phase.phaseName, exp.experimentName, true);
        }

        profiler->logTimersToFile(false, exp.experimentName);
        // d3dDbg.reportLiveDeviceObjects();
    }
    return 0;
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
