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
    static TextureRenderer texRender(deviceHandler, options.device.width,
                                     options.device.height);
    static RayTracer tracer(texRender.getTextureSet(), options.device.width,
                            options.device.height, options);
    static Statistics stats;
    static PicDumper dumper(options.device.height, options.device.height);

    ServiceRegistry::instance().add<InputHandler>(&input);
    ServiceRegistry::instance().add<RayTracer>(&tracer);
    ServiceRegistry::instance().add<TextureRenderer>(&texRender);
    ServiceRegistry::instance().add<Statistics>(&stats);
    ServiceRegistry::instance().add<PicDumper>(&dumper);
    ServiceRegistry::instance().add<DebugGUI>(&dg);
}

struct ExperimentParameters {
    std::string experimentName;
    std::string phaseName;
    Options options;
};

std::vector<std::function<ExperimentParameters(Options, int)>>
createExperiments() {
    std::vector<std::function<ExperimentParameters(Options, int)>> experiments;
    {
        auto blockSize = [](Options o, int i) -> ExperimentParameters {
            int blocksize[5] = {4, 8, 12, 16, 20};
            o.device.blockSize = blocksize[i];
            const int BUFF_SIZE = 128;
            char buf[BUFF_SIZE];
            sprintf_s(buf, BUFF_SIZE, "block_size_%d", blocksize[i]);
            return {"block_size", std::string(buf), o};
        };
        experiments.push_back(blockSize);
    }
    {
        auto resolutionTest = [](Options o, int i) -> ExperimentParameters {
            Vec2i resolutions[5] = {
                {200, 150}, {400, 300}, {600, 450}, {800, 600}, {1000, 750}};
            o.device.width = resolutions[i][X];
            o.device.height = resolutions[i][Y];
            const int BUFF_SIZE = 128;
            char buf[BUFF_SIZE];
            sprintf_s(buf, BUFF_SIZE, "resolution_%d-%d", resolutions[i][X],
                      resolutions[i][Y]);
            return {"resolution", std::string(buf), o};
        };
        experiments.push_back(resolutionTest);
    }
    {
        auto maxTraceDepth = [](Options o, int i) -> ExperimentParameters {
            int depths[5] = {1, 2, 3, 4, 5};
            o.device.maxDepth = depths[i];
            const int BUFF_SIZE = 128;
            char buf[BUFF_SIZE];
            sprintf_s(buf, BUFF_SIZE, "max_trace_depth_%d", depths[i]);
            return {"max_trace_depth", std::string(buf), o};
        };
        experiments.push_back(maxTraceDepth);
    }
    {
        auto lights = [](Options o, int i) -> ExperimentParameters {
            int lights[5] = {10, 20, 30, 40, 50};
            o.host.lightCnt = lights[i];
            const int BUFF_SIZE = 128;
            char buf[BUFF_SIZE];
            sprintf_s(buf, BUFF_SIZE, "lights_%d", lights[i]);
            return {"lights", std::string(buf), o};
        };
        experiments.push_back(lights);
    }
    {
        auto triangles = [](Options o, int i) -> ExperimentParameters {
            pair<std::string, std::string> models[5] = {
                {"../assets/models/metalBarrel/", "metal_barrel.obj"},
                {"../assets/models/metalBarrel/", "metal_barrel.obj"},
                {"../assets/models/metalBarrel/", "metal_barrel.obj"},
                {"../assets/models/metalBarrel/", "metal_barrel.obj"},
                {"../assets/models/metalBarrel/", "metal_barrel.obj"},
            };
            o.host.modelFolder = models[i].first;
            o.host.modelFname = models[i].second;
            const int BUFF_SIZE = 128;
            char buf[BUFF_SIZE];
            sprintf_s(buf, BUFF_SIZE, "model_%s", models[i].second);
            return {"model", std::string(buf), o};
        };
        experiments.push_back(triangles);
    }
    return experiments;
}

#ifndef _TEST
int WINAPI WinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance,
                   LPSTR lpCmdLine, int nCmdShow) {
    Options options;
    options.host.lightCnt = 10;
    options.host.fov = 90;
    options.device.width = 800;
    options.device.height = 600;
    options.device.maxDepth = 5;
    options.device.samples = 1;
    options.device.backgroundColor = Vec3f(0.2f, 0.7f, 0.8f);
    options.device.shadowBias = 0.0001f;
    options.device.scale = tan(deg2rad(options.host.fov * 0.5f));
    options.device.imageAspectRatio =
        options.device.width / (float)options.device.height;
    options.device.gammaCorrection = true;
    options.device.blockSize = 16;

    auto experiments = createExperiments();

    bool doExperiment = true;
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
        Profiler* profiler = Profiler::instance();
        const int experimentRounds = 100;
        const int experimentStages = 5;
        for (auto& exp : experiments) {
            for (int i = 0; i < experimentStages; ++i) {
                ExperimentParameters e = exp(options, i);
                profiler->addPerfTimer(e.phaseName, e.experimentName, true);
                // HACK: add space for borders
                int wndWidth = options.device.width + 16;
                // HACK: add space for borders and header
                int wndHeight = options.device.height + 39;

                DeviceHandler deviceHandler(hInstance, wndWidth, wndHeight);
                D3DDebugger d3dDbg(deviceHandler.getDevice());
                prepareServices(hInstance, &deviceHandler, e.options);
                for (int j = 0; j < experimentRounds; ++j) {
                    profiler->start(e.phaseName, e.experimentName, true);
                    deviceHandler.beginDrawing();
                    ServiceRegistry::instance().updateAll(0.016f);
                    deviceHandler.presentFrame();
                    profiler->stop(e.phaseName, e.experimentName, true);
                }
            }
        }
        profiler->logTimersToFile(true, true, "results");
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
