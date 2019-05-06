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

// 2^n + 1. For diamond square algorithm

// Possible resolutions
//#define PIC_WIDTH 4097
//#define PIC_HEIGHT 4097

//#define PIC_WIDTH 2049
//#define PIC_HEIGHT 2049

//#define PIC_WIDTH 1025
//#define PIC_HEIGHT 1025

//#define PIC_WIDTH 513
//#define PIC_HEIGHT 513

#define PIC_WIDTH 640
#define PIC_HEIGHT 480
//#define PIC_HEIGHT 1440

void doView(HINSTANCE hInstance, HINSTANCE hPrevInstance, LPSTR lpCmdLine,
            int nCmdShow);
void copyFloatsToCharVector(vector<float>& p_arr,
                            vector<unsigned char>& out_img);

void dumpPicToDisk(TextureRenderer* texRender) {
    DeviceHandler::g_returnPressed = false;
    static int cnt = 0;
    cnt++;
    char buf[128];
    sprintf(buf, "R%.3d-%ix%i.png", cnt, PIC_WIDTH, PIC_HEIGHT);
    string timerName(buf);

    unsigned int picSize = PIC_WIDTH * 4 * PIC_HEIGHT;
    vector<float> arr;
    arr.resize(picSize);
    texRender->copyToHostArray(&arr[0]);
    vector<unsigned char> img;
    img.resize(picSize);
    copyFloatsToCharVector(arr, img);
    unsigned error = lodepng::encode(buf, img, PIC_WIDTH, PIC_HEIGHT);
}

void update() {

}

#ifndef _TEST
int WINAPI WinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance,
                   LPSTR lpCmdLine, int nCmdShow) {
    Profiler* prof = Profiler::getInstance();

    int wndWidth = PIC_WIDTH + 16;    // HACK: add space for borders
    int wndHeight = PIC_HEIGHT + 39;  // HACK: add space for borders and header
    DeviceHandler* deviceHandler =
        new DeviceHandler(hInstance, wndWidth, wndHeight);
    D3DDebugger d3dDbg(deviceHandler->getDevice());
    DebugGUI::getInstance()->init(deviceHandler->getDevice(), wndWidth, wndHeight);
    DebugGUI::getInstance()->setSize("Camera", 200, 1400);
    DebugGUI::getInstance()->setPosition("Camera", 0, 0);
    DebugGUI::getInstance()->setVisible("Camera", false);

    DebugGUI::getInstance()->setSize("Rays", 200, 1400);
    DebugGUI::getInstance()->setPosition("Rays", 220, 0);
    DebugGUI::getInstance()->setVisible("Rays", false);
    DebugGUI::getInstance()->setSize("Mouse", 200, 1400);
    DebugGUI::getInstance()->setPosition("Mouse", 420, 0);
    DebugGUI::getInstance()->setVisible("Mouse", false);

    TextureRenderer* texRender = nullptr;
    InputHandler* input = new InputHandler(&hInstance, deviceHandler->getHWnd());
    Camera* camera = new Camera();

    try {
        texRender = new TextureRenderer(deviceHandler, PIC_WIDTH, PIC_HEIGHT, input, camera );
    } catch (thrust::system_error e) {
        Utils::error(e.what());
    }

    if (texRender) {
        Timer timer;
        timer.reset();
        MSG msg = {0};
        while (msg.message != WM_QUIT) {
            if (PeekMessage(&msg, NULL, 0, 0, PM_REMOVE)) {
                TranslateMessage(&msg);
                DispatchMessage(&msg);
            } else {
                update();
                deviceHandler->beginDrawing();
                try {
                        timer.tick();
                        texRender->update(timer.getDt());
                     if (DeviceHandler::g_returnPressed) {
                        dumpPicToDisk(texRender);
                    }
                    texRender->draw();
                } catch (thrust::system_error e) {
                    Utils::error(e.what());
                }

                DebugGUI::getInstance()->draw();
                deviceHandler->presentFrame();
            }
        }
    }
    delete input;
    delete texRender;
    delete deviceHandler;
    d3dDbg.reportLiveDeviceObjects();
}

#else

#define CATCH_CONFIG_RUNNER
//#define CATCH_CONFIG_MAIN  // This tells Catch to provide a main() - only do this in one cpp file
#include "../Catch2/single_include/catch2/catch.hpp"
#include <iostream>
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

void copyFloatsToCharVector(vector<float>& p_arr,
                            vector<unsigned char>& out_img) {
    for (unsigned int i = 0; i < p_arr.size(); i++) {
        int tmp = (int)(p_arr[i] * 256.0f);
        if (tmp > 255) {
            out_img[i] = 255;
        } else if (tmp < 0) {
            out_img[i] = 0;
        } else {
            out_img[i] = (unsigned char)tmp;
        }
    }
}
