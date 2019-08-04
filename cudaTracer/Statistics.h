#pragma once

#include "DebugGUI.h"
#include "Service.h"

class Statistics : public Service {
    float latestDt;
    float latestFps;
    int mouseY;
    int mouseX;

   public:
    Statistics() {
        latestDt = 0.0f;
        latestFps = 0.0f;
        mouseY = 0;
        mouseX = 0;
        DebugGUI* dg = ServiceRegistry::instance().get<DebugGUI>();

        dg->setSize("Mouse", 200, 1400);
        dg->setPosition("Mouse", 420, 0);
        dg->setVisible("Mouse", false);

        
        dg->setSize("Statistics", 150, 50);
        dg->setPosition("Statistics", 0, 0);

        dg->addVar("Statistics", DebugGUI::DG_FLOAT, DebugGUI::READ_ONLY, "dt",
                   &latestDt);
        dg->addVar("Statistics", DebugGUI::DG_FLOAT, DebugGUI::READ_ONLY, "fps",
                   &latestFps);
        dg->addVar("Mouse", DebugGUI::DG_INT, DebugGUI::READ_ONLY, "X",
                   &mouseX);
        dg->addVar("Mouse", DebugGUI::DG_INT, DebugGUI::READ_ONLY, "Y",
                   &mouseY);
    }

    void update(float dt) {
        latestDt = dt;
        latestFps = 1.0f / dt;
        InputHandler* input = ServiceRegistry::instance().get<InputHandler>();
        mouseX = input->getMouse(InputHandler::X);
        mouseY = input->getMouse(InputHandler::Y);
    }
};