#pragma once

#include "Service.h"
#include "DebugGUI.h"

class Statistics : public Service {
    float latestDt;
    float latestFps;

   public:
    Statistics() {
        latestDt = 0.0f;
        latestFps = 0.0f;
        DebugGUI* dg = ServiceRegistry::getInstance().get<DebugGUI>();

        dg->addVar("Statistics", DebugGUI::DG_FLOAT, DebugGUI::READ_ONLY, "dt",
                   &latestDt);
        dg->addVar("Statistics", DebugGUI::DG_FLOAT, DebugGUI::READ_ONLY, "fps",
                   &latestFps);
    }

    void update(float dt) {
        latestDt = dt;
        latestFps = 1.0f / dt;
    }
};