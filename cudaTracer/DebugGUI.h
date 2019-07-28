#ifndef DebugGUI_h
#define DebugGUI_h

#include <sstream>
#include <string>
#include <vector>

#include <d3d11.h>
#include <windows.h>
#include <windowsx.h>

#include <AntTweakBar.h>

#include "DeviceHandler.h"

#include "Service.h"

using namespace std;

// Pre def
class DeviceHandler;

class DebugGUI : public Service {
   public:
    enum Types {
        DG_BOOL = TW_TYPE_BOOLCPP,
        DG_INT = TW_TYPE_INT32,
        DG_CHAR = TW_TYPE_INT8,
        DG_FLOAT = TW_TYPE_FLOAT,
        DG_COLOR = TW_TYPE_COLOR4F,
        DG_VEC3 = TW_TYPE_DIR3F
    };
    enum Permissions { READ_ONLY, READ_WRITE };
    enum Result { FAILED, SUCCESS };

    DebugGUI(ID3D11Device* p_device, int p_wndWidth, int p_wndHeight);

    TwBar* barFromString(string p_barName);  // Should be private

    Result addVar(string p_barName, Types p_type, Permissions p_permissions,
                  string p_name, void* p_var);
    Result addVar(string p_barName, Types p_type, Permissions p_permissions,
                  string p_name, void* p_var, string p_options);

    void setSize(string p_barName, int p_x, int p_y);
    void setPosition(string p_barName, int p_x, int p_y);
    void setVisible(string p_barName, bool visible);

    /** Returns zero on fail and nonzero on success as per TwEventWin */
    int updateMsgProc(HWND wnd, UINT msg, WPARAM wParam, LPARAM lParam);
    void update(float dt);
    void terminate();
    void setBarVisibility(string p_bar, bool p_show);
    void setBarIconification(string p_bar, bool p_iconify);

   private:
    string stringFromParams(string p_barName, string p_varName,
                            string p_paramName, int p_arg);
    string stringFromParams(string p_barName, string p_varName,
                            string p_paramName, int p_arg1, int p_arg2);
    string stringFromParams(string p_barName, string p_varName,
                            string p_paramName, vector<int> p_args);
};

#endif  // DebugGUI_h