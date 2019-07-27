#pragma once
#include <string>
#include <d3d11.h>
class Popup {
   public:
    static void wstringFromString(std::wstring& ws, const std::string& s) {
        std::wstring wsTmp(s.begin(), s.end());
        ws = wsTmp;
    }

    static void stringFromWstring(const std::wstring& ws, std::string& s) {
        std::string sTmp(ws.begin(), ws.end());
        s = sTmp;
    }

    static void error(const std::string& p_file, const std::string& p_function,
                      int p_line, const std::string& p_info) {
        char msg[256];
        sprintf(msg, "%s @ %s:%d, ERROR: %s", p_function.c_str(),
                p_file.c_str(), p_line, p_info.c_str());
        std::wstring msgAsW = L"";
        Popup::wstringFromString(msgAsW, msg);

        MessageBox(NULL, msgAsW.c_str(), L"Error", MB_OK | MB_ICONEXCLAMATION);
    }

    static void error(const std::string& p_info) {
        char msg[256];
        sprintf(msg, "ERROR: %s", p_info.c_str());
        std::wstring msgAsW = L"";
        Popup::wstringFromString(msgAsW, msg);

        MessageBox(NULL, msgAsW.c_str(), L"Error", MB_OK | MB_ICONEXCLAMATION);
    }
};