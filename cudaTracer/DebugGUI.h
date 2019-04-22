#ifndef	DebugGUI_h
#define	DebugGUI_h

#include <sstream>
#include <string>
#include <vector>

#include <windows.h>
#include <windowsx.h>
#include <d3d11.h>

#include <AntTweakBar.h>

#include "DeviceHandler.h"

using namespace std;

// Pre def
class DeviceHandler;

class DebugGUI
{
	//=========================================================================
	// Variables
	//=========================================================================
private:
	//vector<TwBar*> m_bars;
	//vector<string> m_barNames;

public:
	enum Types
	{
		DG_BOOL		= TW_TYPE_BOOLCPP,
		DG_INT		= TW_TYPE_INT32,
		DG_FLOAT	= TW_TYPE_FLOAT,
		DG_COLOR	= TW_TYPE_COLOR4F,
		DG_VEC3		= TW_TYPE_DIR3F
	};
	enum Permissions	{ READ_ONLY, READ_WRITE };
	enum Result			{ FAILED, SUCCESS };

	//=========================================================================
	// Functions
	//=========================================================================
private:
	DebugGUI();
	DebugGUI(DebugGUI const&);			// Don't Implement
	void operator=(DebugGUI const&);		// Don't implement

	string stringFromParams( string p_barName, string p_varName,
		string p_paramName, int p_arg);
	string stringFromParams( string p_barName, string p_varName,
		string p_paramName, int p_arg1, int p_arg2);
	string stringFromParams( string p_barName, string p_varName,
		string p_paramName, vector<int> p_args);

public:
	TwBar* barFromString( string p_barName ); // Should be private

	static DebugGUI* getInstance();
	void init( ID3D11Device* p_device, int p_wndWidth, int p_wndHeight );
	Result addVar( string p_barName, Types p_type, Permissions p_permissions,
		string p_name, void *p_var, string p_options );

	void setSize( string p_barName, int p_x, int p_y );
	void setPosition( string p_barName, int p_x, int p_y);

	/** Returns zero on fail and nonzero on success as per TwEventWin */
	int updateMsgProc( HWND wnd, UINT msg, WPARAM wParam, LPARAM lParam );
	void draw();
	void terminate();
	void setBarVisibility( string p_bar, bool p_show );
	void setBarIconification( string p_bar, bool p_iconify );
};

#endif	//DebugGUI_h