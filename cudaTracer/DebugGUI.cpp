#include "DebugGUI.h"

//=========================================================================
// Private functions
//=========================================================================
DebugGUI::DebugGUI()
{
}

TwBar* DebugGUI::barFromString( string p_barName )
{
	TwBar* bar = TwGetBarByName(p_barName.c_str());

	if(bar == NULL)
	{
		bar = TwNewBar(p_barName.c_str());
		//TwDefine("DebugGUI size='380 720' position='900 0'");
	}
	return bar;
}

string DebugGUI::stringFromParams( string p_barName, string p_varName,
								  string p_paramName, int p_arg)
{
	vector<int> args;
	args.push_back(p_arg);
	return stringFromParams(p_barName, p_varName, p_paramName, args);
}

string DebugGUI::stringFromParams( string p_barName, string p_varName,
								  string p_paramName, int p_arg1, int p_arg2)
{
	vector<int> args;
	args.push_back(p_arg1);
	args.push_back(p_arg2);
	return stringFromParams(p_barName, p_varName, p_paramName, args);
}

string DebugGUI::stringFromParams( string p_barName, string p_varName,
								  string p_paramName, vector<int> p_args )
{
	barFromString(p_barName);

	stringstream ss;
	ss << p_barName;
	if(p_varName.length() > 0)
	{
		ss<<"/"<<p_varName;
	}
	ss<<" "<<p_paramName<<"=";

	// Create space separated string list of args surrounded by '.
	ss<<"'"<<p_args[0];
	for( int i=1; i<(int)p_args.size(); i++ )
	{
		ss<<" "<<p_args[i];
	}
	ss<<"'";

	string result = ss.str();
	return ss.str();
}

//=========================================================================
// Public functions
//=========================================================================
DebugGUI* DebugGUI::getInstance()
{
	// Instantiated on first use. Guaranteed to be destroyed.
	static DebugGUI instance;
	return &instance;
}

void DebugGUI::init( ID3D11Device* p_device, int p_wndWidth, int p_wndHeight )
{
	if( !TwInit( TW_DIRECT3D11, p_device ) ){
		Utils::error( __FILE__, __FUNCTION__, __LINE__, TwGetLastError() );
	}
	if( !TwWindowSize( p_wndWidth, p_wndHeight ) ){
		Utils::error( __FILE__, __FUNCTION__, __LINE__, TwGetLastError() );
	}
}

DebugGUI::Result DebugGUI::addVar( string p_barName, Types p_type,
								  Permissions p_permissions, string p_name, void *p_var, string p_options )
{
	TwBar* bar = barFromString(p_barName);

	Result result = FAILED;
	if( p_permissions == READ_ONLY ){
		if( TwAddVarRO( bar, p_name.c_str(), (TwType)p_type, p_var, p_options.c_str() ) ){
			result = SUCCESS;
		}
	}
	else if( p_permissions == READ_WRITE ){
		if( TwAddVarRW (bar, p_name.c_str(), (TwType)p_type, p_var, p_options.c_str() ) ){
			result = SUCCESS;
		}
	}

	// Set value column width to optimal
	TwSetParam( bar, NULL, "valueswidth", TW_PARAM_CSTRING, 1, "fit" );

	return result;
}

void DebugGUI::setSize( string p_barName, int p_x, int p_y )
{
	int result;
	result = TwDefine(stringFromParams( p_barName, "", "size", p_x, p_y ).c_str());
}

void DebugGUI::setPosition( string p_barName, int p_x, int p_y )
{
	int result;
	result = TwDefine(stringFromParams( p_barName, "", "position", p_x, p_y ).c_str());
}

void DebugGUI::draw()
{
#ifdef USE_DEBUG_GUI
	if( !TwDraw() ){
		Utils::error( __FILE__, __FUNCTION__, __LINE__, TwGetLastError() );
	}
#endif
}

int DebugGUI::updateMsgProc( HWND wnd, UINT msg, WPARAM wParam, LPARAM lParam )
{
	return TwEventWin(wnd, msg, wParam, lParam);
}

void DebugGUI::terminate()
{
	TwTerminate();
}

void DebugGUI::setBarVisibility( string p_bar, bool p_show )
{
	string msg = p_bar;
	if( p_show ) {
		msg += " visible=true";
	} else {
		msg += " visible=false";
	}
	TwDefine( msg.c_str() );  // mybar is hidden
}

void DebugGUI::setBarIconification( string p_bar, bool p_iconify )
{
	string msg = " " + p_bar;
	if( p_iconify ) {
		msg += " iconified=true ";
	} else {
		msg += " iconified=false ";
	}
	TwDefine( msg.c_str() );  // mybar is hidden
}