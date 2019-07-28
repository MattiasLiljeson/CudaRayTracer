#ifndef INPUTHANDLER_H
#define INPUTHANDLER_H

#include <dinput.h>

#pragma comment (lib, "dinput8.lib")
#pragma comment (lib, "dxguid.lib")

class InputHandler
{
private:
	HINSTANCE* hInstance;
	HWND* hWnd;
	LPDIRECTINPUT8 din;    // the pointer to our DirectInput interface
	LPDIRECTINPUTDEVICE8 dinkeyboard;    // the pointer to the keyboard device
	LPDIRECTINPUTDEVICE8 dinmouse;    // the pointer to the mouse device
	BYTE keystate[256];    // the storage for the key-information
	DIMOUSESTATE mousestate;    // the storage for the mouse-information

	bool keys[10];

	long mouse[2];

public:
	enum KEYBOARD{W,A,S,D,SPACE,LCTRL,TAB,F1,F2,F3,F4};
	enum MOUSE{X,Y};

	InputHandler(HINSTANCE* hInstance, HWND* hWnd); // sets up and initializes DirectInput
	~InputHandler();	//closes DirectInput and releases memory
	void reset();
	void detectInput(void);    // gets the current input state
	void update();
	bool getKey(int key);
	long getMouse(int axis);
};

#endif //INPUTHANDLER_H