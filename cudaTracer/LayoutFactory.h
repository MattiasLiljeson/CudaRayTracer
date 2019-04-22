#pragma once

#include <d3d11.h>
//#include <d3dx11.h>

struct LayoutDesc
{
	int m_elementCnt;
	D3D11_INPUT_ELEMENT_DESC* m_layoutPtr;
};

class LayoutFactory
{
private:
	static const int s_POINT_TEXCOORD_CNT = 2;
	static const D3D11_INPUT_ELEMENT_DESC s_pointTexcoord[s_POINT_TEXCOORD_CNT];

public:
	static LayoutDesc getPointTexCoordDesc()
	{
		LayoutDesc desc;
		desc.m_elementCnt = s_POINT_TEXCOORD_CNT;
		desc.m_layoutPtr = (D3D11_INPUT_ELEMENT_DESC*)s_pointTexcoord;
		return desc;
	}
};

const D3D11_INPUT_ELEMENT_DESC LayoutFactory::s_pointTexcoord[] =
{
	{"POSITION", 0, DXGI_FORMAT_R32G32B32_FLOAT,	0, 0,  D3D11_INPUT_PER_VERTEX_DATA,   0 },
	{"TEXCOORD", 0, DXGI_FORMAT_R32G32_FLOAT,		0, 12, D3D11_INPUT_PER_VERTEX_DATA,   0 },
};


//class I_Layout 
//{
//public:
//	virtual D3D11_INPUT_ELEMENT_DESC* getLayoutPtr() = 0;
//	virtual unsigned int getElementCnt() = 0;
//};
//
//class PointTexcoord : public I_Layout
//{
//private:
//	static const int s_ELEMENT_CNT = 2;
//	static const D3D11_INPUT_ELEMENT_DESC s_inputDesc[s_ELEMENT_CNT];
//public:
//	D3D11_INPUT_ELEMENT_DESC* getLayoutPtr() {
//		return (D3D11_INPUT_ELEMENT_DESC*)s_inputDesc;
//	}
//	unsigned int getElementCnt(){
//		return s_ELEMENT_CNT;
//	}
//};
//
//const D3D11_INPUT_ELEMENT_DESC PointTexcoord::s_inputDesc[] =
//{
//	{"POSITION", 0, DXGI_FORMAT_R32G32B32_FLOAT,	0, 0,  D3D11_INPUT_PER_VERTEX_DATA,   0 },
//	{"TEXCOORD", 0, DXGI_FORMAT_R32G32_FLOAT,		0, 12, D3D11_INPUT_PER_VERTEX_DATA,   0 },
//};