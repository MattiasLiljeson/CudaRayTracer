#pragma once

#include <d3d11.h>
//#include <d3dx11.h>

#include <string>
#include <stdint.h>

#include "DeviceHandler.h"

using namespace std;

class ShaderSet
{
public:
	uint8_t* m_vsData;
	int m_vsDataSize;
	ID3D11VertexShader* m_vs;

	uint8_t* m_psData;
	int m_psDataSize;
	ID3D11PixelShader* m_ps;

private:
	DeviceHandler* m_deviceHandler;

public:
	ShaderSet( DeviceHandler* p_deviceHandler );
	~ShaderSet();

	void createSet( string p_filePath, string p_vsEntry, string p_psEntry );
private:

	void createVs( uint8_t* p_vsData, int p_vsDataSize );
	void createPs( uint8_t* p_psData, int p_psDataSize );
	 
	// WARNING. Allocates mem for the read sahder which is returned through
	// param. Must be deleted after use!
	void readShader( const string &p_sourceFilePath, uint8_t*& out_data, int& out_size );

	/*ID3DBlob* readShader( const string &p_sourceFilePath );
	void ShaderSet::compileShader( const string &p_sourceFile,
		const string &p_entryPoint, const string &p_profile,
		ID3DBlob** out_blob );*/
};

