#include "ShaderSet.h"

#include <windows.h>
#include "Utils.h"
#include <D3Dcompiler.h>
#include <fstream>

ShaderSet::ShaderSet( DeviceHandler* p_deviceHandler )
{
	m_deviceHandler = p_deviceHandler;

	m_vsData = nullptr;
	int vsDataSize = -1;
	m_vs = nullptr;
	m_psData = nullptr;
	int psDataSize = -1;
	m_ps = nullptr;
}

ShaderSet::~ShaderSet()
{
	SAFE_RELEASE( m_vs );
	SAFE_RELEASE( m_ps );
	delete [] m_vsData;
	delete [] m_psData;
}

void ShaderSet::createSet( string p_filePath, string p_vsEntry, string p_psEntry )
{
	readShader( "../x64/release/regularVs.cso", m_vsData, m_vsDataSize ); 
	if( m_vsData != nullptr ){
		createVs( m_vsData, m_vsDataSize );
	}

	readShader( "../x64/release/regularPs.cso", m_psData, m_psDataSize ); 
	if( m_psData != nullptr ){
		createPs( m_psData, m_psDataSize );
	}
}

void ShaderSet::readShader(const string& p_sourceFilePath, uint8_t*& out_data, int& out_size) {
	ifstream ifs(p_sourceFilePath, ifstream::in | ifstream::binary);
	if (ifs.good()) {
		ifs.seekg(0, ios::end);
		out_size = ifs.tellg();
		out_data = new uint8_t[out_size];
		ifs.seekg(0, ios::beg);
		ifs.read((char*)& out_data[0], out_size);
	}
	else {
		string msg = string("Could not read shader: ") + p_sourceFilePath;
		Utils::error(__FILE__, __FUNCTION__, __LINE__, msg);
	}
	ifs.close();
}

void ShaderSet::createVs( uint8_t* p_vsData, int p_vsDataSize )
{
	HR(m_deviceHandler->getDevice()->CreateVertexShader(
		p_vsData, p_vsDataSize, NULL, &m_vs));
}

void ShaderSet::createPs( uint8_t* p_psData, int p_psDataSize )
{
	HR(m_deviceHandler->getDevice()->CreatePixelShader(
		p_psData, p_psDataSize,	NULL, &m_ps));
}