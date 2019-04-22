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
	//compileShader( p_filePath, p_vsEntry, "vs_5_0", &m_vsData );
	readShader( "../x64/release/regularVs.cso", m_vsData, m_vsDataSize ); 
	if( m_vsData != nullptr ){
		createVs( m_vsData, m_vsDataSize );
	}

	//compileShader( p_filePath, p_psEntry, "ps_5_0", &m_psData );

	readShader( "../x64/release/regularPs.cso", m_psData, m_psDataSize ); 
	if( m_psData != nullptr ){
		createPs( m_psData, m_psDataSize );
	}
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

void ShaderSet::readShader( const string &p_sourceFilePath, uint8_t*& out_data, int& out_size ){
	ifstream ifs( p_sourceFilePath, ifstream::in | ifstream::binary );
	if( ifs.good() ){
		ifs.seekg( 0, ios::end );  
		out_size = ifs.tellg();  
		out_data = new uint8_t[out_size];
		ifs.seekg(0, ios::beg);  
		ifs.read( (char*)&out_data[0], out_size);
	} else {
		string msg = string("Could not read shader: ") + p_sourceFilePath;
		Utils::error(__FILE__, __FUNCTION__, __LINE__, msg);
	}
	ifs.close(); 
}

//ID3DBlob* ShaderSet::readShader( const string &p_sourceFilePath ){
//	HRESULT hr = S_OK;
//
//	ID3DBlob*	shaderBlob = nullptr;
//	wstring pathAsW = L"";
//	Utils::wstringFromString( pathAsW, p_sourceFilePath );
//	hr = D3DReadFileToBlob( pathAsW.c_str(), &shaderBlob );
//	if( FAILED(hr) ) {
//		string info = string("Could not read blob: ") + p_sourceFilePath;
//		Utils::error(__FILE__, __FUNCTION__, __LINE__, info );
//	}
//	return shaderBlob; 
//}

//void ShaderSet::compileShader( 
//	const string &p_sourceFile, const string &p_entryPoint,
//	const string &p_profile, ID3DBlob** out_blob )
//{
//	ID3DBlob*	compilationErrors = nullptr;
//	ID3DBlob*	shaderBlob = nullptr;
//
//	*out_blob = nullptr;
//
//	DWORD compileFlags = D3DCOMPILE_ENABLE_STRICTNESS;
//
//#if defined(DEBUG) || defined(_DEBUG)
//	compileFlags |= D3DCOMPILE_DEBUG;
//	compileFlags |= D3DCOMPILE_SKIP_OPTIMIZATION;
//	//compileFlags |= D3DCOMPILE_WARNINGS_ARE_ERRORS;
//#else
//	compileFlags |= D3DCOMPILE_OPTIMIZATION_LEVEL3;
//#endif
//
//	wstring wsSource = L"";
//	Utils::wstringFromString( wsSource, p_sourceFile );
//
//	HRESULT res = S_OK;
//	D3DX11CompileFromFile(
//		wsSource.c_str(),
//		nullptr,
//		nullptr,
//		p_entryPoint.c_str(), p_profile.c_str(),
//		compileFlags, 0,
//		nullptr,
//		&shaderBlob, &compilationErrors,
//		&res);
//
//	HRESULT hrCopy = res;
//	if(FAILED(res))
//	{
//		if( compilationErrors )
//		{
//			MessageBoxA(0, (char*)compilationErrors->GetBufferPointer(), 0, 0);
//			SAFE_RELEASE(compilationErrors);
//		}
//		else
//		{
//			HR(hrCopy);
//		}
//	}
//
//	*out_blob = shaderBlob;
//}