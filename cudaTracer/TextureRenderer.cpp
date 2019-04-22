#include "TextureRenderer.h"
#include "Vertex.h"
#include "LayoutFactory.h"

// Kernels
#include "sinGrid.cuh"
#include "gradient.cuh"
#include "diamondSquare.cuh"

// cuda stuph
#include <cuda_runtime.h>
#include "curand_kernel.h"

//=========================================================================
// Public functions
//=========================================================================
TextureRenderer::TextureRenderer( DeviceHandler* p_deviceHandler, int p_texWidth, int p_texHeight )
{
	m_deviceHandler = p_deviceHandler;
	m_texWidth = p_texWidth;
	m_texHeight = p_texHeight;
	m_shaderSet = nullptr;
	m_inputLayout = nullptr,

	initTexture();
	initShaders();
	initInputLayout();
	initQuad();
	initStates();
	initInterop();
}


TextureRenderer::~TextureRenderer()
{
	termInterop();

	delete m_shaderSet;
	m_shaderSet = nullptr;
	
	SAFE_RELEASE( m_textureSet.pTexture );
	SAFE_RELEASE( m_textureSet.pSRView );

	SAFE_RELEASE( m_inputLayout );
	SAFE_RELEASE( m_vertexBuffer );
	SAFE_RELEASE( m_rsDefault );
	SAFE_RELEASE( m_rsWireframe );
}

void TextureRenderer::update( float p_dt )
{
	cudaStream_t    stream = 0;
	const int nbResources = 1;
	cudaGraphicsResource *ppResources[nbResources] =
	{ m_textureSet.cudaResource, };
	cudaGraphicsMapResources(nbResources, ppResources, stream);
	getLastCudaError( "cudaGraphicsMapResources(3) failed" );

	static float t = 0.0f;
	// populate the 2d texture
	{
		// kick off the kernel and send the staging buffer
		// cudaLinearMemory as an argument to allow the kernel to
		// write to it
		cu_diamondSquare( m_textureSet.cudaLinearMemory,
			m_textureSet.width, m_textureSet.height, m_textureSet.pitch,
			m_curandStates );
		getLastCudaError("cuda_texture_2d failed");

		cudaArray *cuArray;
		cudaGraphicsSubResourceGetMappedArray(&cuArray, m_textureSet.cudaResource, 0, 0);

		// then we want to copy cudaLinearMemory to the D3D texture,
		// via its mapped form : cudaArray
		cudaMemcpy2DToArray(
			cuArray, // dst array
			0, 0,    // offset
			m_textureSet.cudaLinearMemory, m_textureSet.pitch,       // src
			m_textureSet.width*4*sizeof(float), m_textureSet.height, // extent
			cudaMemcpyDeviceToDevice); // kind
		getLastCudaError("cudaMemcpy2DToArray failed");
	}

	cudaGraphicsUnmapResources(nbResources, ppResources, stream);
	getLastCudaError("cudaGraphicsUnmapResources(3) failed");
}

void TextureRenderer::draw()
{
	m_deviceHandler->getContext()->VSSetShader( m_shaderSet->m_vs, nullptr, 0 );
	m_deviceHandler->getContext()->PSSetShader( m_shaderSet->m_ps, nullptr, 0 );
	m_deviceHandler->getContext()->IASetInputLayout( m_inputLayout );
	m_deviceHandler->getContext()->IASetPrimitiveTopology(
		D3D11_PRIMITIVE_TOPOLOGY_TRIANGLESTRIP );

	const int SLOT = 0;
	const int BUFFER_CNT = 1;
	const unsigned int STRIDE[] = {sizeof(Vertex)};
	const unsigned int OFFSET[] = {0};
	m_deviceHandler->getContext()->IASetVertexBuffers(
		SLOT, BUFFER_CNT, &m_vertexBuffer, STRIDE, OFFSET );

	const int VERTEX_CNT = 6;
	const int START_VERTEX = 0;
	m_deviceHandler->getContext()->Draw( VERTEX_CNT, START_VERTEX );
}

void TextureRenderer::copyToHostArray( float* out_dest )
{

	// via its mapped form : cudaArray
	cudaMemcpy2D(
		out_dest, m_textureSet.width*4*sizeof(float), // dst and dst pitch
		m_textureSet.cudaLinearMemory, m_textureSet.pitch,       // src
		m_textureSet.width*4*sizeof(float), m_textureSet.height, // extent
		cudaMemcpyDeviceToHost); // kind
	gpuErrchk( cudaPeekAtLastError() );
	getLastCudaError("cudaMemcpy2D failed");
}

//=========================================================================
// Private functions
//=========================================================================
void TextureRenderer::initTexture()
{
	ID3D11Device* device = m_deviceHandler->getDevice();
	ID3D11DeviceContext* context = m_deviceHandler->getContext();

	m_textureSet.width  = m_texWidth;
	m_textureSet.height = m_texHeight;

	D3D11_TEXTURE2D_DESC desc;
	ZeroMemory(&desc, sizeof(D3D11_TEXTURE2D_DESC));
	desc.Width = m_textureSet.width;
	desc.Height = m_textureSet.height;
	desc.MipLevels = 1;
	desc.ArraySize = 1;
	desc.Format = DXGI_FORMAT_R32G32B32A32_FLOAT;
	desc.SampleDesc.Count = 1;
	desc.Usage = D3D11_USAGE_DEFAULT;
	desc.BindFlags = D3D11_BIND_SHADER_RESOURCE;

	HR(device->CreateTexture2D(&desc, nullptr, &m_textureSet.pTexture));
	SET_D3D_OBJECT_NAME( m_textureSet.pTexture, "theTexture" )

	HR(device->CreateShaderResourceView(m_textureSet.pTexture, nullptr, &m_textureSet.pSRView));
	SET_D3D_OBJECT_NAME( m_textureSet.pSRView, "theTextureSRV" )


	m_textureSet.offsetInShader = 0; // to be clean we should look for the offset from the shader code
	context->PSSetShaderResources(m_textureSet.offsetInShader, 1, &m_textureSet.pSRView);
}

void TextureRenderer::initShaders()
{
	m_shaderSet = new ShaderSet( m_deviceHandler );
	m_shaderSet->createSet( "../shader.hlsl", "VS", "PS" );
}

void TextureRenderer::initInputLayout()
{
	LayoutDesc desc = LayoutFactory::getPointTexCoordDesc();
	HR( m_deviceHandler->getDevice()->CreateInputLayout(
		desc.m_layoutPtr,
		desc.m_elementCnt,
		m_shaderSet->m_vsData,
		m_shaderSet->m_vsDataSize,
		&m_inputLayout ));
	SET_D3D_OBJECT_NAME( m_inputLayout, "inputLayout" )
}

void TextureRenderer::initQuad()
{
	Vertex mesh[]= {
		{{ 1,	-1,	0},	{ 1, 1}},
		{{ -1,	-1,	0},	{ 0, 1}},
		{{ 1,	1,	0},	{ 1, 0}},

		{{ -1, -1,	0},	{ 0, 1}},
		{{ 1,	1,	0},	{ 1, 0}},
		{{ -1,	1,	0},	{ 0, 0}}
	};

	D3D11_BUFFER_DESC bd;
	bd.ByteWidth = sizeof(mesh);
	bd.Usage = D3D11_USAGE_IMMUTABLE;
	bd.BindFlags = D3D11_BIND_VERTEX_BUFFER;
	bd.CPUAccessFlags = 0;
	bd.MiscFlags = 0;

	D3D11_SUBRESOURCE_DATA vertBuffSubRes;
	vertBuffSubRes.pSysMem = &mesh[0];

	HR(m_deviceHandler->getDevice()->CreateBuffer(&bd, &vertBuffSubRes, &m_vertexBuffer));
	SET_D3D_OBJECT_NAME( m_vertexBuffer, "vertexBuffer" )

	//Buffer<PTVertex>* quadBuffer;

	//// Create description for buffer
	//BufferConfig::BUFFER_INIT_DESC bufferDesc;
	//bufferDesc.ElementSize = sizeof(PTVertex);
	//bufferDesc.Usage = BufferConfig::BUFFER_DEFAULT;
	//bufferDesc.NumElements = 6;
	//bufferDesc.Type = BufferConfig::VERTEX_BUFFER;
	//bufferDesc.Slot = BufferConfig::SLOT0;

	//// Create buffer from config and data
	//quadBuffer = new Buffer<PTVertex>(m_device,m_deviceContext,&mesh[0],bufferDesc);

	//return quadBuffer;
}

void TextureRenderer::initStates()
{
	D3D11_RASTERIZER_DESC rasterizerDesc;
	rasterizerDesc.FillMode = D3D11_FILL_SOLID;
	rasterizerDesc.CullMode = D3D11_CULL_NONE;
	rasterizerDesc.FrontCounterClockwise = FALSE;
	rasterizerDesc.DepthClipEnable = FALSE;
	rasterizerDesc.ScissorEnable = FALSE;
	rasterizerDesc.AntialiasedLineEnable = FALSE;
	rasterizerDesc.MultisampleEnable = FALSE;
	rasterizerDesc.DepthBias = 0;
	rasterizerDesc.DepthBiasClamp = 0.0f;
	rasterizerDesc.SlopeScaledDepthBias = 0.0f;
	m_deviceHandler->getDevice()->CreateRasterizerState(&rasterizerDesc, &m_rsDefault);
	SET_D3D_OBJECT_NAME( m_rsDefault, "rasterizerStateDefault" )

	// set the changed values for wireframe mode
	rasterizerDesc.FillMode = D3D11_FILL_WIREFRAME;
	rasterizerDesc.CullMode = D3D11_CULL_NONE;
	rasterizerDesc.AntialiasedLineEnable = TRUE;
	m_deviceHandler->getDevice()->CreateRasterizerState(&rasterizerDesc, &m_rsWireframe);
	SET_D3D_OBJECT_NAME( m_rsWireframe, "rasterizerStateWireFrame" )

	if( false ){
		m_deviceHandler->getContext()->RSSetState(m_rsWireframe);
	} else {
		m_deviceHandler->getContext()->RSSetState(m_rsDefault);
	}
}

void TextureRenderer::initInterop()
{
	// 2D
	// register the Direct3D resources that we'll use
	// we'll read to and write from m_textureSet, so don't set any special
	// map flags for it
	gpuErrchk( cudaGraphicsD3D11RegisterResource( &m_textureSet.cudaResource,
		m_textureSet.pTexture, cudaGraphicsRegisterFlagsNone ) );
	getLastCudaError("cudaGraphicsD3D11RegisterResource (m_textureSet) failed");

	// cuda cannot write into the texture directly : the texture is seen
	// as a cudaArray and can only be mapped as a texture
	// Create a buffer so that cuda can write into it
	// pixel fmt is DXGI_FORMAT_R32G32B32A32_FLOAT
	gpuErrchk( cudaMallocPitch(&m_textureSet.cudaLinearMemory, &m_textureSet.pitch,
		m_textureSet.width * sizeof(float) * 4, m_textureSet.height) );
	getLastCudaError("cudaMallocPitch (m_textureSet) failed");
	gpuErrchk( cudaMemset(m_textureSet.cudaLinearMemory, 1,
		m_textureSet.pitch * m_textureSet.height) );

	// Init curand
	m_curandStates = nullptr;
	m_curandStates = cu_initCurand( m_texWidth, m_texHeight );
}

void TextureRenderer::termInterop()
{
	cu_cleanCurand( m_curandStates );
	gpuErrchk( cudaGraphicsUnregisterResource( m_textureSet.cudaResource ) );
	gpuErrchk( cudaFree( m_textureSet.cudaLinearMemory ) );
}