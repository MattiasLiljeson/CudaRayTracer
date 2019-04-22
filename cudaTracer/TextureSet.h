#pragma once

#include <d3d11.h>

// Data structure for 2D texture shared between DX10 and CUDA
struct TextureSet
{
	ID3D11Texture2D         *pTexture;
	ID3D11ShaderResourceView *pSRView;
	cudaGraphicsResource    *cudaResource;
	void                    *cudaLinearMemory;
	size_t                  pitch;
	int                     width;
	int                     height;
#ifndef USEEFFECT
	int                     offsetInShader;
#endif
};