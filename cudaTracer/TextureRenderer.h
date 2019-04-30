#pragma once

#include <cuda_d3d11_interop.h>
#include <cuda_runtime_api.h>
#include "DeviceHandler.h"
#include "ShaderSet.h"
#include "TextureSet.h"

#include "Light.cuh"
#include "Options.cuh"
#include "Sphere.cuh"
#include "GlobalCudaArray.h"



// Forward declarations
class DeviceHandler;

class TextureRenderer {
   private:
    DeviceHandler* m_deviceHandler;

    ID3D11InputLayout* m_inputLayout;
    TextureSet m_textureSet;
    ShaderSet* m_shaderSet;
    ID3D11Buffer* m_vertexBuffer;
    int m_texWidth;
    int m_texHeight;

    // cuda
    GlobalCudaArray<Options, 1> options;
    GlobalCudaArray<Light, 2> lights;
    GlobalCudaArray<Sphere, 2> spheres;

    // Rasterizer states
    ID3D11RasterizerState* m_rsDefault;    // The default rasterizer state
    ID3D11RasterizerState* m_rsWireframe;  // Debug rasterizer

    // unsigned char* m_curandStates;

   public:
    TextureRenderer(DeviceHandler* p_deviceHandler, int p_texWidth,
                    int p_texHeight);
    ~TextureRenderer();

    void update(float p_dt);
    void draw();
    void copyToHostArray(float* out_dest);

   private:
    void initTexture();
    void initShaders();
    void initInputLayout();
    void initQuad();
    void initStates();
    void initInterop();
    void termInterop();
};
