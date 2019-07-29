#pragma once

#include <cuda_d3d11_interop.h>
#include <cuda_runtime_api.h>
#include "D3DCudaTextureSet.h"
#include "DeviceHandler.h"
#include "ShaderSet.h"

#include "Camera.h"
#include "InputHandler.h"
#include "Service.h"

// Forward declarations
class DeviceHandler;

class TextureRenderer : public Service {
   private:
    DeviceHandler* m_deviceHandler;

    ID3D11InputLayout* m_inputLayout;
    D3DCudaTextureSet m_textureSet;
    ShaderSet* m_shaderSet;
    ID3D11Buffer* m_vertexBuffer;
    int m_texWidth;
    int m_texHeight;

    // Rasterizer states
    ID3D11RasterizerState* m_rsDefault;    // The default rasterizer state
    ID3D11RasterizerState* m_rsWireframe;  // Debug rasterizer


   public:
    TextureRenderer(DeviceHandler* p_deviceHandler, int p_texWidth,
                    int p_texHeight);
    ~TextureRenderer();

    void update(float p_dt);
    void copyToHostArray(float* out_dest);
    D3DCudaTextureSet* getTextureSet() { return &m_textureSet; }

   private:
    void initTexture();
    void initShaders();
    void initInputLayout();
    void initQuad();
    void initStates();
    void initInterop();
    void draw();
    void termInterop();
};
