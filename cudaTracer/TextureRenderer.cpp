#include "TextureRenderer.h"
#include "LayoutFactory.h"
#include "Vertex.h"

// Kernels
#include "diamondSquare.cuh"
#include "gradient.cuh"
#include "sinGrid.cuh"

// cuda stuph
#include <cuda_runtime.h>

#include "Mesh.cuh"
#include "curand_kernel.h"

#include "Scene.cuh"
#include "tracer.cuh"

//=========================================================================
// Public functions
//=========================================================================
TextureRenderer::TextureRenderer(DeviceHandler* p_deviceHandler, int p_texWidth,
                                 int p_texHeight, InputHandler* p_input,
                                 Camera* p_camera) {
    m_deviceHandler = p_deviceHandler;
    input = p_input;
    camera = p_camera;
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

TextureRenderer::~TextureRenderer() {
    termInterop();

    delete m_shaderSet;
    m_shaderSet = nullptr;

    SAFE_RELEASE(m_textureSet.pTexture);
    SAFE_RELEASE(m_textureSet.pSRView);

    SAFE_RELEASE(m_inputLayout);
    SAFE_RELEASE(m_vertexBuffer);
    SAFE_RELEASE(m_rsDefault);
    SAFE_RELEASE(m_rsWireframe);
}
void TextureRenderer::update(float p_dt) {
    cudaStream_t stream = 0;
    const int nbResources = 1;
    cudaGraphicsResource* ppResources[nbResources] = {
        m_textureSet.cudaResource,
    };
    cudaGraphicsMapResources(nbResources, ppResources, stream);
    getLastCudaError("cudaGraphicsMapResources(3) failed");

    // handle input
    float speed = 50.0f;
    input->update();
    if (input->getKey(InputHandler::W)) camera->walk(speed * p_dt);
    if (input->getKey(InputHandler::A)) camera->strafe(-speed * p_dt);
    if (input->getKey(InputHandler::S)) camera->walk(-speed * p_dt);
    if (input->getKey(InputHandler::D)) camera->strafe(speed * p_dt);
    if (input->getKey(InputHandler::SPACE)) camera->ascend(speed * p_dt);
    if (input->getKey(InputHandler::LCTRL)) camera->ascend(-speed * p_dt);
    camera->rotateY((float)-input->getMouse(InputHandler::X) * 10 * p_dt);
    camera->pitch((float)-input->getMouse(InputHandler::Y) * 10 * p_dt);
    camera->update();

    Mat44f rotY = Mat44f::rotationY(p_dt * 100);
    // update lights
    for (int i = 0; i < lights.size(); ++i) {
        Vec3f tmp = lights[i].position;
        tmp = rotY.multVec(lights[i].position);
        lights[i].position = tmp;
    }
    lights.copyToDevice();

    // debug stuff
    {
        static Vec3f dir[9];
        static bool added = false;  // only add ATB the first time
        static char buffer[128];
        static int mouseY = 0;
        static int mouseX = 0;

        DebugGUI* dg = DebugGUI::getInstance();
        if (!added) {
            dg->addVar("Mouse", DebugGUI::DG_INT, DebugGUI::READ_ONLY, "X",
                       &mouseX);
            dg->addVar("Mouse", DebugGUI::DG_INT, DebugGUI::READ_ONLY, "Y",
                       &mouseY);
        }
        mouseX = input->getMouse(InputHandler::X);
        mouseY = input->getMouse(InputHandler::Y);

        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                float x = 320.0f * i, y = 240.0f * j;
                float ndc_x =
                    (2.0f * (x + 0.5f) / (float)options.width - 1.0f) *
                    options.imageAspectRatio * options.scale;
                float ndc_y =
                    (1.0f - 2.0f * (y + 0.5f) / (float)options.height) *
                    options.scale;
                if (!added) {
                    // TODO: debug
                    sprintf_s(buffer, "ray %d %d", i, j);
                    DebugGUI::getInstance()->addVar(
                        "Rays", DebugGUI::DG_VEC3, DebugGUI::READ_WRITE,
                        string(buffer), &(dir[i * 3 + j]));
                }
                dir[i * 3 + j] = Vec3f(ndc_x, ndc_y, -1.0f).normalized();
                dir[i * 3 + j] = camera->getCamera().multVec(dir[i * 3 + j]);
                dir[i * 3 + j] = dir[i * 3 + j].normalized();
            }
        }
        added = true;
    }

    // populate the 2d texture
    {
        // kick off the kernel and send the staging buffer
        // cudaLinearMemory as an argument to allow the kernel to
        // write to it

        Scene scene;
        scene.lights = lights.getDevMem();
        scene.lightCnt = lights.size();
        scene.spheres = spheres.getDevMem();
        scene.sphereCnt = spheres.size();
        scene.orig = camera->getPosition();
        scene.camera = camera->getCamera().inversed();

        cudamain(options, scene, m_textureSet.cudaLinearMemory,
                 m_textureSet.pitch);
        getLastCudaError("cuda_texture_2d failed");

        cudaArray* cuArray;
        cudaGraphicsSubResourceGetMappedArray(&cuArray,
                                              m_textureSet.cudaResource, 0, 0);

        // then we want to copy cudaLinearMemory to the D3D texture,
        // via its mapped form : cudaArray
        cudaMemcpy2DToArray(cuArray,  // dst array
                            0, 0,     // offset
                            m_textureSet.cudaLinearMemory,
                            m_textureSet.pitch,  // src
                            m_textureSet.width * (int)4 * sizeof(float),
                            m_textureSet.height,        // extent
                            cudaMemcpyDeviceToDevice);  // kind
        getLastCudaError("cudaMemcpy2DToArray failed");
    }

    cudaGraphicsUnmapResources(nbResources, ppResources, stream);
    getLastCudaError("cudaGraphicsUnmapResources(3) failed");
}

void TextureRenderer::draw() {
    m_deviceHandler->getContext()->VSSetShader(m_shaderSet->m_vs, nullptr, 0);
    m_deviceHandler->getContext()->PSSetShader(m_shaderSet->m_ps, nullptr, 0);
    m_deviceHandler->getContext()->IASetInputLayout(m_inputLayout);
    m_deviceHandler->getContext()->IASetPrimitiveTopology(
        D3D11_PRIMITIVE_TOPOLOGY_TRIANGLESTRIP);

    const int SLOT = 0;
    const int BUFFER_CNT = 1;
    const unsigned int STRIDE[] = {sizeof(Vertex)};
    const unsigned int OFFSET[] = {0};
    m_deviceHandler->getContext()->IASetVertexBuffers(
        SLOT, BUFFER_CNT, &m_vertexBuffer, STRIDE, OFFSET);

    const int VERTEX_CNT = 6;
    const int START_VERTEX = 0;
    m_deviceHandler->getContext()->Draw(VERTEX_CNT, START_VERTEX);
}

void TextureRenderer::copyToHostArray(float* out_dest) {
    // via its mapped form : cudaArray
    cudaMemcpy2D(
        out_dest, m_textureSet.width * 4 * sizeof(float),   // dst and dst pitch
        m_textureSet.cudaLinearMemory, m_textureSet.pitch,  // src
        m_textureSet.width * 4 * sizeof(float), m_textureSet.height,  // extent
        cudaMemcpyDeviceToHost);                                      // kind
    gpuErrchk(cudaPeekAtLastError());
    getLastCudaError("cudaMemcpy2D failed");
}

//=========================================================================
// Private functions
//=========================================================================
void TextureRenderer::initTexture() {
    ID3D11Device* device = m_deviceHandler->getDevice();
    ID3D11DeviceContext* context = m_deviceHandler->getContext();

    m_textureSet.width = m_texWidth;
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
    SET_D3D_OBJECT_NAME(m_textureSet.pTexture, "theTexture")

    HR(device->CreateShaderResourceView(m_textureSet.pTexture, nullptr,
                                        &m_textureSet.pSRView));
    SET_D3D_OBJECT_NAME(m_textureSet.pSRView, "theTextureSRV")

    m_textureSet.offsetInShader =
        0;  // to be clean we should look for the offset from the shader code
    context->PSSetShaderResources(m_textureSet.offsetInShader, 1,
                                  &m_textureSet.pSRView);
}

void TextureRenderer::initShaders() {
    m_shaderSet = new ShaderSet(m_deviceHandler);
    m_shaderSet->createSet("../shader.hlsl", "VS", "PS");
}

void TextureRenderer::initInputLayout() {
    LayoutDesc desc = LayoutFactory::getPointTexCoordDesc();
    HR(m_deviceHandler->getDevice()->CreateInputLayout(
        desc.m_layoutPtr, desc.m_elementCnt, m_shaderSet->m_vsData,
        m_shaderSet->m_vsDataSize, &m_inputLayout));
    SET_D3D_OBJECT_NAME(m_inputLayout, "inputLayout")
}

void TextureRenderer::initQuad() {
    Vertex mesh[] = {
        {{1, -1, 0}, {1, 1}},  {{-1, -1, 0}, {0, 1}}, {{1, 1, 0}, {1, 0}},

        {{-1, -1, 0}, {0, 1}}, {{1, 1, 0}, {1, 0}},   {{-1, 1, 0}, {0, 0}}};

    D3D11_BUFFER_DESC bd;
    bd.ByteWidth = sizeof(mesh);
    bd.Usage = D3D11_USAGE_IMMUTABLE;
    bd.BindFlags = D3D11_BIND_VERTEX_BUFFER;
    bd.CPUAccessFlags = 0;
    bd.MiscFlags = 0;

    D3D11_SUBRESOURCE_DATA vertBuffSubRes;
    vertBuffSubRes.pSysMem = &mesh[0];

    HR(m_deviceHandler->getDevice()->CreateBuffer(&bd, &vertBuffSubRes,
                                                  &m_vertexBuffer));
    SET_D3D_OBJECT_NAME(m_vertexBuffer, "vertexBuffer")

    // Buffer<PTVertex>* quadBuffer;

    //// Create description for buffer
    // BufferConfig::BUFFER_INIT_DESC bufferDesc;
    // bufferDesc.ElementSize = sizeof(PTVertex);
    // bufferDesc.Usage = BufferConfig::BUFFER_DEFAULT;
    // bufferDesc.NumElements = 6;
    // bufferDesc.Type = BufferConfig::VERTEX_BUFFER;
    // bufferDesc.Slot = BufferConfig::SLOT0;

    //// Create buffer from config and data
    // quadBuffer = new
    // Buffer<PTVertex>(m_device,m_deviceContext,&mesh[0],bufferDesc);

    // return quadBuffer;
}

void TextureRenderer::initStates() {
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
    m_deviceHandler->getDevice()->CreateRasterizerState(&rasterizerDesc,
                                                        &m_rsDefault);
    SET_D3D_OBJECT_NAME(m_rsDefault, "rasterizerStateDefault")

    // set the changed values for wireframe mode
    rasterizerDesc.FillMode = D3D11_FILL_WIREFRAME;
    rasterizerDesc.CullMode = D3D11_CULL_NONE;
    rasterizerDesc.AntialiasedLineEnable = TRUE;
    m_deviceHandler->getDevice()->CreateRasterizerState(&rasterizerDesc,
                                                        &m_rsWireframe);
    SET_D3D_OBJECT_NAME(m_rsWireframe, "rasterizerStateWireFrame")

    if (false) {
        m_deviceHandler->getContext()->RSSetState(m_rsWireframe);
    } else {
        m_deviceHandler->getContext()->RSSetState(m_rsDefault);
    }
}

void TextureRenderer::initInterop() {
    // 2D
    // register the Direct3D resources that we'll use
    // we'll read to and write from m_textureSet, so don't set any special
    // map flags for it
    gpuErrchk(cudaGraphicsD3D11RegisterResource(&m_textureSet.cudaResource,
                                                m_textureSet.pTexture,
                                                cudaGraphicsRegisterFlagsNone));
    getLastCudaError("cudaGraphicsD3D11RegisterResource (m_textureSet) failed");

    // cuda cannot write into the texture directly : the texture is seen
    // as a cudaArray and can only be mapped as a texture
    // Create a buffer so that cuda can write into it
    // pixel fmt is DXGI_FORMAT_R32G32B32A32_FLOAT
    gpuErrchk(cudaMallocPitch(
        &m_textureSet.cudaLinearMemory, &m_textureSet.pitch,
        m_textureSet.width * sizeof(float) * 4, m_textureSet.height));
    getLastCudaError("cudaMallocPitch (m_textureSet) failed");
    gpuErrchk(cudaMemset(m_textureSet.cudaLinearMemory, 1,
                         m_textureSet.pitch * m_textureSet.height));

    // Init curand
    // m_curandStates = nullptr;
    // m_curandStates = cu_initCurand( m_texWidth, m_texHeight );

    size_t stacksize = 0;
    gpuErrchk(cudaDeviceGetLimit(&stacksize, cudaLimitStackSize));
    gpuErrchk(cudaDeviceSetLimit(cudaLimitStackSize, 8096));

    options.width = m_textureSet.width;
    options.height = m_textureSet.height;
    options.fov = 90;
    options.backgroundColor = Vec<float, 3>(0.235294f, 0.67451f, 0.843137f);
    options.maxDepth = 5;
    options.bias = 0.00001f;
    options.scale = tan(deg2rad(options.fov * 0.5f));
    options.imageAspectRatio = options.width / (float)options.height;

    const int numLights = 10;
    const float radStep = (2 * PI) / numLights;

    for (int i = 0; i < 10; ++i) {
        Vec3f position(sin(i * radStep) * 5, 0.0f, cos(i * radStep) * 5);
        Vec3f intensity(rand() / (float)RAND_MAX, rand() / (float)RAND_MAX,
                        rand() / (float)RAND_MAX);
        lights.add(Light(position, intensity));
    }
    lights.copyToDevice();

    const int sphereCntSqrt = 5;
    for (int i = 0; i < sphereCntSqrt; ++i) {
        for (int j = 0; j < sphereCntSqrt; ++j) {
            int idx = i * sphereCntSqrt + j;
            // std::cerr << idx << std::endl;
            float scale = 1.0f / sphereCntSqrt;
            float x = -sphereCntSqrt + i * 2.0f;
            float z = -sphereCntSqrt + j * 2.0f;
            Sphere s = Sphere::sphere(Vec3f(x, -4.0f, z), 1.0f);
            s.object.materialType = Object::DIFFUSE_AND_GLOSSY;
            s.object.diffuseColor = Vec3f(0.1f, 0.1f, 0.1f);
            spheres.add(s);
        }
    }

    Sphere mirrorBall = Sphere::sphere(Vec<float, 3>(0.0f, -1.0f, -3.0f), 1.0f);
    mirrorBall.object.materialType = Object::REFLECTION;
    mirrorBall.object.diffuseColor = Vec3f(0.6f, 0.7f, 0.8f);
    spheres.add(mirrorBall);

    Sphere glassBall = Sphere::sphere(Vec<float, 3>(0.5f, -1.5f, 0.5f), 1.5f);
    glassBall.object.ior = 1.5f;
    glassBall.object.materialType = Object::REFLECTION_AND_REFRACTION;
    glassBall.object.diffuseColor = Vec3f(0.8f, 0.7f, 0.6f);
    spheres.add(glassBall);

    spheres.copyToDevice();

    // TODO: fix memory leaks! Delete these pointers!
    GlobalCudaVector<Vec3f>* vertices =
        new GlobalCudaVector<Vec3f>(Vec3f(-5.0f, -3.0f, -6.0f),  //
                                    Vec3f(5.0f, -3.0f, -6.0f),   //
                                    Vec3f(5.0f, -3.0f, -16.0f),  //
                                    Vec3f(-5.0f, -3.0f, -16.0f));
    GlobalCudaVector<int>* indices =
        new GlobalCudaVector<int>(0, 1, 3, 1, 2, 3);
    GlobalCudaVector<Vec2f>* sts =
        new GlobalCudaVector<Vec2f>(Vec2f(0.0f, 0.0f), Vec2f(1.0f, 0.0f),
                                    Vec2f(1.0f, 1.0f), Vec2f(1.0f, 1.0f));
    Mesh* mesh = new Mesh(vertices->getDevMem(), indices->getDevMem(), 2,
                          sts->getDevMem());
    mesh->materialType = Object::DIFFUSE_AND_GLOSSY;
}

void TextureRenderer::termInterop() {
    // cu_cleanCurand( m_curandStates );
    gpuErrchk(cudaGraphicsUnregisterResource(m_textureSet.cudaResource));
    gpuErrchk(cudaFree(m_textureSet.cudaLinearMemory));
}
