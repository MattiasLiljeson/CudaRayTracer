#include "RayTracer.h"
#include "CudaMesh.h"

RayTracer::RayTracer(D3DCudaTextureSet* textureSet, int width, int height,
                     Options options) {
    m_textureSet = textureSet;
    this->options = options;
    blockDim = 16;
    addDebugGuiStuff();
    initScene();
}

void RayTracer::addDebugGuiStuff() {
    DebugGUI* dg = ServiceRegistry::instance().get<DebugGUI>();

    dg->setSize("Rays", 200, 1400);
    dg->setPosition("Rays", 220, 0);
    dg->setVisible("Rays", false);

    dg->addVar("Options", DebugGUI::DG_FLOAT, DebugGUI::READ_WRITE, "fov",
               &options.fov);
    dg->addVar("Options", DebugGUI::DG_INT, DebugGUI::READ_WRITE, "blockdim",
               &blockDim);
    dg->addVar("Options", DebugGUI::DG_CHAR, DebugGUI::READ_WRITE, "maxdepth",
               &options.maxDepth);
    dg->addVar("Options", DebugGUI::DG_CHAR, DebugGUI::READ_WRITE, "samples",
               &options.samples);
}

void RayTracer::initScene() {
    {
        // Init curand
        curandStates = nullptr;
        curandStates = cu_initCurand(options.width, options.height);
    }

    addLights();
    addSpheres();
    addPlane();
    addMesh();
    shapes.copyToDevice();
}

void RayTracer::addLights() {
    const int numLights = 10;
    const float radStep = (2 * PI) / numLights;
    const float intensityFactor = 0.5f;

    for (int i = 0; i < 10; ++i) {
        Vec3f position(sin(i * radStep) * 5, 5.0f, cos(i * radStep) * 5);
        Vec3f intensity(rand() / (float)RAND_MAX * intensityFactor,  //
                        rand() / (float)RAND_MAX * intensityFactor,  //
                        rand() / (float)RAND_MAX * intensityFactor);
        lights.add(Light(position, intensity));
    }
    lights.copyToDevice();
}

void RayTracer::addSpheres() {
    const int sphereCntSqrt = 5;
    float scale = 1.0f / sphereCntSqrt;
    for (int i = 0; i < sphereCntSqrt; ++i) {
        for (int j = 0; j < sphereCntSqrt; ++j) {
            // int idx = i * sphereCntSqrt + j;
            // std::cerr << idx << std::endl;
            float x = -sphereCntSqrt + i * 2.0f;
            float z = -sphereCntSqrt + j * 2.0f;
            Shape s = Shape(Sphere(Vec3f(x, -4.0f, z), 1.0f));
            s.material.materialType = Object::DIFFUSE_AND_GLOSSY;
            s.material.diffuseColor = Vec3f(x * scale, 0.1f, z * scale);
            shapes.add(s);
        }
    }

    Shape mirrorBall = Shape(Sphere(Vec3f(1.0f, 1.0f, 3.0f), 1.0f));
    mirrorBall.material.materialType = Object::REFLECTION;
    mirrorBall.material.diffuseColor = Vec3f(0.6f, 0.7f, 0.8f);
    shapes.add(mirrorBall);

    Shape glassBall = Shape(Sphere(Vec3f(0.5f, -1.5f, 4.5f), 1.5f));
    glassBall.material.ior = 1.5f;
    glassBall.material.materialType = Object::REFLECTION_AND_REFRACTION;
    glassBall.material.diffuseColor = Vec3f(0.8f, 0.7f, 0.6f);
    shapes.add(glassBall);
}

void RayTracer::addPlane() {
    float planeSize = 100.0f;
    std::vector<Vertex> vertices{
        Vertex(Vec3f(-planeSize, -10.0f, planeSize), Vec2f(0.0f, 0.0f)),  //
        Vertex(Vec3f(planeSize, -10.0f, planeSize), Vec2f(1.0f, 0.0f)),   //
        Vertex(Vec3f(planeSize, -10.0f, -planeSize), Vec2f(1.0f, 1.0f)),  //
        Vertex(Vec3f(-planeSize, -10.0f, -planeSize), Vec2f(1.0f, 1.0f))};
    std::vector<int> indices{0, 1, 3, 1, 2, 3};
    std::vector<unsigned char> texData{255, 0,   0,   0,  //
                                       0,   255, 0,   0,  //
                                       0,   0,   255, 0,  //
                                       255, 0,   255, 0};
    static CudaMesh plane(vertices, indices, 2, texData, 2, 2);
    shapes.add(plane.shape);
}

void RayTracer::addMesh() {
    Model barrelModel = ObjFileReader().readFile(
        "../assets/models/plasticBarrel/", "plastic_barrel.obj", false)[0];
    static CudaMesh barrel(barrelModel);
    shapes.add(barrel.shape);
}

void RayTracer::update(float p_dt) {
    handleInput(p_dt);
    updateLights(p_dt);
    perFrameDebugGuiStuff();
    // kick off the kernel and send the staging buffer
    // cudaLinearMemory as an argument to allow the kernel to
    // write to it

    Scene scene;
    scene.lights = lights.getDevMem();
    scene.lightCnt = lights.size();
    scene.shapes = shapes.getDevMem();
    scene.shapeCnt = shapes.size();
    scene.orig = camera.getPosition();
    scene.camera = camera.getCamera().inversed();

    cudamain(options, scene, m_textureSet->cudaLinearMemory,
             m_textureSet->pitch, blockDim, curandStates);
    getLastCudaError("cuda_texture_2d failed");
}

void RayTracer::perFrameDebugGuiStuff() {
    static bool added = false;  // only add ATB the first time
    DebugGUI* dg = ServiceRegistry::instance().get<DebugGUI>();
    static Vec3f dir[9];
    static char buffer[128];
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            float x = 320.0f * i, y = 240.0f * j;
            float ndc_x = (2.0f * (x + 0.5f) / (float)options.width - 1.0f) *
                          options.imageAspectRatio * options.scale;
            float ndc_y = (1.0f - 2.0f * (y + 0.5f) / (float)options.height) *
                          options.scale;
            if (!added) {
                // TODO: debug
                sprintf_s(buffer, "ray %d %d", i, j);
                DebugGUI* dg = ServiceRegistry::instance().get<DebugGUI>();
                dg->addVar("Rays", DebugGUI::DG_VEC3, DebugGUI::READ_WRITE,
                           string(buffer), &(dir[i * 3 + j]));
            }
            dir[i * 3 + j] = Vec3f(ndc_x, ndc_y, -1.0f).normalized();
            dir[i * 3 + j] = camera.getCamera().multVec(dir[i * 3 + j]);
            dir[i * 3 + j] = dir[i * 3 + j].normalized();
        }
    }
    added = true;
}

void RayTracer::handleInput(float p_dt) {
    InputHandler* input = ServiceRegistry::instance().get<InputHandler>();
    static bool inputActive = false;
    if (input->getKey(InputHandler::TAB)) {
        inputActive = !inputActive;
    }
    if (inputActive) {
        const float speed = 10.0f;
        if (input->getKey(InputHandler::W)) camera.walk(speed * p_dt);
        if (input->getKey(InputHandler::A)) camera.strafe(-speed * p_dt);
        if (input->getKey(InputHandler::S)) camera.walk(-speed * p_dt);
        if (input->getKey(InputHandler::D)) camera.strafe(speed * p_dt);
        if (input->getKey(InputHandler::SPACE)) camera.ascend(speed * p_dt);
        if (input->getKey(InputHandler::LCTRL)) camera.ascend(-speed * p_dt);
        camera.rotateY((float)-input->getMouse(InputHandler::X) * speed * p_dt);
        camera.pitch((float)-input->getMouse(InputHandler::Y) * speed * p_dt);
        camera.update();
    }
}

void RayTracer::updateLights(float p_dt) {
    Mat44f rotY = Mat44f::rotationY(p_dt * 100);
    // update lights
    for (int i = 0; i < lights.size(); ++i) {
        Vec3f tmp = lights[i].position;
        tmp = rotY.multVec(lights[i].position);
        lights[i].position = tmp;
    }
    lights.copyToDevice();
}