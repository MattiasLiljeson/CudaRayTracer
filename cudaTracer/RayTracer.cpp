#include "RayTracer.h"
#include "CudaMesh.h"
#include "main.cuh"

RayTracer::RayTracer(D3DCudaTextureSet* textureSet, int width, int height,
                     Options options) {
    m_textureSet = textureSet;
    this->options = options;
    addDebugGuiStuff();
    initScene();
}

void RayTracer::addDebugGuiStuff() {
    DebugGUI* dg = ServiceRegistry::instance().get<DebugGUI>();

    dg->setSize("Rays", 200, 1400);
    dg->setPosition("Rays", 220, 0);
    dg->setVisible("Rays", false);

    dg->setSize("Options", 150, 250);
    dg->setPosition("Options", 0, 75);

    dg->addVar("Options", DebugGUI::DG_FLOAT, DebugGUI::READ_WRITE, "fov",
               &options.host.fov);
    dg->addVar("Options", DebugGUI::DG_INT, DebugGUI::READ_WRITE, "blockdim",
               &options.device.blockSize);
    dg->addVar("Options", DebugGUI::DG_CHAR, DebugGUI::READ_WRITE, "maxdepth",
               &options.device.maxDepth);
    dg->addVar("Options", DebugGUI::DG_CHAR, DebugGUI::READ_WRITE, "samples",
               &options.device.samples);
    dg->addVar("Options", DebugGUI::DG_FLOAT, DebugGUI::READ_WRITE,
               "shadowBias", &options.device.shadowBias);
    dg->addVar("Options", DebugGUI::DG_BOOL, DebugGUI::READ_WRITE,
               "Gamma correction", &options.device.gammaCorrection);
    dg->addVar("Options", DebugGUI::DG_BOOL, DebugGUI::READ_WRITE,
               "Use mesh BVHs", &options.device.useMeshBvhs);
    dg->addVar("Options", DebugGUI::DG_BOOL, DebugGUI::READ_WRITE,
               "Use Scene BVH", &options.device.useSceneBvh);

    dg->addVar("Options", DebugGUI::DG_INT, DebugGUI::READ_ONLY,
               "Height", &options.device.height);
    dg->addVar("Options", DebugGUI::DG_INT, DebugGUI::READ_ONLY,
               "Width", &options.device.width);
    dg->addVar("Options", DebugGUI::DG_FLOAT, DebugGUI::READ_ONLY, "Aspect ratio",
               &options.device.imageAspectRatio);
    dg->addVar("Options", DebugGUI::DG_INT, DebugGUI::READ_ONLY,
               "Lights", &options.host.lightCnt);
}

void RayTracer::initScene() {
    {
        // Init curand
        curandStates = nullptr;
        curandStates =
            cu_initCurand(options.device.width, options.device.height);
    }

    if (!options.host.model.fname.empty()) {
        Material mtl;
        mtl.materialType = Material::DIFFUSE_AND_GLOSSY;
        Mat44f scale = Mat44f::scale(1 / 50.0f, 1 / 50.0f, 1 / 50.0f);
        model::Model model = ObjFileReader().readFile(
            options.host.model.folder, options.host.model.fname, false)[0];;
        static CudaMesh barrel(model, mtl, options.host.model.transform);
        shapes.add(barrel.shape);
    } else {
        addSpheres();
        addPlane();
        addMeshes();
    }
    addLights();
    static BVH::BvhFactory<Shape> bvh(shapes.getHostMemRef());
    for (int i = 0; i < shapes.size(); ++i) {
        shapes[i] = bvh.orderedPrims[i];
    }
    shapes.copyToDevice();
    nodes = GlobalCudaVector<LinearNode>::fromVector(bvh.nodes);
}

float clampedRand(float max) { return (rand() / (float)RAND_MAX) * max; }
float clampedRand(float min, float max) {
    return (rand() / (float)RAND_MAX) * (max - min) + min;
}

void RayTracer::addLights() {
    const float radStep = (2.0f * PI) / options.host.lightCnt;
    const float intensityFactor = 0.5f;
    const float maxDist = 50.0f;

    for (int i = 0; i < options.host.lightCnt; ++i) {
        float power = clampedRand(25.0f);
        Vec3f position(sin(i * radStep) * clampedRand(10, maxDist),
                       clampedRand(25.0f),
                       cos(i * radStep) * clampedRand(10, maxDist));
        Vec3f intensity(clampedRand(intensityFactor),  //
                        clampedRand(intensityFactor),  //
                        clampedRand(intensityFactor));
        lights.add(Light(position, intensity, power));
    }
    lights.copyToDevice();
}

void RayTracer::addSpheres() {
    const int sphereCntSqrt = 10;
    const float dist = 3.0f;
    float scale = 1.0f / sphereCntSqrt;
    for (int i = 0; i < sphereCntSqrt; ++i) {
        for (int j = 0; j < sphereCntSqrt; ++j) {
            float x = -sphereCntSqrt * dist * 0.5f + i * dist;
            float z = 10 + 0.5f + j * dist;
            Shape s = Shape(Sphere(Vec3f(x, -2.0f, z), 1.0f));
            s.material.materialType = Material::DIFFUSE_AND_GLOSSY;
            s.material.diffuseColor = Vec3f(i * scale, 0.1f, j * scale);
            s.material.Kd = 0.8;
            s.material.Ks = scale * i * scale * j * 0.8f + 0.1f;
            s.material.specularExponent = (i * i * i * j * j * j) + 1;
            shapes.add(s);
        }
    }

    Shape mirrorBall = Shape(Sphere(Vec3f(2.5f, 1.0f, 3.0f), 1.0f));
    mirrorBall.material.ior = 64;
    mirrorBall.material.materialType = Material::REFLECTION;
    mirrorBall.material.diffuseColor = Vec3f(0.722f, 0.451f, 0.2f);
    shapes.add(mirrorBall);

    Shape glassBall = Shape(Sphere(Vec3f(1.5f, -1.5f, 6.5f), 1.5f));
    glassBall.material.ior = 1.5f;
    glassBall.material.materialType = Material::REFLECTION_AND_REFRACTION;
    glassBall.material.diffuseColor = Vec3f(0.8f, 0.7f, 0.6f);
    shapes.add(glassBall);
}

void RayTracer::addPlane() {
    float scaleFactor = 25.0f;
    std::vector<Vertex> vertices{
        Vertex(Vec3f(-1.0f, 0.0f, 1.0f), Vec2f(0.0f, 0.0f)),               //
        Vertex(Vec3f(1.0f, 0.0f, 1.0f), Vec2f(scaleFactor, 0.0f)),         //
        Vertex(Vec3f(1.0f, 0.0f, -1.f), Vec2f(scaleFactor, scaleFactor)),  //
        Vertex(Vec3f(-1.0f, 0.0f, -1.f), Vec2f(0.0f, scaleFactor))};
    std::vector<int> indices{0, 1, 3, 1, 2, 3};

    std::string diffuseFname = "../assets/textures/pattern_106/diffus.png";
    std::string normalFname = "../assets/textures/pattern_106/normal.png";
    std::string specularFname = "../assets/textures/pattern_106/specular.png";

    Material mtl;
    mtl.materialType = Material::DIFFUSE_AND_GLOSSY;
    mtl.ior = 4;

    Mat44f scale = Mat44f::scale(100, 1, 100);
    Mat44f translate = Mat44f::translate(0, -3, 0);
    static CudaMesh plane(vertices, indices, 2, diffuseFname, normalFname,
                          specularFname, Texture::WRAP, mtl, scale * translate);
    shapes.add(plane.shape);
}

void RayTracer::addMeshes() {
    Material mtl;
    mtl.materialType = Material::DIFFUSE_AND_GLOSSY;

    Mat44f scale = Mat44f::scale(1 / 50.0f, 1 / 50.0f, 1 / 50.0f);

    model::Model plasticBarrel = ObjFileReader().readFile(
        "../assets/models/plasticBarrel/", "plastic_barrel.obj", false)[0];
    {
        Mat44f translate = Mat44f::translate(0.5, -3, 9);
        static CudaMesh barrel(plasticBarrel, mtl, scale * translate);
        shapes.add(barrel.shape);
    }
    {
        Mat44f translate = Mat44f::translate(-2, -3, 5);
        static CudaMesh barrel(plasticBarrel, mtl, scale * translate);
        shapes.add(barrel.shape);
    }
    {
        Material stainless;
        stainless.materialType = Material::DIFFUSE_AND_GLOSSY;
        stainless.specularExponent = 750;
        model::Model stainlessBarrel = ObjFileReader().readFile(
            "../assets/models/metalBarrel/", "metal_barrel.obj", false)[0];
        Mat44f translate = Mat44f::translate(-4, -3, 4);
        static CudaMesh barrel(stainlessBarrel, mtl, scale * translate);
        shapes.add(barrel.shape);
    }
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
    scene.nodes = nodes.getDevMem();

    cudamain(options.device, scene, m_textureSet->cudaLinearMemory,
             m_textureSet->pitch, options.device.blockSize, curandStates);
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
            float ndc_x =
                (2.0f * (x + 0.5f) / (float)options.device.width - 1.0f) *
                options.device.imageAspectRatio * options.device.scale;
            float ndc_y =
                (1.0f - 2.0f * (y + 0.5f) / (float)options.device.height) *
                options.device.scale;
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
    Mat44f rotY = Mat44f::rotationY(p_dt * 25);
    for (int i = 0; i < lights.size(); ++i) {
        Vec3f tmp = lights[i].position;
        tmp = rotY.multPoint(lights[i].position);
        lights[i].position = tmp;
    }
    lights.copyToDevice();
}