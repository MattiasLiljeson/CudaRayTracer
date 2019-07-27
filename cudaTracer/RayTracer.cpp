#include "RayTracer.h"

inline RayTracer::RayTracer(D3DCudaTextureSet* textureSet, int width,
                            int height, InputHandler* p_input,
                            Camera* p_camera) {
    m_textureSet = textureSet;
    options.width = width;
    options.height = height;
    options.fov = 90;
    options.backgroundColor = Vec<float, 3>(0.235294f, 0.67451f, 0.843137f);
    options.maxDepth = 5;
    options.bias = 0.00001f;
    options.scale = tan(deg2rad(options.fov * 0.5f));
    options.imageAspectRatio = options.width / (float)options.height;

    input = p_input;
    camera = p_camera;
}

inline void RayTracer::initScene() {
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

    Shape mirrorBall = Shape(Sphere(Vec3f(0.0f, -1.0f, -3.0f), 1.0f));
    mirrorBall.material.materialType = Object::REFLECTION;
    mirrorBall.material.diffuseColor = Vec3f(0.6f, 0.7f, 0.8f);
    shapes.add(mirrorBall);

    Shape glassBall = Shape(Sphere(Vec3f(0.5f, -1.5f, 0.5f), 1.5f));
    glassBall.material.ior = 1.5f;
    glassBall.material.materialType = Object::REFLECTION_AND_REFRACTION;
    glassBall.material.diffuseColor = Vec3f(0.8f, 0.7f, 0.6f);
    shapes.add(glassBall);

    {
        // TODO: fix memory leaks! Delete these pointers!
        GlobalCudaVector<Vertex>* vertices = new GlobalCudaVector<Vertex>(
            Vertex(Vec3f(-25.0f, -10.0f, 25.0f), Vec2f(0.0f, 0.0f)),  //
            Vertex(Vec3f(25.0f, -10.0f, 25.0f), Vec2f(1.0f, 0.0f)),   //
            Vertex(Vec3f(25.0f, -10.0f, -25.0f), Vec2f(1.0f, 1.0f)),  //
            Vertex(Vec3f(-25.0f, -10.0f, -25.0f), Vec2f(1.0f, 1.0f)));
        GlobalCudaVector<int>* indices =
            new GlobalCudaVector<int>(0, 1, 3, 1, 2, 3);
        Shape mesh =
            Shape(Mesh(indices->getDevMem(), 2, vertices->getDevMem()));
        mesh.material.materialType = Object::DIFFUSE_AND_GLOSSY;
        shapes.add(mesh);
    }

    {
        ObjFileReader objReader;
        Model barrelModel = objReader.readFile(
            "../assets/models/plasticBarrel/", "plastic_barrel.obj", false)[0];
        // std::vector<Model> barrelModels = objReader.readFile(
        //    "../assets/models/box/", "box.obj", false);
        GlobalCudaVector<Vertex>* vertices =
            GlobalCudaVector<Vertex>::fromVector(barrelModel.getVertices());
        GlobalCudaVector<int>* indices =
            GlobalCudaVector<int>::fromVector(barrelModel.getIndices());
        // TODO: 3 is just a happy guess
        int triCnt = barrelModel.getNumVertices() / 3;
        Shape mesh =
            Shape(Mesh(indices->getDevMem(), triCnt, vertices->getDevMem()));
        mesh.material.materialType = Object::DIFFUSE_AND_GLOSSY;
        shapes.add(mesh);
    }
    shapes.copyToDevice();
}

inline void RayTracer::update(float p_dt) {
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

    // kick off the kernel and send the staging buffer
    // cudaLinearMemory as an argument to allow the kernel to
    // write to it

    Scene scene;
    scene.lights = lights.getDevMem();
    scene.lightCnt = lights.size();
    scene.shapes = shapes.getDevMem();
    scene.shapeCnt = shapes.size();
    scene.orig = camera->getPosition();
    scene.camera = camera->getCamera().inversed();

    cudamain(options, scene, m_textureSet->cudaLinearMemory,
             m_textureSet->pitch);
    getLastCudaError("cuda_texture_2d failed");
}
