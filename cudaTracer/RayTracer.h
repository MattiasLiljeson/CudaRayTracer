#pragma once

#include "Camera.h"
#include "D3DCudaTextureSet.h"
#include "GlobalCudaVector.h"
#include "InputHandler.h"
#include "Light.cuh"
#include "ObjFileReader.h"
#include "Options.cuh"
#include "Scene.cuh"
#include "Service.h"
#include "Sphere.cuh"
#include "lodepng.h"

class RayTracer : public Service {
    Camera camera;

    // cuda
    Options options;
    GlobalCudaVector<Light> lights;
    GlobalCudaVector<Shape> shapes;
    GlobalCudaVector<LinearNode> nodes;
    D3DCudaTextureSet* m_textureSet;
    unsigned char* curandStates;

   public:
    RayTracer(D3DCudaTextureSet* textureSet, int width, int height,
              Options options);

    void addDebugGuiStuff();

    void initScene();
    void addLights();
    void addSpheres();
    void addPlane();
    void addMeshes();
    void update(float p_dt);
    void perFrameDebugGuiStuff();
    void handleInput(float p_dt);
    void updateLights(float p_dt);
};
