#pragma once

#include "Camera.h"
#include "D3DCudaTextureSet.h"
#include "GlobalCudaArray.h"
#include "GlobalCudaVector.h"
#include "InputHandler.h"
#include "Light.cuh"
#include "ObjFileReader.h"
#include "Options.cuh"
#include "Scene.cuh"
#include "Sphere.cuh"
#include "tracer.cuh"

class RayTracer {
    Camera* camera;
    InputHandler* input;

    // cuda
    Options options;
    GlobalCudaVector<Light> lights;
    GlobalCudaVector<Shape> shapes;
    D3DCudaTextureSet* m_textureSet;

   public:
    RayTracer(D3DCudaTextureSet* textureSet, int width, int height,
              InputHandler* p_input, Camera* p_camera);

    void initScene();

    void update(float p_dt);
};
