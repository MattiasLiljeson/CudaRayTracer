#pragma once

#include "Camera.h"
#include "D3DCudaTextureSet.h"
#include "GlobalCudaVector.h"
#include "InputHandler.h"
#include "Light.cuh"

#include "Options.cuh"
#include "Scene.cuh"
#include "Sphere.cuh"

#include "ObjFileReader.h"
#include "lodepng.h"
#include "tracer.cuh"

#include "Service.h"

class CudaMesh {
   public:
    GlobalCudaVector<Vertex> vertices;
    GlobalCudaVector<int> indices;
    GlobalCudaVector<unsigned char> textureData;
    Texture texture;
    Mesh mesh;
    Shape shape;

    CudaMesh(Model model) {
        vertices = GlobalCudaVector<Vertex>::fromVector(model.getVertices());
        indices = GlobalCudaVector<int>::fromVector(model.getIndices());
        // TODO: 3 is just a happy guess
        int triCnt = model.getNumVertices() / 3;

        std::string texFname = model.getMaterials()[0].texturePath;
        std::vector<unsigned char> image;  // the raw pixels
        unsigned width, height;
        unsigned error = lodepng::decode(image, width, height, texFname);
        // if there's an error, display it. TODO: Replace this with something
        // proper..
        if (error) {
            std::cout << "decoder error " << error << ": "
                      << lodepng_error_text(error) << std::endl;
        }
        textureData = GlobalCudaVector<unsigned char>::fromVector(image);
        texture = Texture(width, height, textureData.getDevMem());
        mesh = Mesh(indices.getDevMem(), triCnt, vertices.getDevMem(), texture);
        shape = Shape(mesh);
        shape.material.materialType = Object::DIFFUSE_AND_GLOSSY;
    }

    CudaMesh(std::vector<Vertex> vertices, std::vector<int> indices,
             int p_triCnt, vector<unsigned char> textureData, int texWidth,
             int texHeight) {
        this->vertices = GlobalCudaVector<Vertex>::fromVector(vertices);
        this->indices = GlobalCudaVector<int>::fromVector(indices);
        this->textureData =
            GlobalCudaVector<unsigned char>::fromVector(textureData);
        texture = Texture(texWidth, texHeight, this->textureData.getDevMem());
        this->mesh = Mesh(this->indices.getDevMem(), p_triCnt,
                          this->vertices.getDevMem(), texture);
        shape = Shape(mesh);
        shape.material.materialType = Object::DIFFUSE_AND_GLOSSY;
    }
};

class RayTracer : public Service {
    Camera camera;

    // cuda
    Options options;
    GlobalCudaVector<Light> lights;
    GlobalCudaVector<Shape> shapes;
    D3DCudaTextureSet* m_textureSet;
    int blockDim;

   public:
    RayTracer(D3DCudaTextureSet* textureSet, int width, int height,
              Options options);

    void addDebugGuiStuff();

    void initScene();
    void addLights();
    void addSpheres();
    void addPlane();
    void addMesh();
    void update(float p_dt);
    void perFrameDebugGuiStuff();
    void handleInput(float p_dt);
    void updateLights(float p_dt);
};
