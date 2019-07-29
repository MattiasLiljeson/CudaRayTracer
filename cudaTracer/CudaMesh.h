#pragma once
#include <lodepng.h>
#include <vector>
#include "GlobalCudaVector.h"
#include "Mesh.cuh"
#include "Model.h"
#include "Shape.cuh"
#include "Texture.cuh"
#include "Vertex.cuh"

class CudaMesh {
   public:
    GlobalCudaVector<Vertex> vertices;
    GlobalCudaVector<int> indices;
    GlobalCudaVector<unsigned char> diffuseData;
    GlobalCudaVector<unsigned char> normalsData;
    Mesh mesh;
    Shape shape;

    CudaMesh(const Model &model) {
        vertices = GlobalCudaVector<Vertex>::fromVector(model.getVertices());
        indices = GlobalCudaVector<int>::fromVector(model.getIndices());
        generateTangents();
        // TODO: 3 is just a happy guess
        int triCnt = model.getNumVertices() / 3;

        std::string diffuseFname = model.getMaterials()[0].texturePath;
        Texture diffuseTex = loadPngFromDisk(diffuseFname, &diffuseData);
        std::string normalsFname = model.getMaterials()[0].normalMapPath;
        Texture normalsTex = loadPngFromDisk(normalsFname, &normalsData);

        mesh = Mesh(indices.getDevMem(), triCnt, vertices.getDevMem(),
                    diffuseTex, normalsTex);
        shape = Shape(mesh);
        shape.material.materialType = Object::DIFFUSE_AND_GLOSSY;
    }

    static Texture CudaMesh::loadPngFromDisk(
        std::string texFname, GlobalCudaVector<unsigned char>* data) {
        std::vector<unsigned char> image;  // the raw pixels
        unsigned width, height;
        unsigned error = lodepng::decode(image, width, height, texFname);
        // if there's an error, display it. TODO: Replace this with something
        // proper..
        if (error) {
            std::cout << "decoder error " << error << ": "
                      << lodepng_error_text(error) << std::endl;
        }
        *data = GlobalCudaVector<unsigned char>::fromVector(image);
        return Texture(width, height, data->getDevMem());
    }

    CudaMesh(std::vector<Vertex> vertices, std::vector<int> indices,
             int p_triCnt, std::vector<unsigned char> textureData, int texWidth,
             int texHeight) {
        this->vertices = GlobalCudaVector<Vertex>::fromVector(vertices);
        this->indices = GlobalCudaVector<int>::fromVector(indices);
        GlobalCudaVector<unsigned char> diffuseData =
            GlobalCudaVector<unsigned char>::fromVector(textureData);
        Texture diffuseTex =
            Texture(texWidth, texHeight, diffuseData.getDevMem());
        this->mesh = Mesh(this->indices.getDevMem(), p_triCnt,
                          this->vertices.getDevMem(), diffuseTex, diffuseTex);
        shape = Shape(mesh);
        shape.material.materialType = Object::DIFFUSE_AND_GLOSSY;
    }

    void CudaMesh::generateTangents() {
        // Create and set tangent and bitangent
        Vec3f tangent;
        Vec3f bitangent;
        int faceCnt = (int)indices.size() / 3;
        for (int i = 0; i < faceCnt; ++i) {
            // Vertex indices used when fetching vertices
            int i1 = indices[3 * i];
            int i2 = indices[3 * i + 1];
            int i3 = indices[3 * i + 2];

            calcFaceTangentAndBitangent(vertices[i1],  //
                                        vertices[i2],  //
                                        vertices[i3],  //
                                        &tangent, &bitangent);

            vertices[i1].tangent = tangent;
            vertices[i2].tangent = tangent;
            vertices[i3].tangent = tangent;

            vertices[i1].bitangent = bitangent;
            vertices[i2].bitangent = bitangent;
            vertices[i3].bitangent = bitangent;
        }
    }

    void CudaMesh::calcFaceTangentAndBitangent(Vertex v1, Vertex v2, Vertex v3,
                                               Vec3f *out_tangent,
                                               Vec3f *out_bitangent) {
        // Calculate the two vectors for this face.
        Vec3f edge1 = v2.position - v1.position;
        Vec3f edge2 = v3.position - v1.position;
        // Calculate the tu and tv texture space vectors.
        Vec2f uv1 = v2.texCoord - v1.texCoord;
        Vec2f uv2 = v3.texCoord - v1.texCoord;

        // Calculate the denominator of the tangent/binormal equation.
        float f = 1.0f / (uv1[X] * uv2[Y] - uv2[X] * uv1[Y]);

        // Calculate the cross products and multiply by the coefficient to get
        // the tangent and binormal.
        (*out_tangent)[X] = (uv2[Y] * edge1[X] - uv1[Y] * edge2[X]) * f;
        (*out_tangent)[Y] = (uv2[Y] * edge1[Y] - uv1[Y] * edge2[Y]) * f;
        (*out_tangent)[Z] = (uv2[Y] * edge1[Z] - uv1[Y] * edge2[Z]) * f;
        *out_tangent = out_tangent->normalized();

        (*out_bitangent)[X] = (-uv2[X] * edge1[X] + uv1[X] * edge2[X]) * f;
        (*out_bitangent)[Y] = (-uv2[X] * edge1[Y] + uv1[X] * edge2[Y]) * f;
        (*out_bitangent)[Z] = (-uv2[X] * edge1[Z] + uv1[X] * edge2[Z]) * f;

        *out_bitangent = out_bitangent->normalized();
    }
};