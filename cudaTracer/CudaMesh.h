#pragma once
#include <lodepng.h>
#include <vector>
#include "BVH.h"
#include "BoundingBox.cuh"
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
    GlobalCudaVector<Triangle> triangles;
    GlobalCudaVector<LinearNode> nodes;
    GlobalCudaVector<unsigned char> diffuseData;
    GlobalCudaVector<unsigned char> normalsData;
    Mesh mesh;
    Shape shape;

    CudaMesh(const Model &model) {
        vertices = GlobalCudaVector<Vertex>::fromVector(model.getVertices());
        indices = GlobalCudaVector<int>::fromVector(model.getIndices());

        std::vector<Triangle> tris;
        for (int i = 0; i < model.getIndices().size(); i += 3) {
            Triangle tri(model.getIndices()[i],      //
                         model.getIndices()[i + 1],  //
                         model.getIndices()[i + 2]);
            tris.push_back(tri);
        }
        BVH::BvhFactory bvh(model.getVertices(), tris);
        triangles = GlobalCudaVector<Triangle>::fromVector(tris);
        generateTangents();
        nodes = GlobalCudaVector<LinearNode>::fromVector(bvh.nodes);
        
        // TODO: 3 is just a happy guess
        int triCnt = model.getNumVertices() / 3;

        std::string diffuseFname = model.getMaterials()[0].texturePath;
        Texture diffuseTex = loadPngFromDisk(diffuseFname, &diffuseData);
        std::string normalsFname = model.getMaterials()[0].normalMapPath;
        Texture normalsTex = loadPngFromDisk(normalsFname, &normalsData);

        mesh = Mesh(triangles.getDevMem(), triCnt, vertices.getDevMem(),
                    diffuseTex, normalsTex, nodes.getDevMem());
        shape = Shape(mesh);
        shape.material.materialType = Object::DIFFUSE_AND_GLOSSY;
    }

    static Texture CudaMesh::loadPngFromDisk(
        std::string texFname, GlobalCudaVector<unsigned char> *data) {
        std::vector<unsigned char> image;  // the raw pixels
        unsigned width, height;
        unsigned error = lodepng::decode(image, width, height, texFname);
        // if there's an error, display it. TODO: Replace this with something
        // proper..
        if (error) {
            std::string errMsg = "Error: ";
            errMsg += error;
            errMsg += ", ";
            errMsg += lodepng_error_text(error);
            errMsg += ". When decoding the png file: ";
            errMsg += texFname;
            Popup::error(errMsg);
        }
        *data = GlobalCudaVector<unsigned char>::fromVector(image);
        return Texture(width, height, data->getDevMem());
    }

    //CudaMesh(std::vector<Vertex> vertices, std::vector<int> indices,
    //         int p_triCnt, std::vector<unsigned char> diffuseTexData,
    //         std::vector<unsigned char> bumpTexData, int texWidth,
    //         int texHeight) {
    //    this->vertices = GlobalCudaVector<Vertex>::fromVector(vertices);
    //    this->indices = GlobalCudaVector<int>::fromVector(indices);
    //    generateTangents();
    //    diffuseData =
    //        GlobalCudaVector<unsigned char>::fromVector(diffuseTexData);
    //    normalsData = GlobalCudaVector<unsigned char>::fromVector(bumpTexData);
    //    Texture diffuseTex =
    //        Texture(texWidth, texHeight, diffuseData.getDevMem());
    //    Texture bumpTex = Texture(texWidth, texHeight, normalsData.getDevMem());
    //    this->mesh = Mesh(this->indices.getDevMem(), p_triCnt,
    //                      this->vertices.getDevMem(), diffuseTex, bumpTex);
    //    shape = Shape(mesh);
    //    shape.material.materialType = Object::DIFFUSE_AND_GLOSSY;
    //}

    void CudaMesh::generateTangents() {
        // Create and set tangent and bitangent
        Vec3f tangent;
        Vec3f bitangent;
        int faceCnt = (int)triangles.size();
        for (int i = 0; i < faceCnt; ++i) {
            // Vertex indices used when fetching vertices
            int i1 = triangles[i][0];
            int i2 = triangles[i][1];
            int i3 = triangles[i][2];

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