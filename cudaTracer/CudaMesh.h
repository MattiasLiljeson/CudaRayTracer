#pragma once
#include <lodepng.h>
#include <vector>
#include "BVH.h"
#include "BoundingBox.cuh"
#include "GlobalCudaVector.h"
#include "Mesh.cuh"
#include "Model.h"
#include "Popup.h"
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
    GlobalCudaVector<unsigned char> specularData;
    Mesh mesh;
    Shape shape;

    CudaMesh(std::vector<Vertex> vertices, std::vector<int> indices,
             int p_triCnt, std::vector<unsigned char> diffuseAsChars,
             std::vector<unsigned char> normalsAsChars, int texWidth,
             int texHeight, Material material, Mat44f transformation) {
        prepareGeometry(vertices, indices);

        diffuseData =
            GlobalCudaVector<unsigned char>::fromVector(diffuseAsChars);
        normalsData =
            GlobalCudaVector<unsigned char>::fromVector(normalsAsChars);
        Texture diffuse = Texture(texWidth, texHeight, diffuseData.getDevMem());
        Texture normals = Texture(texWidth, texHeight, normalsData.getDevMem());

        createShape(diffuse, normals, normals, material, transformation);
    }

    CudaMesh(const model::Model &model, Material material,
             Mat44f transformation) {
        prepareGeometry(model.getVertices(), model.getIndices());

        std::string diffuseFname = model.getMaterials()[0].texturePath;
        Texture diffuse = loadPngFromDisk(diffuseFname, &diffuseData);
        std::string normalsFname = model.getMaterials()[0].normalMapPath;
        Texture normals = loadPngFromDisk(normalsFname, &normalsData);
        std::string specularFname = model.getMaterials()[0].specularMapPath;
        Texture specular = loadPngFromDisk(specularFname, &specularData);

        createShape(diffuse, normals, specular, material, transformation);
    }

    void prepareGeometry(std::vector<Vertex> vertices,
                         std::vector<int> indices) {
        this->vertices = GlobalCudaVector<Vertex>::fromVector(vertices);
        this->indices = GlobalCudaVector<int>::fromVector(indices);
        std::vector<Triangle> tris;
        for (int i = 0; i < indices.size(); i += 3) {
            Triangle tri(indices[i],      //
                         indices[i + 1],  //
                         indices[i + 2]);
            tris.push_back(tri);
        }
        BVH::BvhFactory<Triangle> bvh(vertices, tris, 10);
        triangles = GlobalCudaVector<Triangle>::fromVector(bvh.orderedPrims);
        generateTangents();
        nodes = GlobalCudaVector<LinearNode>::fromVector(bvh.nodes);
    }

    void createShape(Texture diffuse, Texture normals, Texture specular,
                     Material material, Mat44f transformation) {
        mesh =
            Mesh(triangles.getDevMem(), triangles.size(), vertices.getDevMem(),
                 diffuse, normals, specular, nodes.getDevMem(), transformation);

        BoundingBox shapeBb = nodes.getHostMemRef()[0].bb;
        shapeBb.bbmin = transformation.multPoint(shapeBb.bbmin);        
        shapeBb.bbmax = transformation.multPoint(shapeBb.bbmax);
        
        shape = Shape(mesh, shapeBb);
        shape.material = material;
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