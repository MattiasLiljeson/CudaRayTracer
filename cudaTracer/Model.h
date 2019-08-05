#ifndef MODEL_H
#define MODEL_H

#include <string>
#include <vector>

//#include "Utils.h"
#include "Vertex.cuh"
namespace model {
struct ObbFromFile {
    bool defined;
    enum { X, Y, Z };
    float normals[3][3];
    float lengths[3];

    ObbFromFile() {
        defined = false;
        for (int i = 0; i < 3; i++) {
            lengths[i] = 0.0f;
            for (int j = 0; j < 3; j++) {
                normals[i][j] = 0.0f;
            }
        }
    }
};

struct Material {
    std::string mtlName;
    std::string texturePath;
    std::string normalMapPath;
    std::string specularMapPath;
};

class Model {
   public:
    // These are public for speed
    std::string name;

    // Blend maps are not implemented yet
    bool useBlendMap;
    std::string blendMapPath;

    ObbFromFile obb;  // Used for culling
    std::vector<int> indices;
    std::vector<Vertex> vertices;
    std::vector<Material> materials;

    Model();
    void clear();
    void addVertex(Vertex p_vertex);
    void setVertices(std::vector<Vertex> p_vertices);
    void addIndex(int p_index);
    void setIndices(std::vector<int> p_indices);
    void setBlendMapPath(std::string p_blendMapPath);
    void addMaterial(Material p_material);
    void addMaterial(std::string p_texturePath, std::string p_normalMapPath,
                     std::string p_specularMap);
    void setNormalMaps(std::vector<std::string> p_normalMapPaths);
    void setTextures(std::vector<std::string> p_texturePaths);
    void setSpecularMap(std::vector<std::string> p_specularMapPaths);
    void setUseBlendMap(bool p_useBlendMap);

    std::vector<Vertex> getVertices() const;
    std::vector<int> getIndices() const;
    std::string getBlendMapPath() const;
    std::vector<Material> getMaterials() const;
    std::vector<std::string> getNormalMapPaths() const;
    std::vector<std::string> getTexturePaths() const;
    std::vector<std::string> getSpecularMapPaths() const;
    int getNumMaterials() const;
    int getNumVertices() const;
    int getNumIndices() const;
    bool getUseBlendMap() const;
};
}  // namespace Model
#endif  // MODEL_H