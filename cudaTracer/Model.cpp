#include "Model.h"

Model::Model() { clear(); }

void Model::clear() {
    indices.clear();
    vertices.clear();
    useBlendMap = false;
    blendMapPath = "";
    materials.clear();
}

void Model::addVertex(Vertex p_vertex) { vertices.push_back(p_vertex); }

void Model::setVertices(std::vector<Vertex> p_vertices) {
    vertices = p_vertices;
}

void Model::addIndex(int p_index) { indices.push_back(p_index); }

void Model::setIndices(std::vector<int> p_indices) { indices = p_indices; }

void Model::setBlendMapPath(std::string p_blendMapPath) {
    blendMapPath = p_blendMapPath;
}

void Model::addMaterial(Material p_material) {
    materials.push_back(p_material);
}

void Model::addMaterial(std::string p_texturePath, std::string p_normalMapPath,
                        std::string p_specularMap) {
    Material material;
    material.texturePath = p_texturePath;
    material.normalMapPath = p_normalMapPath;
    material.specularMapPath = p_specularMap;
    materials.push_back(material);
}

void Model::setNormalMaps(std::vector<std::string> p_normalMapPaths) {
    if (materials.size() < p_normalMapPaths.size())
        materials.resize(p_normalMapPaths.size());
    for (unsigned int i = 0; i < p_normalMapPaths.size(); i++)
        materials[i].normalMapPath = p_normalMapPaths[i];
}

void Model::setTextures(std::vector<std::string> p_texturePaths) {
    if (materials.size() < p_texturePaths.size())
        materials.resize(p_texturePaths.size());
    for (unsigned int i = 0; i < p_texturePaths.size(); i++)
        materials[i].texturePath = p_texturePaths[i];
}

void Model::setSpecularMap(std::vector<std::string> p_specularMapPaths) {
    if (materials.size() < p_specularMapPaths.size())
        materials.resize(p_specularMapPaths.size());
    for (unsigned int i = 0; i < p_specularMapPaths.size(); i++)
        materials[i].specularMapPath = p_specularMapPaths[i];
}

void Model::setUseBlendMap(bool p_useBlendMap) { useBlendMap = p_useBlendMap; }

std::vector<Vertex> Model::getVertices() const { return vertices; }

std::vector<int> Model::getIndices() const { return indices; }

std::string Model::getBlendMapPath() const { return blendMapPath; }

std::vector<Material> Model::getMaterials() const { return materials; }

std::vector<std::string> Model::getNormalMapPaths() const {
    std::vector<std::string> normalMapPaths;
    for (unsigned int i = 0; i < materials.size(); i++)
        normalMapPaths.push_back(materials[i].normalMapPath);
    return normalMapPaths;
}

std::vector<std::string> Model::getTexturePaths() const {
    std::vector<std::string> texturePaths;
    for (unsigned int i = 0; i < materials.size(); i++)
        texturePaths.push_back(materials[i].texturePath);
    return texturePaths;
}

std::vector<std::string> Model::getSpecularMapPaths() const {
    std::vector<std::string> specularMapPaths;
    for (unsigned int i = 0; i < materials.size(); i++)
        specularMapPaths.push_back(materials[i].specularMapPath);
    return specularMapPaths;
}

int Model::getNumMaterials() const { return (int)materials.size(); }

int Model::getNumVertices() const { return (int)vertices.size(); }

int Model::getNumIndices() const { return (int)indices.size(); }

bool Model::getUseBlendMap() const { return useBlendMap; }