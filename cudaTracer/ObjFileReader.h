#ifndef OBJFILEREADER_H
#define OBJFILEREADER_H

#include <array>
#include <string>
#include <vector>

#include "Model.h"



class ObjFileReader {
   private:
    model::Model currModel;
    std::vector<model::Model> models;
    std::string fileName;
    std::string folder;
    bool startAtZero;

    // File parsing
    int numIndices;
    std::vector<std::array<float, 3>> readNorm;
    std::vector<std::array<float, 3>> readPos;
    std::vector<std::array<float, 2>> readTexCoord;

    // std::vector<std::string> mtlNames;
    model::Material currMaterial;
    std::vector<model::Material> materials;

    // std::vector<Vertex> vertices;

    void readObjFile();
    void readMtlFile(std::vector<std::string> p_lineWords);
    void readNormals(std::vector<std::string> p_lineWords);
    void readVertices(std::vector<std::string> p_lineWords);
    void readTextureUV(std::vector<std::string> p_lineWords);
    void readFaces(std::vector<std::string> p_lineWords);
    void createModel();
    void createFace(std::vector<std::string> p_lineWords);

    std::vector<std::string> triFromQuad(std::vector<std::string> p_lineWords,
                                         int p_triNum);
    std::vector<std::string> split(std::string p_str, char p_delim);

   public:
    ObjFileReader();
    ~ObjFileReader();

    std::vector<model::Model> readFile(std::string pFolder, std::string pFileName,
                                bool p_startAtZero);
};

#endif  // OBJFILEREADER_H