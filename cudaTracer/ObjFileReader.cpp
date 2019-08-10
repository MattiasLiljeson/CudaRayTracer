#include <deque>
#include <fstream>
#include <sstream>

#include "ObjFileReader.h"
#include "Popup.h"

using model::Material;
using model::Model;
using model::ObbFromFile;

ObjFileReader::ObjFileReader() {
    // numVertices = 0;
    numIndices = 0;

    // TEST: split function
    // std::string a = "a b c  d ///   def/ghi   ";
    // std::deque<std::string> test1 = split(a, ' ');
    // std::deque<std::string> test2 = split(a, '/');
    // std::deque<std::string> test3 = split(a, 'd');
    // std::deque<std::string> test4 = split(a, 'm');
}

ObjFileReader::~ObjFileReader() {
    // Nothing to implement yet.
}

std::vector<Model> ObjFileReader::readFile(std::string pFolder,
                                           std::string pFileName,
                                           bool p_startAtZero) {
    // Clean from earlier uses of this object
    currModel.clear();
    models.clear();
    numIndices = 0;
    readNorm.clear();
    readPos.clear();
    readTexCoord.clear();
    materials.clear();

    // Create the object
    fileName = pFileName;
    folder = pFolder;
    startAtZero = p_startAtZero;

    std::string objFilePath = folder + fileName;
    std::fstream objFile(objFilePath);

    if (!objFile) {
        std::string msg =
            std::string("Could not read obj file: ") + objFilePath;
        Popup::error(__FILE__, __FUNCTION__, __LINE__, msg);
    } else {
        std::deque<std::string> lines;
        std::string tmp;
        while (!objFile.eof()) {
            getline(objFile, tmp);
            lines.push_back(tmp);
        }

        int count = 0;
        std::string line;
        // Words at a line is all characters that are grouped together and
        // separated from other words by ' '.
        std::vector<std::string> lineWords;
        // std::string prefix; // first word on a line
        for (unsigned int i = 0; i < lines.size(); i++) {
            count++;

            line = lines[i];
            if (line.size() > 0)  // protect from empty line
            {
                lineWords = split(line, ' ');
                lineWords.erase(lineWords.begin());

                // Roughly sorted in number of occurences in .obj-files
                if (line[0] == 'v') {
                    if (line[1] == 't')  // vt = tex coords
                        readTextureUV(lineWords);
                    else if (line[1] == 'n')  // vn = normals
                        readNormals(lineWords);
                    else  // v = pos coords
                        readVertices(lineWords);
                } else if (line[0] == 'f') {
                    readFaces(lineWords);
                } else if (line[0] == 'g') {
                    createModel();
                    int korv = 0;
                } else if (line[0] == 'm')  // mtllib = file containing material
                                            // definitions
                {
                    readMtlFile(lineWords);
                } else if (line[0] == 'u')  // usemtl = which material to use
                {
                    for (unsigned int i = 0; i < materials.size(); i++) {
                        if (materials[i].mtlName == lineWords.front())
                            currModel.addMaterial(materials[i]);
                    }
                }

                // lineWords.clear(); //clear the std::deque from old entries
            }
        }
        createModel();
    }
    return models;
}

void ObjFileReader::readMtlFile(std::vector<std::string> p_lineWords) {
    // Read .mtl-file;
    std::string mtlFileName = p_lineWords[0];

    std::string line;
    std::string prefix;
    std::fstream mtlFile(folder + mtlFileName);

    if (mtlFile.good()) {
        Material material;
        std::string mtlName;
        while (!mtlFile.eof()) {
            std::string temp = "";
            std::stringstream lineStreamMtl;
            getline(mtlFile, line);
            if (!line.empty()) {
                lineStreamMtl << line;
                lineStreamMtl >> prefix;
                std::transform(prefix.begin(), prefix.end(), prefix.begin(),
                               ::tolower);

                if (prefix == "newmtl") {
                    if (material.mtlName != "") {
                        materials.push_back(material);

                        material.mtlName = "";
                        material.texturePath = "";
                        material.normalMapPath = "";
                        material.specularMapPath = "";
                    }
                    lineStreamMtl >> material.mtlName;
                }
                // Diffuse and ambient are seen as the same component
                if (prefix == "map_kd" || prefix == "map_ka") {
                    lineStreamMtl >> temp;
                    material.texturePath = folder + temp;
                }
                // Specular map and specular highlight map are seen as the same
                else if (prefix == "map_ks" || prefix == "map_ns") {
                    lineStreamMtl >> temp;
                    material.specularMapPath = folder + temp;
                }
                // Bump maps are in this case seen as normal maps
                else if (prefix == "map_bump" || prefix == "bump") {
                    lineStreamMtl >> temp;
                    material.normalMapPath = folder + temp;
                }
            }
        }
        materials.push_back(material);
    }
}

void ObjFileReader::readNormals(std::vector<std::string> p_lineWords) {
    std::array<float, 3> normal;
    //_stream >> normal[Vertex::X] >> normal[Vertex::Y] >> normal[Vertex::Z];
    normal[Vertex::X] = (float)atof(p_lineWords[0].c_str());
    normal[Vertex::Y] = (float)atof(p_lineWords[1].c_str());
    normal[Vertex::Z] = (float)atof(p_lineWords[2].c_str());
    readNorm.push_back(normal);
}

void ObjFileReader::readVertices(std::vector<std::string> p_lineWords) {
    std::array<float, 3> pos;
    //_stream >> pos[Vertex::X] >> pos[Vertex::Y] >> pos[Vertex::Z];
    pos[Vertex::X] = (float)atof(p_lineWords[0].c_str());
    pos[Vertex::Y] = (float)atof(p_lineWords[1].c_str());
    pos[Vertex::Z] = (float)atof(p_lineWords[2].c_str());
    readPos.push_back(pos);
}

void ObjFileReader::readTextureUV(std::vector<std::string> p_lineWords) {
    std::array<float, 2> uv;
    //_stream >> uv[Vertex::U] >> uv[Vertex::V];
    uv[Vertex::U] = (float)atof(p_lineWords[0].c_str());
    uv[Vertex::V] = (float)atof(p_lineWords[1].c_str());
    readTexCoord.push_back(uv);
}

void ObjFileReader::readFaces(std::vector<std::string> p_lineWords) {
    if (p_lineWords.size() == 3)  // 3 vertices = triangle
    {
        createFace(p_lineWords);
    } else if (p_lineWords.size() == 4)  // 4 vertices = quad
    {
        createFace(triFromQuad(p_lineWords, 1));
        createFace(triFromQuad(p_lineWords, 2));
    }
}

void ObjFileReader::createFace(std::vector<std::string> p_lineWords) {
    // char tmp; // Used to "eat" '/'

    // Init as 1 to protect from empty std::string stream. (0 as effective
    // index)
    int indexPos = 1;
    int texPos = 1;
    int normPos = 1;

    std::vector<std::string> elements;
    for (int i = 0; i < 3; i++) {
        elements = split(p_lineWords[i], '/');
        Vertex vertex;
        indexPos = atoi(elements[0].c_str());
        texPos = atoi(elements[1].c_str());
        normPos = atoi(elements[2].c_str());
        //	_stream >> indexPos >> tmp >> texPos >> tmp >> normPos;

        vertex.position[Vertex::X] = readPos[indexPos - 1][Vertex::X];
        vertex.position[Vertex::Y] = readPos[indexPos - 1][Vertex::Y];
        vertex.position[Vertex::Z] = readPos[indexPos - 1][Vertex::Z];

        if (readTexCoord.size() > 0)  // If the model has uv coords
        {
            vertex.texCoord[Vertex::U] = readTexCoord[texPos - 1][Vertex::U];
            vertex.texCoord[Vertex::V] = readTexCoord[texPos - 1][Vertex::V];
        } else {  // Else, Use hardcoded
            vertex.texCoord[Vertex::U] = 0.3f * i;
            vertex.texCoord[Vertex::V] = 0.3f * i;
        }

        if (!startAtZero) {  // Works for bth logo
            vertex.normal[Vertex::X] = readNorm[normPos - 1][Vertex::X];
            vertex.normal[Vertex::Y] = readNorm[normPos - 1][Vertex::Y];
            vertex.normal[Vertex::Z] = readNorm[normPos - 1][Vertex::Z];
        } else {  // for teapot
            vertex.normal[Vertex::X] = readNorm[normPos][Vertex::X];
            vertex.normal[Vertex::Y] = readNorm[normPos][Vertex::Y];
            vertex.normal[Vertex::Z] = readNorm[normPos][Vertex::Z];
        }

        currModel.addVertex(vertex);
        currModel.addIndex(numIndices);
        numIndices++;
    }
}

void ObjFileReader::createModel() {
    // If a model exists, push it back
    if (currModel.getNumIndices() > 0) {
        currModel.setUseBlendMap(
            false);                 // Obj-files have no support for blendmaps
        currModel.name = fileName;  // + " " + models.size();
        models.push_back(currModel);
    }

    // Start from phresh
    currModel.clear();
    numIndices = 0;
}

std::vector<std::string> ObjFileReader::split(std::string p_str, char p_delim) {
    std::vector<std::string> strings;
    std::string currStr;

    for (unsigned int i = 0; i < p_str.length(); i++) {
        if (p_str[i] != p_delim) {
            currStr.push_back(p_str[i]);
        } else {
            if (currStr.size() > 0)  // Protect from trailing delimiters
                strings.push_back(currStr);
            currStr.clear();
        }
    }

    if (currStr.size() > 0)  // Protect from trailing delimiters
        strings.push_back(currStr);

    return strings;
}

std::vector<std::string> ObjFileReader::triFromQuad(
    std::vector<std::string> p_lineWords, int p_triNum) {
    std::vector<std::string> newLineWords;
    if (p_triNum == 1) {
        newLineWords.push_back(p_lineWords[0]);
        newLineWords.push_back(p_lineWords[1]);
        newLineWords.push_back(p_lineWords[2]);
    } else if (p_triNum == 2) {
        newLineWords.push_back(p_lineWords[0]);
        newLineWords.push_back(p_lineWords[2]);
        newLineWords.push_back(p_lineWords[3]);
    }
    return newLineWords;
}