#include "CubeModelFactory.h"
CubeModelFactory::CubeModelFactory()
{
	//empty
}

CubeModelFactory::~CubeModelFactory()
{
	//empty
}

vector<Vertex> CubeModelFactory::createVertices()
{
	// create vertices to represent the corners of the Cube
	vector<Vertex> vertices;
	vertices.push_back(Vertex(-1.0f, -1.0f, 1.0f,  0.0f, 0.0f, 1.0f,  0.0f, 0.0f));
	vertices.push_back(Vertex(1.0f, -1.0f, 1.0f,   0.0f, 0.0f, 1.0f,  0.0f, 1.0f));
	vertices.push_back(Vertex(-1.0f, 1.0f, 1.0f,   0.0f, 0.0f, 1.0f,  1.0f, 0.0f));
	vertices.push_back(Vertex(1.0f, 1.0f, 1.0f,    0.0f, 0.0f, 1.0f,  1.0f, 1.0f));

	vertices.push_back(Vertex(-1.0f, -1.0f, -1.0f, 0.0f, 0.0f, -1.0f, 0.0f, 0.0f));
	vertices.push_back(Vertex(-1.0f, 1.0f, -1.0f,  0.0f, 0.0f, -1.0f, 0.0f, 1.0f));
	vertices.push_back(Vertex(1.0f, -1.0f, -1.0f,  0.0f, 0.0f, -1.0f, 1.0f, 0.0f));
	vertices.push_back(Vertex(1.0f, 1.0f, -1.0f,   0.0f, 0.0f, -1.0f, 1.0f, 1.0f));

	vertices.push_back(Vertex(-1.0f, 1.0f, -1.0f,  0.0f, 1.0f, 0.0f,  0.0f, 0.0f));
	vertices.push_back(Vertex(-1.0f, 1.0f, 1.0f,   0.0f, 1.0f, 0.0f,  0.0f, 1.0f));
	vertices.push_back(Vertex(1.0f, 1.0f, -1.0f,   0.0f, 1.0f, 0.0f,  1.0f, 0.0f));
	vertices.push_back(Vertex(1.0f, 1.0f, 1.0f,    0.0f, 1.0f, 0.0f,  1.0f, 1.0f));

	vertices.push_back(Vertex(-1.0f, -1.0f, -1.0f, 0.0f, -1.0f, 0.0f, 0.0f, 0.0f));
	vertices.push_back(Vertex(1.0f, -1.0f, -1.0f,  0.0f, -1.0f, 0.0f, 0.0f, 1.0f));
	vertices.push_back(Vertex(-1.0f, -1.0f, 1.0f,  0.0f, -1.0f, 0.0f, 1.0f, 0.0f));
	vertices.push_back(Vertex(1.0f, -1.0f, 1.0f,   0.0f, -1.0f, 0.0f, 1.0f, 1.0f));

	vertices.push_back(Vertex(1.0f, -1.0f, -1.0f,  1.0f, 0.0f, 0.0f,  0.0f, 0.0f));
	vertices.push_back(Vertex(1.0f, 1.0f, -1.0f,   1.0f, 0.0f, 0.0f,  0.0f, 1.0f));
	vertices.push_back(Vertex(1.0f, -1.0f, 1.0f,   1.0f, 0.0f, 0.0f,  1.0f, 0.0f));
	vertices.push_back(Vertex(1.0f, 1.0f, 1.0f,    1.0f, 0.0f, 0.0f,  1.0f, 1.0f));

	vertices.push_back(Vertex(-1.0f, -1.0f, -1.0f, -1.0f, 0.0f, 0.0f, 0.0f, 0.0f));
	vertices.push_back(Vertex(-1.0f, -1.0f, 1.0f,  -1.0f, 0.0f, 0.0f, 0.0f, 1.0f));
	vertices.push_back(Vertex(-1.0f, 1.0f, -1.0f,  -1.0f, 0.0f, 0.0f, 1.0f, 0.0f));
	vertices.push_back(Vertex(-1.0f, 1.0f, 1.0f,   -1.0f, 0.0f, 0.0f, 1.0f, 1.0f));

	return vertices;
}

vector<int> CubeModelFactory::createIndicies()
{
	// create the index buffer out of DWORDs
	vector<int> indices(36);
	indices[0] = 0;   indices[1]  = 1;  indices[2] = 2;    // side 1
	indices[3] = 2;   indices[4]  = 1;  indices[5] = 3;
	indices[6] = 4;   indices[7]  = 5;  indices[8] = 6;    // side 2
	indices[9] = 6;   indices[10] = 5;  indices[11] = 7;
	indices[12] = 8;  indices[13] = 9;  indices[14] = 10;    // side 3
	indices[15] = 10; indices[16] = 9;  indices[17] = 11;
	indices[18] = 12; indices[19] = 13; indices[20] = 14;    // side 4
	indices[21] = 14; indices[22] = 13; indices[23] = 15;
	indices[24] = 16; indices[25] = 17; indices[26] = 18;    // side 5
	indices[27] = 18; indices[28] = 17; indices[29] = 19;
	indices[30] = 20; indices[31] = 21; indices[32] = 22;    // side 6
	indices[33] = 22; indices[34] = 21; indices[35] = 23;

	return indices;
}

Model CubeModelFactory::createCubeObject(string p_texFilePath)
{
	Model model;
	model.setIndices(createIndicies());
	model.setVertices(createVertices());
	model.addMaterial(p_texFilePath, "", ""); // HACK: no normalmap
	model.setUseBlendMap(false); //HACK: hard-coded
	model.name = "Cube from CubeModelfactory"; 

	return model;
}