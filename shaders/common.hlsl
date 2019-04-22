#pragma once

Texture2D g_tex : register (t0);

struct VertexIn
{
	float3 position : POSITION;
	float2 texCoord : TEXCOORD;
};

struct VertexOut
{
	float4 position : SV_POSITION;
	float2 texCoord : TEXCOORD;
};