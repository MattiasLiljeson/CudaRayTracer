#include "common.hlsl"

float4 PS( VertexOut p_input ) : SV_TARGET
{
	uint3 idx = uint3( p_input.position.x, p_input.position.y, 0 );
	return g_tex.Load(idx);
	return float4( p_input.texCoord.x, p_input.texCoord.y, 0.3f, 1.0f );
}