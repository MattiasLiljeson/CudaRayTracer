#ifndef MAIN_CUH
#define MAIN_CUH

#include "Options.cuh"
#include "Scene.cuh"

void cudamain(const Options &options, const Scene &scene, const void *surface,
              size_t pitch, int blockDim, unsigned char *rngStates);

unsigned char *cu_initCurand(int width, int height);
void cu_cleanCurand(unsigned char *p_rngStates);
#endif  // !MAIN_CUH