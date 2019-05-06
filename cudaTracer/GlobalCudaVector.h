#pragma once

#include <vector>
#include "cudaUtils.h"

template <typename T>
class GlobalCudaVector {
   public:
    GlobalCudaVector() { cudaMem = nullptr; }

    ~GlobalCudaVector() { freeCudaMem(); }

    void copyToDevice() {
        if (memChanged) {
            allocateOnDevice();
        }
        gpuErrchk(
            cudaMemcpy(cudaMem, &hostMem[0], byteCnt(), cudaMemcpyDefault));
        copied = true;
    }

    T* getDevMem() {
        if (!copied) {
            Utils::error(__FILE__, __FUNCTION__, __LINE__,
                         "Memory not copied to device");
        }
        return cudaMem;
    }

    T& operator[](int idx) { return hostMem[idx]; }
    const T& operator[](int idx) const { return hostMem[idx]; }

    void pushBack(T o) {
        memChanged = true;
        copied = false;
        hostMem.push_back(o);
    }

    int size() { return hostMem.size(); }

    auto begin() { return hostMem.begin(); }
    auto end() { return hostMem.end(); }

   private:
    void freeCudaMem() {
        if (cudaMem != nullptr) {
            gpuErrchk(cudaFree(cudaMem));
        }
    }

    void allocateOnDevice() {
        freeCudaMem();
        gpuErrchk(cudaMalloc((void**)&cudaMem, byteCnt()));
    }

    int byteCnt() { return sizeof(T) * hostMem.size(); }

    bool memChanged = true;
    bool copied = false;
    std::vector<T> hostMem;
    T* cudaMem;
};