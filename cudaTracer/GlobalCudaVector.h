#pragma once

#include <iostream>
#include <vector>
#include "cudaUtils.h"

template <typename T>
class GlobalCudaVector {
   public:
       //TODO: newing stuff like this is ugly...
    static GlobalCudaVector* fromVector(std::vector<T> vec) {
        GlobalCudaVector* gcv = new GlobalCudaVector;
        for (const auto& e : vec) {
            gcv->add(e);
        }
        return gcv;
    }
    GlobalCudaVector() { cudaMem = nullptr; }

    template <typename... Ts>
    GlobalCudaVector(Ts... initialData) {
        cudaMem = nullptr;
        add(initialData...);
    }

    ~GlobalCudaVector() { freeCudaMem(); }

    void copyToDevice() {
        if (memChanged) {
            allocateOnDevice();
            memChanged = false;
        }
        gpuErrchk(
            cudaMemcpy(cudaMem, &hostMem[0], byteCnt(), cudaMemcpyDefault));
        copied = true;
    }

    T* getDevMem() {
        if (!copied) {
            // Popup::error(__FILE__, __FUNCTION__, __LINE__,
            //             "Memory not copied to device");
            copyToDevice();
        }
        return cudaMem;
    }

    T& operator[](int idx) { return hostMem[idx]; }
    const T& operator[](int idx) const { return hostMem[idx]; }

    template <typename... Ts>
    void add(T first, Ts... rest) {
        add(first);
        add(rest...);
    }

    void add(T o) {
        memChanged = true;
        copied = false;
        hostMem.push_back(o);
    }

    int size() { return (int)hostMem.size(); }

    auto begin() { return hostMem.begin(); }
    auto end() { return hostMem.end(); }

   private:
    void freeCudaMem() {
        if (cudaMem != nullptr) {
            gpuErrchk(cudaFree(cudaMem));
        }
    }

    void allocateOnDevice() {
#ifdef _DEBUG
        std::cerr << "Allocating CUDA mem for " << size() << " '"
                  << typeid(T).name() << "' (" << sizeof(T) << "). Totally "
                  << byteCnt() << " bytes" << std::endl;
#endif

        freeCudaMem();
        gpuErrchk(cudaMalloc((void**)&cudaMem, byteCnt()));
    }

    int byteCnt() { return sizeof(T) * (int)hostMem.size(); }

    bool memChanged = true;
    bool copied = false;
    std::vector<T> hostMem;
    T* cudaMem;
};