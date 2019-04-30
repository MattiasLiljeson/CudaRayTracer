#pragma once

template <typename T, int Size>
class GlobalCudaArray {
   public:
    GlobalCudaArray() { allocateOnDevice(); }

    ~GlobalCudaArray() { gpuErrchk(cudaFree(cudaMem)); }

    void allocateOnDevice() { gpuErrchk(cudaMalloc((void**)&cudaMem, size)); }

    void copyToDevice() {
        gpuErrchk(
            cudaMemcpy(cudaMem, &hostMem[0], size, cudaMemcpyHostToDevice));
        copied = true;
    }

    T* getDevMem() {
        if (!copied) {
            Utils::error(__FILE__, __FUNCTION__, __LINE__,
                         "Memory not copied to device");
        }
        return cudaMem;
    }

    constexpr int getSize() { return Size; } 

    T& operator[](int idx) {
        return hostMem[idx];
    }
    const T& operator[](int idx) const { return hostMem[idx]; }

   private:
    bool copied = false;
    T hostMem[Size];
    T* cudaMem;
    const int size = sizeof(T) * Size;
};