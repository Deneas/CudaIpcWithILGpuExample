namespace CSharp
{
    using System.Runtime.CompilerServices;
    using ILGPU.Runtime;
    using ILGPU.Runtime.Cuda;
    using ILGPU.Util;

    public static class CudaMemoryBufferExt
    {
        [UnsafeAccessor(UnsafeAccessorKind.Method, Name = "set_NativePtr")]
        private static extern void SetNativePtr(MemoryBuffer buffer, IntPtr nativePtr);
    
        [UnsafeAccessor(UnsafeAccessorKind.Method, Name = "Init")]
        private static extern void ReinitBuffer(MemoryBuffer buffer,long length, int elementSize);

        [UnsafeAccessor(UnsafeAccessorKind.Field, Name = "isDisposed")]
        private static extern ref bool IsDisposedField(DisposeBase buffer);
        
        // ToDo: In an actual implementation we either create a new class inheriting from MemoryBuffer
        // or extend CudaMemoryBuffer to handle IPC memory
        
        public static bool TryCreateBufferFromIpcMemHandle(this CudaAccelerator cudaAccelerator,
            out CudaMemoryBuffer cudaMemoryBuffer, 
            CudaIpcMemHandle ipcMemHandle,
            int length,
            int elementSize)
        {
            cudaMemoryBuffer = new CudaMemoryBuffer(cudaAccelerator, 0, 1);
            var result = cudaAccelerator.IpcOpenMemHandle(out var nativePtr, ipcMemHandle,
                CudaIpcMemFlags.LazyEnablePeerAccess);
            if (result == CudaError.CUDA_SUCCESS)
            {
                ref bool isDisposedField = ref IsDisposedField(cudaMemoryBuffer);
                isDisposedField = true;
                SetNativePtr(cudaMemoryBuffer, nativePtr);
                ReinitBuffer(cudaMemoryBuffer, length, elementSize);
                return true;
            }

            return false;
        }

        public static void IpcClose(this CudaMemoryBuffer cudaMemoryBuffer)
        {
            var cudaAccel = (CudaAccelerator)cudaMemoryBuffer.Accelerator;
            cudaAccel.IpcCloseMemHandle(cudaMemoryBuffer.NativePtr);
            SetNativePtr(cudaMemoryBuffer, IntPtr.Zero);
            ref bool isDisposed = ref IsDisposedField(cudaMemoryBuffer);
            isDisposed = false;
            cudaMemoryBuffer.Dispose();
        }
    }
}