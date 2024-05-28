namespace CSharp
{
    using ILGPU.Runtime;
    using ILGPU.Runtime.Cuda;

    public static class CudaAcceleratorExtensions
    {
        public static CudaIpcMemoryBuffer MapIpc(this CudaAccelerator accelerator, CudaIpcMemHandle ipcMemHandle, long length, int elementSize)
        {
            return new CudaIpcMemoryBuffer(accelerator, ipcMemHandle, length, elementSize);
        }

        public static CudaIpcMemHandle GetIpcHandle(this CudaAccelerator accelerator, MemoryBuffer memoryBuffer)
        {
            CudaException.ThrowIfFailed(
            CudaAPIExt.IpcGetMemHandle(accelerator, out var ipcMemHandle, memoryBuffer.NativePtr)
            );
            return ipcMemHandle;
        }
    }
}