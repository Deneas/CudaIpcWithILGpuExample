namespace CSharp
{
    using ILGPU.Runtime.Cuda;

    public static class CudaAcceleratorExtensions
    {
        public static CudaIpcMemoryBuffer MapIpc(this CudaAccelerator accelerator, CudaIpcMemHandle ipcMemHandle, long length, int elementSize)
        {
            return new CudaIpcMemoryBuffer(accelerator, ipcMemHandle, length, elementSize);
        }
    }
}