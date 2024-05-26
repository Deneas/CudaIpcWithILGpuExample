namespace CSharp
{
    using System.Runtime.CompilerServices;
    using System.Runtime.InteropServices;
    using ILGPU.Runtime.Cuda;

    /// <summary>
    /// Extends the ILGPU CUDA API to allow for Interprocess Communication.
    /// </summary>
    public static partial class CudaAPIExt
    {
        public const string LibNameWindows = CudaAPI.LibNameWindows;
        public const string LibNameLinux = CudaAPI.LibNameLinux; // MacOs has the same name

        private static partial class Windows
        {
            [LibraryImport(LibNameWindows, EntryPoint = "cuIpcGetMemHandle")]
            public static partial
                CudaError cuIpcGetMemHandle(
                    out CudaIpcMemHandle ipcMemHandle,
                    IntPtr devPtr);
            
            [LibraryImport(LibNameWindows, EntryPoint = "cuIpcOpenMemHandle")]
            public static partial
                CudaError cuIpcOpenMemHandle(
                    out IntPtr devPtr,
                    CudaIpcMemHandle ipcMemHandle,
                    CudaIpcMemFlags flags);

            [LibraryImport(LibNameWindows, EntryPoint = "cuIpcCloseMemHandle")]
            public static partial
                CudaError cuIpcCloseMemHandle(
                    IntPtr devPtr);
        }

        private static partial class Linux
        {
            [LibraryImport(LibNameLinux, EntryPoint = "cuIpcGetMemHandle")]
            public static partial
                CudaError cuIpcGetMemHandle(
                    out CudaIpcMemHandle ipcMemHandle,
                    IntPtr devPtr);
            
            [LibraryImport(LibNameLinux, EntryPoint = "cuIpcOpenMemHandle")]
            public static partial
                CudaError cuIpcOpenMemHandle(
                    out IntPtr devPtr,
                    CudaIpcMemHandle ipcMemHandle,
                    CudaIpcMemFlags flags);

            [LibraryImport(LibNameLinux, EntryPoint = "cuIpcCloseMemHandle")]
            public static partial
                CudaError cuIpcCloseMemHandle(
                    IntPtr devPtr);
        }
        
        /// <summary>
        /// Get the IPC handle for a memory buffer.
        /// </summary>
        /// <param name="cudaAccelerator">Accelerator of the CUDA device to be used</param>
        /// <param name="ipcMemHandle"></param>
        /// <param name="devPtr"></param>
        /// <returns>CUDA result code</returns>
        /// <remarks>This will zero the memory in the buffer!</remarks>
        public static CudaError IpcGetMemHandle(
            this CudaAccelerator cudaAccelerator,
            out CudaIpcMemHandle ipcMemHandle,
            IntPtr devPtr)
        {
            cudaAccelerator.Bind();
            return OperatingSystem.IsWindows()
                ? Windows.cuIpcGetMemHandle(out ipcMemHandle, devPtr)
                : Linux.cuIpcGetMemHandle(out ipcMemHandle, devPtr);
        }

        /// <summary>
        /// Open a memory buffer from an IPC handle.
        /// </summary>
        /// <param name="cudaAccelerator">Accelerator of the CUDA device to be used</param>
        /// <param name="devPtr">Newly allocated memory</param>
        /// <param name="ipcMemHandle">A IPC memory handle from another process</param>
        /// <param name="flags"><see cref="CudaIpcMemFlags"/></param>
        /// <returns>CUDA result code</returns>
        /// <remarks>This will not work with an IPC handle from the same process.</remarks>
        public static
            CudaError IpcOpenMemHandle(
                this CudaAccelerator cudaAccelerator,
                out IntPtr devPtr,
                CudaIpcMemHandle ipcMemHandle,
                CudaIpcMemFlags flags)
        {
            cudaAccelerator.Bind();
            return OperatingSystem.IsWindows()
                ? Windows.cuIpcOpenMemHandle(out devPtr, ipcMemHandle, flags)
                : Linux.cuIpcOpenMemHandle(out devPtr, ipcMemHandle, flags);
        }
        /// <summary>
        /// Close a memory buffer opened with <see cref="IpcOpenMemHandle"/>.
        /// </summary>
        /// <param name="cudaAccelerator">Accelerator of the CUDA device to be used</param>
        /// <param name="devPtr">Newly allocated memory</param>
        /// <returns>CUDA result code</returns>
        /// <remarks>   This will decrease the reference count of memory in <see cref="devPtr"/> by one,
        ///             only if the count reaches 0 the memory will be unmapped.
        ///             The original memory in the exported process and mappings in other processes will be unaffected.
        /// </remarks>
        public static
            CudaError IpcCloseMemHandle(
                this CudaAccelerator cudaAccelerator,
                IntPtr devPtr)
        {
            cudaAccelerator.Bind();
            return OperatingSystem.IsWindows()
                ? Windows.cuIpcCloseMemHandle(devPtr)
                : Linux.cuIpcCloseMemHandle(devPtr);
        }
    }

    /// <summary>
    /// Represents a CUDA IPC memory handle.
    /// </summary>
    public struct CudaIpcMemHandle
    {
        public Handle Data;
    
        [InlineArray(64)]
        public struct Handle
        {
            private Byte Element;
        }
    }

    /// <summary>
    /// Controls the behaviour of <see cref="CudaAPIExt.IpcOpenMemHandle">IpcOpenMemHandle</see>
    /// </summary>
    [Flags]
    public enum CudaIpcMemFlags
    {
        None = 0x0,
        LazyEnablePeerAccess = 0x1
    }
}