﻿namespace CSharp
{
    using ILGPU;
    using ILGPU.Runtime;
    using ILGPU.Runtime.Cuda;
    using static ILGPU.Runtime.Cuda.CudaAPI;

    public class CudaIpcMemoryBuffer : MemoryBuffer
    {
        #region Static

        /// <summary>
        /// Performs a Cuda memset operation.
        /// </summary>
        /// <typeparam name="T">The element type.</typeparam>
        /// <param name="stream">
        /// The Cuda stream to use (can be null to preserve backwards compatibility).
        /// </param>
        /// <param name="value">The value to write into the buffer.</param>
        /// <param name="targetView">The target view to write to.</param>
        public static void CudaMemSet<T>(
            CudaStream? stream,
            byte value,
            in ArrayView<T> targetView)
            where T : unmanaged
        {
            if (targetView.GetAcceleratorType() != AcceleratorType.Cuda)
            {
                throw new NotSupportedException("Target accelerator not supported!");
            }

            using var binding = stream?.Accelerator.BindScoped();
            CudaException.ThrowIfFailed(
                CurrentAPI.Memset(
                    targetView.LoadEffectiveAddressAsPtr(),
                    value,
                    new IntPtr(targetView.LengthInBytes),
                    stream));
        }

        /// <summary>
        /// Performs a Cuda copy operation.
        /// </summary>
        /// <typeparam name="T">The element type.</typeparam>
        /// <param name="stream">
        /// The Cuda stream to use (can be null to preserve backwards compatibility).
        /// </param>
        /// <param name="sourceView">The source view to copy from.</param>
        /// <param name="targetView">The target view to copy to.</param>
        public static void CudaCopy<T>(
            CudaStream? stream,
            in ArrayView<T> sourceView,
            in ArrayView<T> targetView)
            where T : unmanaged
        {
            var sourceType = sourceView.GetAcceleratorType();
            var targetType = targetView.GetAcceleratorType();

            if (sourceType == AcceleratorType.OpenCL ||
                targetType == AcceleratorType.OpenCL)
            {
                throw new NotSupportedException("Target accelerator not supported!");
            }

            var sourceAddress = sourceView.LoadEffectiveAddressAsPtr();
            var targetAddress = targetView.LoadEffectiveAddressAsPtr();
            var length = new IntPtr(targetView.LengthInBytes);

            // a) Copy from CPU to GPU
            // b) Copy from GPU to CPU
            // c) Copy from GPU to GPU
            using var binding = stream?.Accelerator.BindScoped();
            CudaException.ThrowIfFailed(
                CurrentAPI.MemcpyAsync(
                    targetAddress,
                    sourceAddress,
                    length,
                    stream));
        }

        #endregion

        #region Instance

        /// <summary>
        /// Constructs a new Cuda IPC buffer.
        /// </summary>
        /// <param name="accelerator">The accelerator.</param>
        /// <param name="ipcMemHandle">The IPC memory handle.</param>
        /// <param name="length">The length of this buffer.</param>
        /// <param name="elementSize">The element size.</param>
        public CudaIpcMemoryBuffer(
            CudaAccelerator accelerator,
            CudaIpcMemHandle ipcMemHandle,
            long length,
            int elementSize)
            : base(accelerator, length, elementSize)
        {
            if (LengthInBytes == 0)
            {
                NativePtr = IntPtr.Zero;
            }
            else
            {
                CudaException.ThrowIfFailed(
                    CudaAPIExt.IpcOpenMemHandle(
                        accelerator,
                        out IntPtr resultPtr,
                        ipcMemHandle,
                        CudaIpcMemFlags.LazyEnablePeerAccess));
                NativePtr = resultPtr;
            }
        }

        #endregion

        #region Methods

        /// <inheritdoc/>
        protected override unsafe void MemSet(
            AcceleratorStream stream,
            byte value,
            in ArrayView<byte> targetView) =>
            CudaMemSet(stream as CudaStream, value, targetView);

        /// <inheritdoc/>
        protected override void CopyFrom(
            AcceleratorStream stream,
            in ArrayView<byte> sourceView,
            in ArrayView<byte> targetView) =>
            CudaCopy(stream as CudaStream, sourceView, targetView);

        /// <inheritdoc/>
        protected override unsafe void CopyTo(
            AcceleratorStream stream,
            in ArrayView<byte> sourceView,
            in ArrayView<byte> targetView) =>
            CudaCopy(stream as CudaStream, sourceView, targetView);

        #endregion

        #region IDisposable

        /// <summary>
        /// Disposes this Cuda buffer.
        /// </summary>
        protected override void DisposeAcceleratorObject(bool disposing)
        {
            var cudaStatus = CudaAPIExt.IpcCloseMemHandle(((CudaAccelerator)Accelerator), NativePtr);
            if (disposing)
                CudaException.ThrowIfFailed(cudaStatus);
            NativePtr = IntPtr.Zero;
        }

        #endregion
    }
}