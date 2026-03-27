use std::ffi::c_void;
use std::os::fd::RawFd;

use cust::sys::{CUcontext, CUresult, CUstream};
use libc::c_uint;

#[repr(C)]
pub struct AVCUDADeviceContext {
    pub cuda_ctx: CUcontext,
    pub stream: CUstream,
    pub internarl: *mut c_void,
}

// Opaque handle types for CUDA external memory interop
pub type CUexternalMemory = *mut c_void;
pub type CUmipmappedArray = *mut c_void;

/// Handle type for importing opaque FDs (exported from Vulkan)
pub const CU_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD: c_uint = 1;

/// CUDA array format: unsigned 8-bit integers
pub const CU_AD_FORMAT_UNSIGNED_INT8: c_uint = 0x01;

#[repr(C)]
pub struct CudaExternalMemoryHandleDesc {
    pub handle_type: c_uint,
    pub handle: CudaExternalMemoryHandle,
    pub size: u64,
    pub flags: c_uint,
    pub reserved: [c_uint; 16],
}

#[repr(C)]
pub union CudaExternalMemoryHandle {
    pub fd: i32,
    _padding: [u8; 64],
}

#[repr(C)]
pub struct CudaExternalMemoryMipmappedArrayDesc {
    pub offset: u64,
    pub array_desc: CudaArray3dDescriptor,
    pub num_levels: c_uint,
    pub reserved: [c_uint; 16],
}

#[repr(C)]
pub struct CudaArray3dDescriptor {
    pub width: usize,
    pub height: usize,
    pub depth: usize,
    pub format: c_uint,
    pub num_channels: c_uint,
    pub flags: c_uint,
}

unsafe extern "C" {
    pub fn cuImportExternalMemory(
        ext_mem: *mut CUexternalMemory,
        desc: *const CudaExternalMemoryHandleDesc,
    ) -> CUresult;

    pub fn cuExternalMemoryGetMappedMipmappedArray(
        mipmap: *mut CUmipmappedArray,
        ext_mem: CUexternalMemory,
        desc: *const CudaExternalMemoryMipmappedArrayDesc,
    ) -> CUresult;

    pub fn cuMipmappedArrayGetLevel(
        array: *mut cust::sys::CUarray,
        mipmap: CUmipmappedArray,
        level: c_uint,
    ) -> CUresult;

    pub fn cuDestroyExternalMemory(ext_mem: CUexternalMemory) -> CUresult;
}

/// Import a Vulkan-exported opaque FD into CUDA and return a mapped CUarray.
pub unsafe fn import_vulkan_memory_to_cuda_array(
    fd: RawFd,
    size: u64,
    width: u32,
    height: u32,
) -> std::result::Result<(CUexternalMemory, CUmipmappedArray, cust::sys::CUarray), CUresult> {
    let mut ext_mem: CUexternalMemory = std::ptr::null_mut();

    let handle_desc = CudaExternalMemoryHandleDesc {
        handle_type: CU_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD,
        handle: CudaExternalMemoryHandle { fd },
        size,
        flags: 0,
        reserved: [0; 16],
    };

    let result = cuImportExternalMemory(&mut ext_mem, &handle_desc);
    if result != CUresult::CUDA_SUCCESS {
        return Err(result);
    }

    let mut mipmap: CUmipmappedArray = std::ptr::null_mut();

    let mipmap_desc = CudaExternalMemoryMipmappedArrayDesc {
        offset: 0,
        array_desc: CudaArray3dDescriptor {
            width: width as usize,
            height: height as usize,
            depth: 0,
            format: CU_AD_FORMAT_UNSIGNED_INT8,
            num_channels: 4, // RGBA
            flags: 0,
        },
        num_levels: 1,
        reserved: [0; 16],
    };

    let result = cuExternalMemoryGetMappedMipmappedArray(&mut mipmap, ext_mem, &mipmap_desc);
    if result != CUresult::CUDA_SUCCESS {
        cuDestroyExternalMemory(ext_mem);
        return Err(result);
    }

    let mut array: cust::sys::CUarray = std::ptr::null_mut();
    let result = cuMipmappedArrayGetLevel(&mut array, mipmap, 0);
    if result != CUresult::CUDA_SUCCESS {
        cuDestroyExternalMemory(ext_mem);
        return Err(result);
    }

    Ok((ext_mem, mipmap, array))
}
