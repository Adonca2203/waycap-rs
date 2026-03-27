use std::ffi::CStr;
use std::os::fd::RawFd;

use ash::vk;

use crate::types::{
    error::{Result, WaycapError},
    video_frame::DmaBufPlane,
};

unsafe impl Sync for VulkanContext {}
unsafe impl Send for VulkanContext {}

#[derive(Clone, Copy)]
#[allow(clippy::upper_case_acronyms)]
pub enum GpuVendor {
    NVIDIA,
    AMD,
    INTEL,
    UNKNOWN,
}

const VENDOR_ID_NVIDIA: u32 = 0x10DE;
const VENDOR_ID_AMD: u32 = 0x1002;
const VENDOR_ID_INTEL: u32 = 0x8086;

pub struct VulkanContext {
    _entry: ash::Entry,
    instance: ash::Instance,
    device: ash::Device,
    physical_device: vk::PhysicalDevice,
    queue: vk::Queue,
    command_pool: vk::CommandPool,
    command_buffer: vk::CommandBuffer,
    fence: vk::Fence,

    // Persistent image for CUDA export
    persistent_image: vk::Image,
    persistent_memory: vk::DeviceMemory,
    persistent_memory_size: u64,

    // Extension loaders
    external_memory_fd: ash::khr::external_memory_fd::Device,

    gpu_vendor: GpuVendor,
    width: u32,
    height: u32,
}

impl VulkanContext {
    pub fn new(width: u32, height: u32) -> Result<Self> {
        let entry = unsafe { ash::Entry::load() }
            .map_err(|e| format!("Failed to load Vulkan loader: {e}"))?;

        let app_info = vk::ApplicationInfo::default()
            .application_name(c"waycap")
            .api_version(vk::make_api_version(0, 1, 1, 0));

        let instance_extensions = [vk::KHR_GET_PHYSICAL_DEVICE_PROPERTIES2_NAME.as_ptr()];

        let instance_ci = vk::InstanceCreateInfo::default()
            .application_info(&app_info)
            .enabled_extension_names(&instance_extensions);

        let instance = unsafe { entry.create_instance(&instance_ci, None) }
            .map_err(|e| format!("Failed to create Vulkan instance: {e}"))?;

        let physical_devices = unsafe { instance.enumerate_physical_devices() }
            .map_err(|e| format!("Failed to enumerate physical devices: {e}"))?;

        if physical_devices.is_empty() {
            return Err("No Vulkan physical devices found".into());
        }

        let physical_device = physical_devices[0];
        let props = unsafe { instance.get_physical_device_properties(physical_device) };

        let gpu_vendor = match props.vendor_id {
            VENDOR_ID_NVIDIA => GpuVendor::NVIDIA,
            VENDOR_ID_AMD => GpuVendor::AMD,
            VENDOR_ID_INTEL => GpuVendor::INTEL,
            id => {
                log::error!("Unknown GPU vendor ID: 0x{id:04X}");
                GpuVendor::UNKNOWN
            }
        };

        // Find a queue family that supports transfer
        let queue_families =
            unsafe { instance.get_physical_device_queue_family_properties(physical_device) };

        let queue_family_index = queue_families
            .iter()
            .position(|qf| qf.queue_flags.contains(vk::QueueFlags::GRAPHICS))
            .ok_or("No graphics queue family found")?
            as u32;

        let queue_priority = [1.0f32];
        let queue_ci = [vk::DeviceQueueCreateInfo::default()
            .queue_family_index(queue_family_index)
            .queue_priorities(&queue_priority)];

        let device_extensions = [
            vk::KHR_EXTERNAL_MEMORY_NAME.as_ptr(),
            vk::KHR_EXTERNAL_MEMORY_FD_NAME.as_ptr(),
            vk::EXT_EXTERNAL_MEMORY_DMA_BUF_NAME.as_ptr(),
            vk::EXT_IMAGE_DRM_FORMAT_MODIFIER_NAME.as_ptr(),
            vk::KHR_IMAGE_FORMAT_LIST_NAME.as_ptr(),
        ];

        let device_ci = vk::DeviceCreateInfo::default()
            .queue_create_infos(&queue_ci)
            .enabled_extension_names(&device_extensions);

        let device = unsafe { instance.create_device(physical_device, &device_ci, None) }
            .map_err(|e| format!("Failed to create Vulkan device: {e}"))?;

        let queue = unsafe { device.get_device_queue(queue_family_index, 0) };

        let pool_ci = vk::CommandPoolCreateInfo::default()
            .queue_family_index(queue_family_index)
            .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);

        let command_pool = unsafe { device.create_command_pool(&pool_ci, None) }
            .map_err(|e| format!("Failed to create command pool: {e}"))?;

        let alloc_ci = vk::CommandBufferAllocateInfo::default()
            .command_pool(command_pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);

        let command_buffer = unsafe { device.allocate_command_buffers(&alloc_ci) }
            .map_err(|e| format!("Failed to allocate command buffer: {e}"))?[0];

        let fence = unsafe {
            device.create_fence(&vk::FenceCreateInfo::default(), None)
        }
        .map_err(|e| format!("Failed to create fence: {e}"))?;

        let external_memory_fd = ash::khr::external_memory_fd::Device::new(&instance, &device);

        // Create persistent image for CUDA export
        let (persistent_image, persistent_memory, persistent_memory_size) =
            Self::create_exportable_image(&device, &instance, physical_device, width, height)?;

        log::debug!(
            "VulkanContext created: {:?} ({}x{}), vendor=0x{:04X}",
            unsafe { CStr::from_ptr(props.device_name.as_ptr()) },
            width,
            height,
            props.vendor_id
        );

        Ok(Self {
            _entry: entry,
            instance,
            device,
            physical_device,
            queue,
            command_pool,
            command_buffer,
            fence,
            persistent_image,
            persistent_memory,
            persistent_memory_size,
            external_memory_fd,
            gpu_vendor,
            width,
            height,
        })
    }

    fn create_exportable_image(
        device: &ash::Device,
        instance: &ash::Instance,
        physical_device: vk::PhysicalDevice,
        width: u32,
        height: u32,
    ) -> Result<(vk::Image, vk::DeviceMemory, u64)> {
        let mut external_info = vk::ExternalMemoryImageCreateInfo::default()
            .handle_types(vk::ExternalMemoryHandleTypeFlags::OPAQUE_FD);

        let image_ci = vk::ImageCreateInfo::default()
            .push_next(&mut external_info)
            .image_type(vk::ImageType::TYPE_2D)
            .format(vk::Format::R8G8B8A8_UNORM)
            .extent(vk::Extent3D { width, height, depth: 1 })
            .mip_levels(1)
            .array_layers(1)
            .samples(vk::SampleCountFlags::TYPE_1)
            .tiling(vk::ImageTiling::LINEAR)
            .usage(vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::TRANSFER_SRC)
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .initial_layout(vk::ImageLayout::UNDEFINED);

        let image = unsafe { device.create_image(&image_ci, None) }
            .map_err(|e| format!("Failed to create persistent image: {e}"))?;

        let mem_reqs = unsafe { device.get_image_memory_requirements(image) };
        let mem_props = unsafe { instance.get_physical_device_memory_properties(physical_device) };

        let mem_type_index = find_memory_type(
            &mem_props,
            mem_reqs.memory_type_bits,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        )
        .ok_or("No suitable memory type for persistent image")?;

        let mut export_info = vk::ExportMemoryAllocateInfo::default()
            .handle_types(vk::ExternalMemoryHandleTypeFlags::OPAQUE_FD);

        let alloc_info = vk::MemoryAllocateInfo::default()
            .push_next(&mut export_info)
            .allocation_size(mem_reqs.size)
            .memory_type_index(mem_type_index);

        let memory = unsafe { device.allocate_memory(&alloc_info, None) }
            .map_err(|e| format!("Failed to allocate persistent memory: {e}"))?;

        unsafe { device.bind_image_memory(image, memory, 0) }
            .map_err(|e| format!("Failed to bind persistent image memory: {e}"))?;

        Ok((image, memory, mem_reqs.size))
    }

    /// Export the persistent image's memory as an opaque FD for CUDA import.
    /// Note: the FD can only be exported once — Vulkan transfers ownership.
    pub fn export_persistent_memory_fd(&self) -> Result<(RawFd, u64)> {
        let fd_info = vk::MemoryGetFdInfoKHR::default()
            .memory(self.persistent_memory)
            .handle_type(vk::ExternalMemoryHandleTypeFlags::OPAQUE_FD);

        let fd = unsafe { self.external_memory_fd.get_memory_fd(&fd_info) }
            .map_err(|e| format!("Failed to export memory FD: {e}"))?;

        Ok((fd, self.persistent_memory_size))
    }

    /// Import a DMA-BUF frame and copy its contents to the persistent image.
    pub fn import_dmabuf_and_copy(
        &self,
        planes: &[DmaBufPlane],
        _format: u32,
        width: u32,
        height: u32,
        modifier: u64,
    ) -> Result<()> {
        let plane = &planes[0];

        // Create temp image backed by the DMA-BUF
        let drm_modifier = [modifier];
        let mut drm_list = vk::ImageDrmFormatModifierListCreateInfoEXT::default()
            .drm_format_modifiers(&drm_modifier);

        let format_list_view = [vk::Format::B8G8R8A8_UNORM];
        let mut format_list = vk::ImageFormatListCreateInfo::default()
            .view_formats(&format_list_view);

        let mut external_info = vk::ExternalMemoryImageCreateInfo::default()
            .handle_types(vk::ExternalMemoryHandleTypeFlags::DMA_BUF_EXT);

        let image_ci = vk::ImageCreateInfo::default()
            .push_next(&mut external_info)
            .push_next(&mut format_list)
            .push_next(&mut drm_list)
            .image_type(vk::ImageType::TYPE_2D)
            .format(vk::Format::B8G8R8A8_UNORM) // DRM ARGB8888 = BGRA in memory
            .extent(vk::Extent3D { width, height, depth: 1 })
            .mip_levels(1)
            .array_layers(1)
            .samples(vk::SampleCountFlags::TYPE_1)
            .tiling(vk::ImageTiling::DRM_FORMAT_MODIFIER_EXT)
            .usage(vk::ImageUsageFlags::TRANSFER_SRC)
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .initial_layout(vk::ImageLayout::UNDEFINED);

        let temp_image = unsafe { self.device.create_image(&image_ci, None) }
            .map_err(|e| format!("Failed to create temp DMA-BUF image: {e}"))?;

        let result = self.import_and_copy_inner(temp_image, plane, width, height);

        unsafe { self.device.destroy_image(temp_image, None) };

        result
    }

    fn import_and_copy_inner(
        &self,
        temp_image: vk::Image,
        plane: &DmaBufPlane,
        width: u32,
        height: u32,
    ) -> Result<()> {
        let mem_reqs = unsafe { self.device.get_image_memory_requirements(temp_image) };
        let mem_props = unsafe {
            self.instance
                .get_physical_device_memory_properties(self.physical_device)
        };

        let mem_type_index = find_memory_type(
            &mem_props,
            mem_reqs.memory_type_bits,
            vk::MemoryPropertyFlags::empty(),
        )
        .ok_or("No suitable memory type for DMA-BUF import")?;

        let mut import_fd_info = vk::ImportMemoryFdInfoKHR::default()
            .handle_type(vk::ExternalMemoryHandleTypeFlags::DMA_BUF_EXT)
            .fd(plane.fd);

        let alloc_info = vk::MemoryAllocateInfo::default()
            .push_next(&mut import_fd_info)
            .allocation_size(mem_reqs.size)
            .memory_type_index(mem_type_index);

        let temp_memory = unsafe { self.device.allocate_memory(&alloc_info, None) }
            .map_err(|e| format!("Failed to import DMA-BUF memory: {e}"))?;

        let bind_result = unsafe { self.device.bind_image_memory(temp_image, temp_memory, 0) };
        if let Err(e) = bind_result {
            unsafe { self.device.free_memory(temp_memory, None) };
            return Err(format!("Failed to bind DMA-BUF image memory: {e}").into());
        }

        let copy_result = self.record_and_submit_copy(temp_image, width, height);

        // Vulkan took ownership of the FD on successful import, only free memory object
        unsafe { self.device.free_memory(temp_memory, None) };

        copy_result
    }

    fn record_and_submit_copy(
        &self,
        src_image: vk::Image,
        width: u32,
        height: u32,
    ) -> Result<()> {
        let cb = self.command_buffer;

        unsafe {
            self.device
                .reset_command_buffer(cb, vk::CommandBufferResetFlags::empty())
                .map_err(|e| format!("Failed to reset command buffer: {e}"))?;

            self.device
                .begin_command_buffer(cb, &vk::CommandBufferBeginInfo::default()
                    .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT))
                .map_err(|e| format!("Failed to begin command buffer: {e}"))?;

            // Transition src (DMA-BUF) → TRANSFER_SRC_OPTIMAL
            let src_barrier = vk::ImageMemoryBarrier::default()
                .image(src_image)
                .old_layout(vk::ImageLayout::UNDEFINED)
                .new_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
                .src_access_mask(vk::AccessFlags::empty())
                .dst_access_mask(vk::AccessFlags::TRANSFER_READ)
                .subresource_range(full_subresource_range());

            // Transition dst (persistent) → TRANSFER_DST_OPTIMAL
            let dst_barrier = vk::ImageMemoryBarrier::default()
                .image(self.persistent_image)
                .old_layout(vk::ImageLayout::UNDEFINED)
                .new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                .src_access_mask(vk::AccessFlags::empty())
                .dst_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                .subresource_range(full_subresource_range());

            self.device.cmd_pipeline_barrier(
                cb,
                vk::PipelineStageFlags::TOP_OF_PIPE,
                vk::PipelineStageFlags::TRANSFER,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[src_barrier, dst_barrier],
            );

            let region = vk::ImageCopy {
                src_subresource: vk::ImageSubresourceLayers {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    mip_level: 0,
                    base_array_layer: 0,
                    layer_count: 1,
                },
                src_offset: vk::Offset3D::default(),
                dst_subresource: vk::ImageSubresourceLayers {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    mip_level: 0,
                    base_array_layer: 0,
                    layer_count: 1,
                },
                dst_offset: vk::Offset3D::default(),
                extent: vk::Extent3D { width, height, depth: 1 },
            };

            self.device.cmd_copy_image(
                cb,
                src_image,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                self.persistent_image,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &[region],
            );

            // Transition persistent → GENERAL so CUDA can read
            let final_barrier = vk::ImageMemoryBarrier::default()
                .image(self.persistent_image)
                .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                .new_layout(vk::ImageLayout::GENERAL)
                .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                .dst_access_mask(vk::AccessFlags::MEMORY_READ)
                .subresource_range(full_subresource_range());

            self.device.cmd_pipeline_barrier(
                cb,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::ALL_COMMANDS,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[final_barrier],
            );

            self.device
                .end_command_buffer(cb)
                .map_err(|e| format!("Failed to end command buffer: {e}"))?;

            self.device
                .reset_fences(&[self.fence])
                .map_err(|e| format!("Failed to reset fence: {e}"))?;

            let submit = [vk::SubmitInfo::default().command_buffers(&[cb])];
            self.device
                .queue_submit(self.queue, &submit, self.fence)
                .map_err(|e| format!("Failed to submit copy command: {e}"))?;

            self.device
                .wait_for_fences(&[self.fence], true, u64::MAX)
                .map_err(|e| format!("Failed to wait for copy fence: {e}"))?;
        }

        Ok(())
    }

    pub fn get_gpu_vendor(&self) -> GpuVendor {
        self.gpu_vendor
    }
}

impl Drop for VulkanContext {
    fn drop(&mut self) {
        unsafe {
            let _ = self.device.device_wait_idle();
            self.device.destroy_fence(self.fence, None);
            self.device.destroy_command_pool(self.command_pool, None);
            self.device.destroy_image(self.persistent_image, None);
            self.device.free_memory(self.persistent_memory, None);
            self.device.destroy_device(None);
            self.instance.destroy_instance(None);
        }
    }
}

fn full_subresource_range() -> vk::ImageSubresourceRange {
    vk::ImageSubresourceRange {
        aspect_mask: vk::ImageAspectFlags::COLOR,
        base_mip_level: 0,
        level_count: 1,
        base_array_layer: 0,
        layer_count: 1,
    }
}

fn find_memory_type(
    mem_props: &vk::PhysicalDeviceMemoryProperties,
    type_bits: u32,
    required: vk::MemoryPropertyFlags,
) -> Option<u32> {
    (0..mem_props.memory_type_count).find(|&i| {
        (type_bits & (1 << i)) != 0
            && mem_props.memory_types[i as usize]
                .property_flags
                .contains(required)
    })
}

/// Standalone GPU vendor detection without creating a full context.
pub fn detect_gpu_vendor() -> Result<GpuVendor> {
    let entry = unsafe { ash::Entry::load() }
        .map_err(|e| format!("Failed to load Vulkan loader: {e}"))?;

    let app_info = vk::ApplicationInfo::default()
        .application_name(c"waycap-detect")
        .api_version(vk::make_api_version(0, 1, 0, 0));

    let instance_ci = vk::InstanceCreateInfo::default().application_info(&app_info);

    let instance = unsafe { entry.create_instance(&instance_ci, None) }
        .map_err(|e| format!("Failed to create Vulkan instance: {e}"))?;

    let devices = unsafe { instance.enumerate_physical_devices() }
        .map_err(|e| format!("Failed to enumerate devices: {e}"))?;

    let vendor = if devices.is_empty() {
        GpuVendor::UNKNOWN
    } else {
        let props = unsafe { instance.get_physical_device_properties(devices[0]) };
        match props.vendor_id {
            VENDOR_ID_NVIDIA => GpuVendor::NVIDIA,
            VENDOR_ID_AMD => GpuVendor::AMD,
            VENDOR_ID_INTEL => GpuVendor::INTEL,
            _ => GpuVendor::UNKNOWN,
        }
    };

    unsafe { instance.destroy_instance(None) };
    Ok(vendor)
}
