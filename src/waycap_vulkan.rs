use std::{
    ffi::{CStr, CString},
    ptr::NonNull,
};

use ash::{
    vk::{self},
    Device, Entry, Instance,
};
use libc::c_char;
use raw_window_handle::WaylandDisplayHandle;

use crate::{types::video_frame::RawVideoFrame, waycap_egl::GpuVendor};

pub struct VulkanImage {
    persistent_image: vk::Image,
    persistent_image_memory: vk::DeviceMemory,
    persistent_image_view: vk::ImageView,
    pub persistent_image_size: u64,
    pub persistent_image_memory_fd: i32,
}

pub struct VulkanContext {
    entry: Entry,
    instance: Instance,
    device: Device,
    physical_device: vk::PhysicalDevice,
    queue: vk::Queue,
    command_pool: vk::CommandPool,
    persistent_image: Option<VulkanImage>,
    gpu_vendor: GpuVendor,
}

impl VulkanContext {
    pub fn new() -> Result<Self, Box<dyn std::error::Error>> {
        let entry = unsafe { Entry::load()? };
        let instance = Self::create_instance(&entry)?;
        let physical_device = Self::select_physical_device(&instance)?;

        let gpu_vendor = Self::get_gpu_vendor(&instance, physical_device)?;
        let (device, queue) = Self::create_device(&instance, physical_device)?;
        let command_pool = Self::create_command_pool(&device, physical_device, &instance)?;

        Ok(Self {
            entry,
            instance,
            device,
            physical_device,
            queue,
            command_pool,
            persistent_image: None,
            gpu_vendor,
        })
    }

    fn create_instance(entry: &Entry) -> Result<Instance, Box<dyn std::error::Error>> {
        let wayland_display = wayland_client::Display::connect_to_env().unwrap();

        let layer_names = [c"VK_LAYER_KHRONOS_validation"];
        let layers_names_raw: Vec<*const c_char> = layer_names
            .iter()
            .map(|raw_name| raw_name.as_ptr())
            .collect();

        let raw_handle = raw_window_handle::RawDisplayHandle::Wayland(WaylandDisplayHandle::new(
            NonNull::new(wayland_display.c_ptr() as *mut std::ffi::c_void).unwrap(),
        ));

        let mut extension_names = ash_window::enumerate_required_extensions(raw_handle)
            .unwrap()
            .to_vec();

        extension_names.push(ash::khr::external_memory_capabilities::NAME.as_ptr());
        extension_names.push(ash::khr::get_physical_device_properties2::NAME.as_ptr());

        let app_name = CString::new("Waycap")?;
        let engine_name = CString::new("Waycap")?;

        let app_info = vk::ApplicationInfo::default()
            .application_name(&app_name)
            .application_version(vk::make_api_version(0, 1, 0, 0))
            .engine_name(&engine_name)
            .engine_version(vk::make_api_version(0, 1, 0, 0))
            .api_version(vk::API_VERSION_1_3);

        let create_info = vk::InstanceCreateInfo::default()
            .application_info(&app_info)
            .enabled_layer_names(&layers_names_raw)
            .enabled_extension_names(&extension_names)
            .flags(vk::InstanceCreateFlags::default());

        unsafe { Ok(entry.create_instance(&create_info, None)?) }
    }

    fn select_physical_device(
        instance: &Instance,
    ) -> Result<vk::PhysicalDevice, Box<dyn std::error::Error>> {
        let physical_devices = unsafe { instance.enumerate_physical_devices()? };

        if physical_devices.is_empty() {
            return Err("No physical devices found".into());
        }

        for &device in &physical_devices {
            let properties = unsafe { instance.get_physical_device_properties(device) };
            let device_name =
                unsafe { CStr::from_ptr(properties.device_name.as_ptr()).to_string_lossy() };

            if properties.device_type == vk::PhysicalDeviceType::DISCRETE_GPU {
                log::debug!("Utilizing GPU: {device_name:}");
                return Ok(device);
            }
        }

        let properties = unsafe { instance.get_physical_device_properties(physical_devices[0]) };
        let device_name =
            unsafe { CStr::from_ptr(properties.device_name.as_ptr()).to_string_lossy() };
        log::debug!("Defaulting to GPU: {device_name:}");
        Ok(physical_devices[0])
    }

    fn get_gpu_vendor(
        instance: &Instance,
        physical_device: vk::PhysicalDevice,
    ) -> Result<GpuVendor, Box<dyn std::error::Error>> {
        let properties = unsafe { instance.get_physical_device_properties(physical_device) };

        let vendor = match properties.vendor_id {
            0x10DE => GpuVendor::NVIDIA,
            0x1002 => GpuVendor::AMD,
            0x8086 => GpuVendor::INTEL,
            _ => GpuVendor::UNKNOWN,
        };

        Ok(vendor)
    }

    fn create_device(
        instance: &Instance,
        physical_device: vk::PhysicalDevice,
    ) -> Result<(Device, vk::Queue), Box<dyn std::error::Error>> {
        let queue_family_properties =
            unsafe { instance.get_physical_device_queue_family_properties(physical_device) };
        let queue_family_index = queue_family_properties
            .iter()
            .enumerate()
            .find(|(_, properties)| properties.queue_flags.contains(vk::QueueFlags::GRAPHICS))
            .map(|(index, _)| index as u32)
            .ok_or("No graphics queue family found")?;

        let queue_priorities = [1.0f32];
        let queue_create_info = vk::DeviceQueueCreateInfo::default()
            .queue_family_index(queue_family_index)
            .queue_priorities(&queue_priorities);

        let device_extensions = [
            ash::khr::external_memory::NAME.as_ptr(),
            ash::khr::external_memory_fd::NAME.as_ptr(),
            ash::ext::external_memory_dma_buf::NAME.as_ptr(),
            ash::ext::image_drm_format_modifier::NAME.as_ptr(),
        ];

        let device_create_info = vk::DeviceCreateInfo::default()
            .queue_create_infos(std::slice::from_ref(&queue_create_info))
            .enabled_extension_names(&device_extensions);

        let device = unsafe { instance.create_device(physical_device, &device_create_info, None)? };
        let queue = unsafe { device.get_device_queue(queue_family_index, 0) };

        Ok((device, queue))
    }

    fn create_command_pool(
        device: &Device,
        physical_device: vk::PhysicalDevice,
        instance: &Instance,
    ) -> Result<vk::CommandPool, Box<dyn std::error::Error>> {
        let queue_family_properties =
            unsafe { instance.get_physical_device_queue_family_properties(physical_device) };

        let queue_family_index = queue_family_properties
            .iter()
            .enumerate()
            .find(|(_, properties)| properties.queue_flags.contains(vk::QueueFlags::GRAPHICS))
            .map(|(index, _)| index as u32)
            .unwrap();

        let create_info = vk::CommandPoolCreateInfo::default()
            .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER)
            .queue_family_index(queue_family_index);

        let command_pool = unsafe { device.create_command_pool(&create_info, None)? };

        Ok(command_pool)
    }

    fn find_memory_type(
        type_filter: u32,
        properties: vk::MemoryPropertyFlags,
        mem_properties: &vk::PhysicalDeviceMemoryProperties,
    ) -> Result<u32, Box<dyn std::error::Error>> {
        for i in 0..mem_properties.memory_type_count {
            if (type_filter & (1 << i)) != 0
                && mem_properties.memory_types[i as usize]
                    .property_flags
                    .contains(properties)
            {
                return Ok(i);
            }
        }

        Err("Failed to find suitable memory type".into())
    }

    pub fn initalize_image(
        &mut self,
        width: u32,
        height: u32,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let mut external_memory_create_info = vk::ExternalMemoryImageCreateInfo::default()
            .handle_types(vk::ExternalMemoryHandleTypeFlags::OPAQUE_FD);

        let image_create_info = vk::ImageCreateInfo::default()
            .push_next(&mut external_memory_create_info)
            .image_type(vk::ImageType::TYPE_2D)
            .format(vk::Format::R8G8B8A8_UNORM)
            .extent(vk::Extent3D {
                width,
                height,
                depth: 1,
            })
            .mip_levels(1)
            .array_layers(1)
            .samples(vk::SampleCountFlags::TYPE_1)
            .tiling(vk::ImageTiling::OPTIMAL)
            .usage(vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::SAMPLED)
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .initial_layout(vk::ImageLayout::UNDEFINED);

        let image = unsafe { self.device.create_image(&image_create_info, None)? };

        let memory_requirements = unsafe { self.device.get_image_memory_requirements(image) };
        let memory_properties = unsafe {
            self.instance
                .get_physical_device_memory_properties(self.physical_device)
        };

        let memory_type_index = Self::find_memory_type(
            memory_requirements.memory_type_bits,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
            &memory_properties,
        )?;

        let mut export_memory_allocate_info = vk::ExportMemoryAllocateInfo::default()
            .handle_types(vk::ExternalMemoryHandleTypeFlags::OPAQUE_FD);

        let allocate_info = vk::MemoryAllocateInfo::default()
            .push_next(&mut export_memory_allocate_info)
            .allocation_size(memory_requirements.size)
            .memory_type_index(memory_type_index);

        let image_memory = unsafe { self.device.allocate_memory(&allocate_info, None)? };
        unsafe { self.device.bind_image_memory(image, image_memory, 0)? };

        let view_create_info = vk::ImageViewCreateInfo::default()
            .image(image)
            .view_type(vk::ImageViewType::TYPE_2D)
            .format(vk::Format::R8G8B8A8_UNORM)
            .components(vk::ComponentMapping::default())
            .subresource_range(vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: 1,
            });

        let image_view = unsafe { self.device.create_image_view(&view_create_info, None)? };

        let get_memory_fd_fn: vk::PFN_vkGetMemoryFdKHR = unsafe {
            let name = std::ffi::CString::new("vkGetMemoryFdKHR").unwrap();
            let func = self
                .instance
                .get_device_proc_addr(self.device.handle(), name.as_ptr());

            match func {
                Some(f) => std::mem::transmute(f),
                None => return Err("vkGetMemoryFdKHR function not available".into()),
            }
        };

        let memory_get_fd_info = vk::MemoryGetFdInfoKHR::default()
            .memory(image_memory)
            .handle_type(vk::ExternalMemoryHandleTypeFlags::OPAQUE_FD);

        let mut fd: std::ffi::c_int = -1;

        let result =
            unsafe { get_memory_fd_fn(self.device.handle(), &memory_get_fd_info, &mut fd) };

        if result != vk::Result::SUCCESS {
            return Err(format!("vkGetMemoryFdKHR failed: {:?}", result).into());
        }

        self.persistent_image = Some(VulkanImage {
            persistent_image: image,
            persistent_image_memory: image_memory,
            persistent_image_view: image_view,
            persistent_image_size: memory_requirements.size,
            persistent_image_memory_fd: fd,
        });

        log::trace!("Created persistent vulkan image with fd: {fd:}");
        Ok(())
    }

    pub fn get_image(&self) -> &Option<VulkanImage> {
        &self.persistent_image
    }

    pub fn update_image_from_dmabuf(
        &self,
        raw_image: &RawVideoFrame,
    ) -> Result<(), Box<dyn std::error::Error>> {
        assert!(raw_image.dmabuf_fd.is_some());

        let original_fd = raw_image.dmabuf_fd.unwrap();

        // Need to copy the FD so pipewire can reuse it while vulkan does its thing
        let duplicated_fd = unsafe { libc::dup(original_fd) };
        if duplicated_fd == -1 {
            return Err("Failed to duplicate DMA-BUF file descriptor".into());
        }

        unsafe {
            self.device.device_wait_idle()?;
        }

        let mut external_memory_create_info = vk::ExternalMemoryImageCreateInfo::default()
            .handle_types(vk::ExternalMemoryHandleTypeFlags::DMA_BUF_EXT);

        let plane_layout = &[vk::SubresourceLayout {
            offset: raw_image.offset as u64,
            size: 0,
            row_pitch: raw_image.stride as u64,
            array_pitch: 0,
            depth_pitch: 0,
        }];

        let mut drm_format_modifier_info =
            vk::ImageDrmFormatModifierExplicitCreateInfoEXT::default()
                .drm_format_modifier(raw_image.modifier as u64)
                .plane_layouts(plane_layout);

        let image_create_info = vk::ImageCreateInfo::default()
            .push_next(&mut external_memory_create_info)
            .push_next(&mut drm_format_modifier_info)
            .image_type(vk::ImageType::TYPE_2D)
            .format(vk::Format::B8G8R8A8_UNORM)
            .extent(vk::Extent3D {
                width: raw_image.dimensions.width,
                height: raw_image.dimensions.height,
                depth: 1,
            })
            .mip_levels(1)
            .array_layers(1)
            .samples(vk::SampleCountFlags::TYPE_1)
            .tiling(vk::ImageTiling::DRM_FORMAT_MODIFIER_EXT)
            .usage(vk::ImageUsageFlags::TRANSFER_SRC)
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .initial_layout(vk::ImageLayout::UNDEFINED);

        let source_image = match unsafe { self.device.create_image(&image_create_info, None) } {
            Ok(img) => {
                img
            }
            Err(e) => {
                return Err(e.into());
            }
        };

        let memory_requirements =
            unsafe { self.device.get_image_memory_requirements(source_image) };

        let memory_properties = unsafe {
            self.instance
                .get_physical_device_memory_properties(self.physical_device)
        };

        let memory_type_index = Self::find_memory_type(
            memory_requirements.memory_type_bits,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
            &memory_properties,
        )
        .or_else(|_| {
            log::debug!("DEVICE_LOCAL not found, trying empty flags");
            Self::find_memory_type(
                memory_requirements.memory_type_bits,
                vk::MemoryPropertyFlags::empty(),
                &memory_properties,
            )
        })
        .map_err(|e| {
            unsafe {
                self.device.destroy_image(source_image, None);
            }
            e
        })?;

        let mut import_memory_fd_info = vk::ImportMemoryFdInfoKHR::default()
            .handle_type(vk::ExternalMemoryHandleTypeFlags::DMA_BUF_EXT)
            .fd(duplicated_fd);

        let allocate_info = vk::MemoryAllocateInfo::default()
            .push_next(&mut import_memory_fd_info)
            .allocation_size(memory_requirements.size)
            .memory_type_index(memory_type_index);

        let source_memory = match unsafe { self.device.allocate_memory(&allocate_info, None) } {
            Ok(mem) => {
                mem
            }
            Err(e) => {
                unsafe {
                    self.device.destroy_image(source_image, None);
                }
                return Err(format!("Memory allocation failed: {:?}", e).into());
            }
        };

        unsafe {
            if let Err(e) = self
                .device
                .bind_image_memory(source_image, source_memory, 0)
            {
                log::error!("Failed to bind image memory: {:?}", e);
                self.device.free_memory(source_memory, None);
                self.device.destroy_image(source_image, None);
                return Err(e.into());
            }
        }

        if let Err(e) = self.copy_image_to_persistent(
            source_image,
            raw_image.dimensions.width,
            raw_image.dimensions.height,
        ) {
            unsafe {
                self.device.free_memory(source_memory, None);
                self.device.destroy_image(source_image, None);
            }
            return Err(e);
        }

        unsafe {
            self.device.free_memory(source_memory, None);
            self.device.destroy_image(source_image, None);

            self.device.device_wait_idle()?;
        }

        Ok(())
    }

    fn copy_image_to_persistent(
        &self,
        source_image: vk::Image,
        width: u32,
        height: u32,
    ) -> Result<(), Box<dyn std::error::Error>> {
        assert!(self.persistent_image.is_some());

        let allocate_info = vk::CommandBufferAllocateInfo::default()
            .command_pool(self.command_pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);

        let command_buffers = unsafe { self.device.allocate_command_buffers(&allocate_info)? };
        let command_buffer = command_buffers[0];

        let begin_info = vk::CommandBufferBeginInfo::default()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);

        unsafe {
            self.device
                .begin_command_buffer(command_buffer, &begin_info)?
        };

        let source_barrier = vk::ImageMemoryBarrier::default()
            .old_layout(vk::ImageLayout::UNDEFINED)
            .new_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .image(source_image)
            .subresource_range(vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: 1,
            })
            .src_access_mask(vk::AccessFlags::empty())
            .dst_access_mask(vk::AccessFlags::TRANSFER_READ);

        let dst_barrier = vk::ImageMemoryBarrier::default()
            .old_layout(vk::ImageLayout::UNDEFINED)
            .new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .image(self.persistent_image.as_ref().unwrap().persistent_image)
            .subresource_range(vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: 1,
            })
            .src_access_mask(vk::AccessFlags::empty())
            .dst_access_mask(vk::AccessFlags::TRANSFER_WRITE);

        unsafe {
            self.device.cmd_pipeline_barrier(
                command_buffer,
                vk::PipelineStageFlags::TOP_OF_PIPE,
                vk::PipelineStageFlags::TRANSFER,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[source_barrier, dst_barrier],
            );
        }

        let image_copy = vk::ImageCopy::default()
            .src_subresource(vk::ImageSubresourceLayers {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                mip_level: 0,
                base_array_layer: 0,
                layer_count: 1,
            })
            .src_offset(vk::Offset3D { x: 0, y: 0, z: 0 })
            .dst_subresource(vk::ImageSubresourceLayers {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                mip_level: 0,
                base_array_layer: 0,
                layer_count: 1,
            })
            .dst_offset(vk::Offset3D { x: 0, y: 0, z: 0 })
            .extent(vk::Extent3D {
                width,
                height,
                depth: 1,
            });

        unsafe {
            self.device.cmd_copy_image(
                command_buffer,
                source_image,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                self.persistent_image.as_ref().unwrap().persistent_image,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &[image_copy],
            );

            self.device.end_command_buffer(command_buffer)?;
        }

        let buffer_info = &[command_buffer];
        let submit_info = vk::SubmitInfo::default().command_buffers(buffer_info);

        unsafe {
            self.device
                .queue_submit(self.queue, &[submit_info], vk::Fence::null())?;
            self.device.queue_wait_idle(self.queue)?;
            self.device
                .free_command_buffers(self.command_pool, buffer_info);
        }

        Ok(())
    }
}

impl Drop for VulkanContext {
    fn drop(&mut self) {
        unsafe {
            self.device.device_wait_idle().unwrap();
            self.device.destroy_command_pool(self.command_pool, None);

            if let Some(image) = &self.persistent_image {
                self.device
                    .destroy_image_view(image.persistent_image_view, None);
                self.device.free_memory(image.persistent_image_memory, None);
                self.device.destroy_image(image.persistent_image, None);
            }
            self.device.destroy_device(None);
            self.instance.destroy_instance(None);
        }
    }
}
