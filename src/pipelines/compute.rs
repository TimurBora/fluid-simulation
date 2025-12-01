use crate::simulation::{Particle, SimulationParams};
use wgpu::util::DeviceExt;

pub struct ComputePipelineState {
    pub compute_densities_pipeline: wgpu::ComputePipeline,
    pub compute_pressures_pipeline: wgpu::ComputePipeline,
    pub compute_new_positions_pipeline: wgpu::ComputePipeline,

    pub position_x_buffer: wgpu::Buffer,
    pub position_y_buffer: wgpu::Buffer,
    pub velocity_x_buffer: wgpu::Buffer,
    pub velocity_y_buffer: wgpu::Buffer,

    pub densities_buffer: wgpu::Buffer,
    pub pressures_buffer: wgpu::Buffer,
    pub simulation_params_buffer: wgpu::Buffer,

    pub compute_bind_group_0: wgpu::BindGroup,
    pub compute_bind_group_1: wgpu::BindGroup,
    pub compute_bind_group_2: wgpu::BindGroup,
}

impl ComputePipelineState {
    pub fn new(
        device: &wgpu::Device,
        particles: &[Particle],
        simulation_params: &SimulationParams,
    ) -> Self {
        let compute_shader =
            device.create_shader_module(wgpu::include_wgsl!("../shaders/physics.wgsl"));

        let position_x: Vec<f32> = particles.iter().map(|p| p.position_x).collect();
        let position_y: Vec<f32> = particles.iter().map(|p| p.position_y).collect();
        let velocity_x: Vec<f32> = particles.iter().map(|p| p.velocity_x).collect();
        let velocity_y: Vec<f32> = particles.iter().map(|p| p.velocity_y).collect();

        let position_x_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Position X Buffer"),
            contents: bytemuck::cast_slice(&position_x),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::VERTEX,
        });

        let position_y_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Position Y Buffer"),
            contents: bytemuck::cast_slice(&position_y),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::VERTEX,
        });

        let velocity_x_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Velocity X Buffer"),
            contents: bytemuck::cast_slice(&velocity_x),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::VERTEX,
        });

        let velocity_y_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Velocity Y Buffer"),
            contents: bytemuck::cast_slice(&velocity_y),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::VERTEX,
        });

        let densities_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Densities Buffer"),
            contents: bytemuck::cast_slice(&vec![0.0; simulation_params.particles_len as usize]),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });

        let pressures_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Pressures Buffer"),
            contents: bytemuck::cast_slice(&vec![0.0; simulation_params.particles_len as usize]),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });

        let simulation_params_buffer =
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Simulation Params Buffer"),
                contents: bytemuck::cast_slice(&[*simulation_params]),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });

        let compute_bind_group_layout_0 =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Compute Bind Group Layout 0"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });

        let compute_bind_group_layout_1 =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Compute Bind Group Layout 1"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });

        let compute_bind_group_layout_2 =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Compute Bind Group Layout 2"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
            });

        let compute_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Compute Pipeline Layout"),
                bind_group_layouts: &[
                    &compute_bind_group_layout_0,
                    &compute_bind_group_layout_1,
                    &compute_bind_group_layout_2,
                ],
                push_constant_ranges: &[],
            });

        let compute_densities_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Physics Compute Densities Pipeline"),
                layout: Some(&compute_pipeline_layout),
                module: &compute_shader,
                entry_point: Some("compute_densities"),
                compilation_options: Default::default(),
                cache: Default::default(),
            });

        let compute_pressures_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Physics Compute Pressures Pipeline"),
                layout: Some(&compute_pipeline_layout),
                module: &compute_shader,
                entry_point: Some("compute_pressures"),
                compilation_options: Default::default(),
                cache: Default::default(),
            });

        let compute_new_positions_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Physics Compute Main Pipeline"),
                layout: Some(&compute_pipeline_layout),
                module: &compute_shader,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: Default::default(),
            });

        let compute_bind_group_0 = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Bind Group 0"),
            layout: &compute_bind_group_layout_0,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: position_x_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: position_y_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: velocity_x_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: velocity_y_buffer.as_entire_binding(),
                },
            ],
        });

        let compute_bind_group_1 = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Bind Group 1"),
            layout: &compute_bind_group_layout_1,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: densities_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: pressures_buffer.as_entire_binding(),
                },
            ],
        });

        let compute_bind_group_2 = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Bind Group 2"),
            layout: &compute_bind_group_layout_2,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: simulation_params_buffer.as_entire_binding(),
            }],
        });

        Self {
            compute_densities_pipeline,
            compute_pressures_pipeline,
            compute_new_positions_pipeline,

            compute_bind_group_0,
            compute_bind_group_1,
            compute_bind_group_2,

            simulation_params_buffer,

            position_x_buffer,
            position_y_buffer,
            velocity_x_buffer,
            velocity_y_buffer,

            densities_buffer,
            pressures_buffer,
        }
    }
}
