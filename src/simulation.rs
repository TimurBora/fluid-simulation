use std::f32::consts::PI;

use cgmath::num_traits::{Pow, pow};

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Particle {
    pub position_x: f32,
    pub position_y: f32,
    pub velocity_x: f32,
    pub velocity_y: f32,
}

impl Particle {
    pub fn new(position: [f32; 2], velocity: [f32; 2]) -> Self {
        Self {
            position_x: position[0],
            position_y: position[1],
            velocity_x: velocity[0],
            velocity_y: velocity[1],
        }
    }

    pub fn desc<'a>() -> Vec<wgpu::VertexBufferLayout<'a>> {
        vec![
            wgpu::VertexBufferLayout {
                array_stride: std::mem::size_of::<f32>() as wgpu::BufferAddress,
                step_mode: wgpu::VertexStepMode::Instance,
                attributes: &wgpu::vertex_attr_array![1 => Float32],
            },
            wgpu::VertexBufferLayout {
                array_stride: std::mem::size_of::<f32>() as wgpu::BufferAddress,
                step_mode: wgpu::VertexStepMode::Instance,
                attributes: &wgpu::vertex_attr_array![2 => Float32],
            },
            wgpu::VertexBufferLayout {
                array_stride: std::mem::size_of::<f32>() as wgpu::BufferAddress,
                step_mode: wgpu::VertexStepMode::Instance,
                attributes: &wgpu::vertex_attr_array![3 => Float32],
            },
            wgpu::VertexBufferLayout {
                array_stride: std::mem::size_of::<f32>() as wgpu::BufferAddress,
                step_mode: wgpu::VertexStepMode::Instance,
                attributes: &wgpu::vertex_attr_array![4 => Float32],
            },
        ]
    }
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct SimulationParams {
    time_step: f32,
    particle_mass: f32,
    rest_density: f32,
    stiffness: f32,

    smoothing_radius: f32,
    restitution: f32,
    viscosity: f32,
    pub particles_len: u32,

    gravity_force: [f32; 2],
    _padding: [f32; 2],

    smoothing_radius_sq: f32,
    density_smoothing_function_coeff: f32,
    gradient_pressure_smoothing_function_coeff: f32,
    laplacian_viscosity_smoothing_function_coeff: f32,
}

impl SimulationParams {
    pub fn new(
        time_step: f32,
        particle_mass: f32,
        rest_density: f32,
        stiffness: f32,
        smoothing_radius: f32,
        restitution: f32,
        viscosity: f32,
        gravity_force: [f32; 2],
        particles_len: u32,
    ) -> Self {
        let smoothing_radius_sq: f32 = smoothing_radius * smoothing_radius;
        let density_smoothing_function_coeff: f32 = 4.0 / (PI * smoothing_radius.pow(8.0));
        let gradient_pressure_smoothing_function_coeff: f32 =
            -30.0 / (PI * smoothing_radius.pow(5.0));
        let laplacian_viscosity_smoothing_function_coeff: f32 =
            40.0 * (PI * smoothing_radius.pow(4.0));

        Self {
            time_step,
            particle_mass,
            rest_density,
            stiffness,
            smoothing_radius,
            restitution,
            viscosity,
            gravity_force,
            particles_len,
            _padding: [0.0; 2],
            smoothing_radius_sq,
            density_smoothing_function_coeff,
            gradient_pressure_smoothing_function_coeff,
            laplacian_viscosity_smoothing_function_coeff,
        }
    }
}
