use std::sync::Arc;

use winit::window::Window;

use crate::constants::BACKGROUND_COLOR;
use crate::pipelines::compute::ComputePipelineState;
use crate::pipelines::render::RenderPipelineState;
use crate::simulation::{Particle, SimulationParams};

pub struct State {
    pub window: Arc<Window>,
    surface: wgpu::Surface<'static>,
    is_surface_configured: bool,
    render_pipeline_state: RenderPipelineState,
    compute_pipeline_state: ComputePipelineState,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    particles: Vec<Particle>,
    simulation_params: SimulationParams,
}

impl State {
    pub async fn new(window: Arc<Window>) -> anyhow::Result<State> {
        let size = window.inner_size();
        let instance = wgpu::Instance::default();
        let surface = instance.create_surface(Arc::clone(&window))?;
        let adapter = Self::init_adapter(&instance, &surface).await?;
        let (device, queue) = Self::init_device(&adapter).await?;
        let config = surface
            .get_default_config(&adapter, size.width, size.height)
            .unwrap();
        surface.configure(&device, &config);

        let render_pipeline_state = RenderPipelineState::new(&device, &config);

        let simulation_params = SimulationParams::new(
            1.0 / 60.0,        // time_step — увеличиваем шаг для большей текучести
            10.0,              // particle_mass — уменьшаем массу для легкости
            5000.0,            // rest_density — стандартная плотность воды
            0.8,               // stiffness — увеличиваем жесткость для лучшего сохранения формы
            0.2,               // smoothing_radius — увеличиваем радиус взаимодействия
            0.1,               // restitution — уменьшаем отскок для вязкости воды
            20.5,              // viscosity — значительно уменьшаем вязкость для текучести
            [0.0, -100_000.0], // gravity_force — реальное ускорение свободного падения
            10_000,            // particles_len
        );

        let particles_len = simulation_params.particles_len as usize;
        let grid_size = (particles_len as f32).sqrt().ceil() as usize;

        let spacing = 1.0 / grid_size as f32;
        let start = -0.5 + spacing / 2.0;

        let mut particles = Vec::new();
        for i in 0..grid_size {
            for j in 0..grid_size {
                if particles.len() >= particles_len {
                    break;
                }

                let x = start + i as f32 * spacing;
                let y = start + j as f32 * spacing;

                particles.push(Particle::new(
                    [x, y],
                    [rand::random::<f32>() * 0.1 - 0.05, -0.05],
                ));
            }
        }

        let compute_pipeline_state =
            ComputePipelineState::new(&device, &particles, &simulation_params);

        Ok(Self {
            window,
            surface,
            render_pipeline_state,
            compute_pipeline_state,
            device,
            queue,
            config,
            is_surface_configured: false,
            particles,
            simulation_params,
        })
    }

    async fn init_adapter(
        instance: &wgpu::Instance,
        surface: &wgpu::Surface<'_>,
    ) -> anyhow::Result<wgpu::Adapter> {
        Ok(instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                force_fallback_adapter: false,
                compatible_surface: Some(surface),
            })
            .await
            .expect("No adapter found"))
    }

    async fn init_device(adapter: &wgpu::Adapter) -> anyhow::Result<(wgpu::Device, wgpu::Queue)> {
        Ok(adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: None,
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
                memory_hints: wgpu::MemoryHints::MemoryUsage,
                trace: wgpu::Trace::Off,
            })
            .await?)
    }

    pub fn resize(&mut self, new_size: winit::dpi::PhysicalSize<u32>) {
        if new_size.width > 0 && new_size.height > 0 {
            self.config.width = new_size.width;
            self.config.height = new_size.height;

            self.surface.configure(&self.device, &self.config);
            self.is_surface_configured = true;
        }
    }

    pub fn handle_key(
        &mut self,
        event_loop: &winit::event_loop::ActiveEventLoop,
        code: winit::keyboard::KeyCode,
        is_pressed: bool,
    ) {
        match (code, is_pressed) {
            (winit::keyboard::KeyCode::Escape, true) => event_loop.exit(),
            (winit::keyboard::KeyCode::Enter, true) => event_loop.exit(),
            _ => {}
        }
    }

    pub fn handle_mouse_moved(&mut self, _position: winit::dpi::PhysicalPosition<f64>) {
        let size = self.window.inner_size();

        if size.width > 0 && size.height > 0 {}
    }

    pub fn render(&mut self) -> anyhow::Result<(), wgpu::SurfaceError> {
        self.window.request_redraw();

        if !self.is_surface_configured {
            return Ok(());
        }

        let output = self.surface.get_current_texture()?;

        let view = output
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Render Encoder"),
            });

        let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("Render Pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color {
                        r: BACKGROUND_COLOR[0],
                        g: BACKGROUND_COLOR[1],
                        b: BACKGROUND_COLOR[2],
                        a: 1.0,
                    }),
                    store: wgpu::StoreOp::Store,
                },
                depth_slice: None,
            })],
            depth_stencil_attachment: None,
            occlusion_query_set: None,
            timestamp_writes: None,
        });

        render_pass.set_pipeline(&self.render_pipeline_state.render_pipeline);

        render_pass.set_vertex_buffer(0, self.render_pipeline_state.vertex_buffer.slice(..));
        render_pass.set_vertex_buffer(1, self.compute_pipeline_state.position_x_buffer.slice(..));
        render_pass.set_vertex_buffer(2, self.compute_pipeline_state.position_y_buffer.slice(..));
        render_pass.set_vertex_buffer(3, self.compute_pipeline_state.velocity_x_buffer.slice(..));
        render_pass.set_vertex_buffer(4, self.compute_pipeline_state.velocity_y_buffer.slice(..));

        render_pass.set_index_buffer(
            self.render_pipeline_state.index_buffer.slice(..),
            wgpu::IndexFormat::Uint16,
        );

        render_pass.draw_indexed(
            0..self.render_pipeline_state.num_indices,
            0,
            0..self.particles.len() as _,
        );

        drop(render_pass);

        self.queue.submit(std::iter::once(encoder.finish()));
        output.present();

        Ok(())
    }

    pub fn update(&mut self) {
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Render Encoder"),
            });

        let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Compute Pass Densities"),
            timestamp_writes: None,
        });

        compute_pass.set_pipeline(&self.compute_pipeline_state.compute_densities_pipeline);
        compute_pass.set_bind_group(0, &self.compute_pipeline_state.compute_bind_group_0, &[]);
        compute_pass.set_bind_group(1, &self.compute_pipeline_state.compute_bind_group_1, &[]);
        compute_pass.set_bind_group(2, &self.compute_pipeline_state.compute_bind_group_2, &[]);

        compute_pass.dispatch_workgroups((self.particles.len() as u32 + 63).div_ceil(64), 1, 1);
        compute_pass.set_pipeline(&self.compute_pipeline_state.compute_pressures_pipeline);

        compute_pass.dispatch_workgroups((self.particles.len() as u32 + 63).div_ceil(64), 1, 1);

        compute_pass.set_pipeline(&self.compute_pipeline_state.compute_new_positions_pipeline);

        compute_pass.dispatch_workgroups((self.particles.len() as u32 + 63).div_ceil(64), 1, 1);
        drop(compute_pass);

        self.queue.submit(std::iter::once(encoder.finish()));
    }
}
