use std::sync::Arc;

use winit::{
    application::ApplicationHandler,
    dpi::LogicalSize,
    event::{KeyEvent, WindowEvent},
    event_loop::{ActiveEventLoop, EventLoop},
    keyboard::PhysicalKey,
    window::{WindowAttributes, WindowId},
};

use super::state;

const WINDOWS_INNER_SIZE: LogicalSize<u32> = LogicalSize::new(800, 600);

#[derive(Default)]
struct App {
    state: Option<state::State>,
}

impl App {
    fn new(_: &EventLoop<state::State>) -> Self {
        Self { state: None }
    }
}

impl ApplicationHandler<state::State> for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let attrs = WindowAttributes::default()
            .with_title("Fluid simulation")
            .with_inner_size(WINDOWS_INNER_SIZE)
            .with_visible(true);

        let window = Arc::new(event_loop.create_window(attrs).unwrap());

        self.state = Some(pollster::block_on(state::State::new(window)).unwrap());
    }

    #[allow(unused_mut)]
    fn user_event(&mut self, _event_loop: &ActiveEventLoop, mut event: state::State) {
        self.state = Some(event);
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _: WindowId, event: WindowEvent) {
        let state = match &mut self.state {
            Some(canvas) => canvas,
            None => return,
        };

        match event {
            WindowEvent::CloseRequested => {
                println!("The close button was pressed; stopping");
                event_loop.exit();
            }

            WindowEvent::Resized(size) => {
                state.resize(size);
            }

            WindowEvent::RedrawRequested => {
                state.update();
                match state.render() {
                    Ok(_) => {}
                    Err(wgpu::SurfaceError::Lost | wgpu::SurfaceError::Outdated) => {
                        let size = state.window.inner_size();
                        state.resize(size);
                    }
                    Err(e) => {
                        log::error!("Unable to render {e}");
                    }
                }
            }

            WindowEvent::KeyboardInput {
                event:
                    KeyEvent {
                        physical_key: PhysicalKey::Code(code),
                        state: key_state,
                        ..
                    },
                ..
            } => state.handle_key(event_loop, code, key_state.is_pressed()),

            WindowEvent::CursorMoved {
                position,
                device_id,
            } => state.handle_mouse_moved(position),

            _ => (),
        }
    }
}

pub fn run() -> anyhow::Result<()> {
    env_logger::init();

    let event_loop = EventLoop::with_user_event().build()?;
    let mut app = App::new(&event_loop);
    event_loop.run_app(&mut app)?;

    Ok(())
}
