mod app;
mod constants;
mod simulation;
mod state;

mod pipelines {
    pub mod compute;
    pub mod render;
}

fn main() -> anyhow::Result<()> {
    app::run()
}
