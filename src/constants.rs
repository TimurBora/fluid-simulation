use crate::pipelines::render::Vertex;

pub const VERTICES: &[Vertex] = &[
    Vertex::const_new([-0.5, -0.5]),
    Vertex::const_new([0.5, -0.5]),
    Vertex::const_new([0.5, 0.5]),
    Vertex::const_new([-0.5, 0.5]),
];

pub const INDICES: &[u16] = &[0, 1, 2, 0, 2, 3];

pub const BACKGROUND_COLOR: [f64; 3] = [0.0, 0.0, 0.0];
