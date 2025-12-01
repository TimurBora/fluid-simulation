struct VertexInput {
    @location(0) position: vec2<f32>,
}

struct ParticleInput {
    @location(1) position_x: f32,
    @location(2) position_y: f32,
    @location(3) velocity_x: f32,
    @location(4) velocity_y: f32,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) color: vec3<f32>,
    @location(1) local_position: vec2<f32>,
}

const PARTICLE_SCALE: f32 = 1.0 / 15.0;

@vertex
fn vs_main(
  model: VertexInput,
  particle: ParticleInput,
) -> VertexOutput {
    var output: VertexOutput;

    let velocity_length = length(vec2<f32>(particle.velocity_x, particle.velocity_y));

    let t = smoothstep(0.5, 3.0, velocity_length);

    let mixed = mix(
        vec3<f32>(0.0, 0.0, 1.0), 
        vec3<f32>(1.0, 0.0, 0.0),
        t
    );

    output.color = vec3<f32>(mixed);

    output.local_position = model.position;

    let scaled = model.position * PARTICLE_SCALE;
    let moved = scaled + vec2<f32>(particle.position_x, particle.position_y);
output.clip_position = vec4<f32>(moved, 0.0, 1.0); 

    return output;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let radius = 0.5;

    let r_length_sq = in.local_position.x * in.local_position.x + in.local_position.y * in.local_position.y;
    if r_length_sq > radius * radius {
        discard;
    }

    return vec4<f32>(in.color, 1.0);
}

