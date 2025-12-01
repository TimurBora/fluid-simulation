
const pi_value: f32 = 3.14159;

struct SimulationParams {
    time_step: f32, // 4 => 4
    particle_mass: f32, // 4 => 8
    rest_density: f32, // 4 => 12
    stiffness: f32, // 4 => 16
    smoothing_radius: f32, // 4 => 4
    restitution: f32, // 4 => 8
    viscosity: f32, // 4 => 12
    particles_len: u32, // 4 => 16
    gravity_force: vec2<f32>, // 8 => 8
    _padding: vec2<f32>, // 8 => 16
    smoothing_radius_sq: f32,
    density_smoothing_function_coeff: f32,
    gradient_pressure_smoothing_function_coeff: f32,
    laplacian_viscosity_smoothing_function_coeff: f32,
};

@group(0) @binding(0) var<storage, read_write> position_x: array<f32>;
@group(0) @binding(1) var<storage, read_write> position_y: array<f32>;
@group(0) @binding(2) var<storage, read_write> velocity_x: array<f32>;
@group(0) @binding(3) var<storage, read_write> velocity_y: array<f32>;

@group(0) @binding(4) var<storage, read_write> hashes: array<u32>;
@group(0) @binding(5) var<storage, read_write> indices: array<u32>;
@group(0) @binding(6) var<storage, read_write> block_sums: array<u32>;

@group(0) @binding(7) var<storage, read_write> prefix_sum_output: array<u32>;
@group(0) @binding(8) var<storage, read_write> sums_output: array<u32>;

@group(0) @binding(9) var<storage, read_write> lsb: array<u32>;
@group(0) @binding(11) var<storage, read_write> predicate_scan: array<u32>;

@group(1) @binding(0) var<storage, read_write> densities: array<f32>;
@group(1) @binding(1) var<storage, read_write> pressures: array<f32>;

@group(2) @binding(0) var<uniform> simulation_params: SimulationParams;

fn density_smoothing_function(r_x: f32, r_y: f32) -> f32 {
    let h = simulation_params.smoothing_radius;
    let r_length_sq = r_x * r_x + r_y * r_y;

    if r_length_sq > simulation_params.smoothing_radius_sq {
        return 0.0;
    }

    let r_length = sqrt(r_length_sq);
    let h_minus_r = simulation_params.smoothing_radius_sq - r_length * r_length;

    return simulation_params.density_smoothing_function_coeff * h_minus_r * h_minus_r * h_minus_r;
}

fn calculate_density(i: u32) -> f32 {
    var density: f32 = 0.0;

    for (var j: u32 = 0u; j < simulation_params.particles_len; j++) {
        density += density_smoothing_function(position_x[i] - position_x[j], position_y[i] - position_y[j]);
    }

    return simulation_params.particle_mass * density;
}

fn gradient_pressure_smoothing_function(r_x: f32, r_y: f32) -> vec2<f32> {
    let r_length_sq = r_x * r_x + r_y * r_y;

    if r_length_sq > simulation_params.smoothing_radius_sq || r_length_sq < 1.0e-8 {
        return vec2<f32>(0.0, 0.0);
    }

    let r_length = sqrt(r_length_sq);
    let h_minus_r = simulation_params.smoothing_radius - r_length;
    let coeff = simulation_params.gradient_pressure_smoothing_function_coeff * pow(h_minus_r, 2.0) / r_length ;

    return vec2<f32>(coeff * r_x, coeff * r_y);
}

fn calculate_pressure(i: u32) -> f32 {
    return simulation_params.stiffness * (densities[i] - simulation_params.rest_density);
}

fn calculate_pressure_force(i: u32) -> vec2<f32> {
    var pressure_force = vec2<f32>(0.0, 0.0);

    for (var j: u32 = 0u; j < simulation_params.particles_len; j++) {
        if densities[j] < 0.0001 || i == j { 
            continue;
        }

        pressure_force -= (pressures[i] + pressures[j]) / (2f * densities[j]) * gradient_pressure_smoothing_function(position_x[i] - position_x[j], position_y[i] - position_y[j]);
    }

    return simulation_params.particle_mass * pressure_force;
}

fn laplacian_viscosity_smoothing_function(r_length: f32) -> f32 {
    let h = simulation_params.smoothing_radius;

    if r_length < 0.0001 || r_length > h {
        return 0.0;
    }

    return simulation_params.laplacian_viscosity_smoothing_function_coeff * (h - r_length);
}

fn calculate_viscosity_force(i: u32) -> vec2<f32> {
    var viscosity_force = vec2<f32>(0.0, 0.0);

    for (var j: u32 = 0u; j < simulation_params.particles_len; j++) {
        let r_x = position_x[i] - position_x[j];
        let r_y = position_y[i] - position_y[j];
        let r_length_sq = r_x * r_x + r_y * r_y;

        if i == j { continue; }
        if densities[j] < 0.0001 || r_length_sq < 1.0e-8 {
            continue;
        }

        let r_length = sqrt(r_length_sq);
        let viscosity_velocity = vec2<f32>(velocity_x[j] - velocity_x[i], velocity_y[j] - velocity_y[i]);

        viscosity_force += (simulation_params.particle_mass * viscosity_velocity / densities[j]) * laplacian_viscosity_smoothing_function(r_length);
    }

    return simulation_params.viscosity * viscosity_force;
}


    @compute
    @workgroup_size(64)
fn compute_densities(
    @builtin(global_invocation_id) global_invocation_id: vec3<u32>
) {
    let i = global_invocation_id.x;

    if i >= simulation_params.particles_len {
        return;
    }

    densities[i] = calculate_density(i);
}

    @compute
    @workgroup_size(64)
fn compute_pressures(
    @builtin(global_invocation_id) global_invocation_id: vec3<u32>
) {
    let i = global_invocation_id.x;

    if i >= simulation_params.particles_len {
        return;
    }

    pressures[i] = calculate_pressure(i);
}


fn get_cell_coordinates(position_x: f32, position_y: f32) -> vec2<u32> {
    let cell = vec2<u32>(
        u32(ceil(position_x / simulation_params.smoothing_radius)),
        u32(ceil(position_y / simulation_params.smoothing_radius))
    );

    return cell;
}

fn hash_cell(cell_x: u32, cell_y: u32) -> u32 {
    let x = cell_x * 15823u;
    let y = cell_y * 9737333u;

    return x + y;
}

@compute
@workgroup_size(256)
fn calculate_hashes(
    @builtin(global_invocation_id) global_invocation_id: vec3<u32>
) {
    let i = global_invocation_id.x;

    if i >= simulation_params.particles_len {
        return;
    }

    let cell_coordinate = get_cell_coordinates(position_x[i], position_y[i]);
    hashes[i] = hash_cell(cell_coordinate.x, cell_coordinate.y) / simulation_params.particles_len;
    indices[i] = i;
}

const bank_size:u32 = 32u;
fn bank_conflict_free_idx(idx: u32) -> u32 {
    var chunk_id: u32 = idx / bank_size;
    return idx + chunk_id;
}

const n: u32 = 512u;
var<workgroup> temp_array: array<u32, 532>;
@compute @workgroup_size(256)
fn compute_local_prefix_sum(@builtin(global_invocation_id) global_invocation_id: vec3<u32>,
    @builtin(local_invocation_id) local_invocation_id: vec3<u32>,
    @builtin(workgroup_id) workgroup_id: vec3<u32>) {

    var thread_id: u32 = local_invocation_id.x;

    var global_thread_id: u32 = global_invocation_id.x;

    if thread_id < (n >> 1u) {
        temp_array[bank_conflict_free_idx(2u * thread_id)] = hashes[2u * global_thread_id];
        temp_array[bank_conflict_free_idx(2u * thread_id + 1u)] = hashes[2u * global_thread_id + 1u];
    }

    workgroupBarrier();

    var offset: u32 = 1u;
    for (var d: u32 = n >> 1u; d > 0u; d >>= 1u) {
        if thread_id < d {
            var ai: u32 = offset * (2u * thread_id + 1u) - 1u;
            var bi: u32 = offset * (2u * thread_id + 2u) - 1u;
            temp_array[bank_conflict_free_idx(bi)] += temp_array[bank_conflict_free_idx(ai)];
        }

        offset *= 2u;
    }

    workgroupBarrier();

    if thread_id == 0u {
        block_sums[workgroup_id.x] = temp_array[bank_conflict_free_idx(n - 1u)];
        temp_array[bank_conflict_free_idx(n - 1u)] = 0u;
    }

    workgroupBarrier();

    for (var d: u32 = 1u; d < n; d *= 2u) {
        offset >>= 1u;
        if thread_id < d {
            var ai: u32 = offset * (2u * thread_id + 1u) - 1u;
            var bi: u32 = offset * (2u * thread_id + 2u) - 1u;
            var temp: u32 = temp_array[bank_conflict_free_idx(ai)];
            temp_array[bank_conflict_free_idx(ai)] = temp_array[bank_conflict_free_idx(bi)];
            temp_array[bank_conflict_free_idx(bi)] += temp;
        }
        workgroupBarrier();
    }

    if thread_id < (n >> 1u) {
        prefix_sum_output[2u * global_thread_id] = temp_array[bank_conflict_free_idx(2u * thread_id)];
        prefix_sum_output[2u * global_thread_id + 1u] = temp_array[bank_conflict_free_idx(2u * thread_id + 1u)];
    }
}

@compute @workgroup_size(256)
fn compute_block_sums(@builtin(global_invocation_id) global_invocation_id: vec3<u32>,
    @builtin(local_invocation_id) local_invocation_id: vec3<u32>,
    @builtin(workgroup_id) workgroup_id: vec3<u32>) {

    var thread_id: u32 = local_invocation_id.x;

    var global_thread_id: u32 = global_invocation_id.x;

    if thread_id < (n >> 1u) {
        temp_array[bank_conflict_free_idx(2u * thread_id)] = block_sums[2u * global_thread_id];
        temp_array[bank_conflict_free_idx(2u * thread_id + 1u)] = block_sums[2u * global_thread_id + 1u];
    }

    workgroupBarrier();

    var offset: u32 = 1u;
    for (var d: u32 = n >> 1u; d > 0u; d >>= 1u) {
        if thread_id < d {
            var ai: u32 = offset * (2u * thread_id + 1u) - 1u;
            var bi: u32 = offset * (2u * thread_id + 2u) - 1u;
            temp_array[bank_conflict_free_idx(bi)] += temp_array[bank_conflict_free_idx(ai)];
        }

        offset *= 2u;
    }

    workgroupBarrier();

    if thread_id == 0u {
        temp_array[bank_conflict_free_idx(n - 1u)] = 0u;
    }
    workgroupBarrier();

    for (var d: u32 = 1u; d < n; d *= 2u) {
        offset >>= 1u;
        if thread_id < d {
            var ai: u32 = offset * (2u * thread_id + 1u) - 1u;
            var bi: u32 = offset * (2u * thread_id + 2u) - 1u;
            var temp: u32 = temp_array[bank_conflict_free_idx(ai)];
            temp_array[bank_conflict_free_idx(ai)] = temp_array[bank_conflict_free_idx(bi)];
            temp_array[bank_conflict_free_idx(bi)] += temp;
        }
        workgroupBarrier();
    }

    if thread_id < (n >> 1u) {
        sums_output[2u * global_thread_id] = temp_array[bank_conflict_free_idx(2u * thread_id)];
        sums_output[2u * global_thread_id + 1u] = temp_array[bank_conflict_free_idx(2u * thread_id + 1u)];
    }
}

@compute @workgroup_size(256)
fn add_block_sums_to_prefix_sum_output(@builtin(global_invocation_id) global_invocation_id: vec3<u32>,
    @builtin(local_invocation_id) local_invocation_id: vec3<u32>,
    @builtin(workgroup_id) workgroup_id: vec3<u32>) {
    var thread_id: u32 = local_invocation_id.x;
    var global_thread_id: u32 = global_invocation_id.x;
    if thread_id < (n >> 1u) {
        prefix_sum_output[2u * global_thread_id] += sums_output[workgroup_id.x];
        prefix_sum_output[2u * global_thread_id + 1u] += sums_output[workgroup_id.x];
    }
}

@compute
@workgroup_size(64)
fn main(
    @builtin(global_invocation_id) global_invocation_id: vec3<u32>
) {
    let i = global_invocation_id.x;

    if i >= simulation_params.particles_len {
        return;
    }

    let force = simulation_params.gravity_force + calculate_pressure_force(i) + calculate_viscosity_force(i);

    let acceleration = (force / densities[i]) * simulation_params.time_step;

    velocity_x[i] += acceleration.x;
    velocity_y[i] += acceleration.y;
    position_x[i] += velocity_x[i] * simulation_params.time_step;
    position_y[i] += velocity_y[i] * simulation_params.time_step;

    if position_x[i] < -1.0 || position_x[i] > 1.0 {
        velocity_x[i] *= (-1f) * simulation_params.restitution;
        position_x[i] = clamp(position_x[i], -0.99, 0.99);
    }

    if position_y[i] < -1.0 || position_y[i] > 1.0 {
        velocity_y[i] *= (-1f) * simulation_params.restitution;
        position_y[i] = clamp(position_y[i], -0.99, 0.99);
    }
}

