

@binding(0) @group(0) var<storage, read> input :array<u32>;
@binding(1) @group(0) var<storage, read_write> output :array<vec4<u32>>;
@binding(2) @group(0) var<storage, read_write> sums: array<u32>;
@binding(0) @group(1) var<uniform> radixMaskId:u32;
const bank_size:u32 = 32u;
const n:u32 = 512u;
var<workgroup> temp0: array<u32,532>;
var<workgroup> temp1: array<u32,532>;
var<workgroup> temp2: array<u32,532>;
var<workgroup> temp3: array<u32,532>;
fn bank_conflict_free_idx(idx: u32) -> u32 {
    var chunk_id: u32 = idx / bank_size;
    return idx + chunk_id;
}
 
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) GlobalInvocationID: vec3<u32>,
    @builtin(local_invocation_id) LocalInvocationID: vec3<u32>,
    @builtin(workgroup_id) WorkgroupID: vec3<u32>) {
    var thid: u32 = LocalInvocationID.x;
    var globalThid: u32 = GlobalInvocationID.x;
    var mask: u32 = u32(3) << (radixMaskId << 1u);
    if thid < (n >> 1u) {

        var val: u32 = (input[2u * globalThid] & mask) >> (radixMaskId << 1u);

        if val == 0u {
            temp0[bank_conflict_free_idx(2u * thid)] = 1u;
        } else if val == 1 {
            temp1[bank_conflict_free_idx(2u * thid)] = 1u;
        } else if val == 2 {
            temp2[bank_conflict_free_idx(2u * thid)] = 1u;
        } else if val == 3 {
            temp3[bank_conflict_free_idx(2u * thid)] = 1u;
        }

        val = (input[2u * globalThid + 1u] & mask) >> (radixMaskId << 1u);

        if val == 0u {
            temp0[bank_conflict_free_idx(2u * thid + 1u)] = 1u;
        } else if val == 1 {
            temp1[bank_conflict_free_idx(2u * thid + 1u)] = 1u;
        } else if val == 2 {
            temp2[bank_conflict_free_idx(2u * thid + 1u)] = 1u;
        } else if val == 3 {
            temp3[bank_conflict_free_idx(2u * thid + 1u)] = 1u;
        }
    }
    workgroupBarrier();
    var offset: u32 = 1u;

    for (var d: u32 = n >> 1u; d > 0u; d >>= 1u) {
        if thid < d {
            var ai: u32 = offset * (2u * thid + 1u) - 1u;
            var bi: u32 = offset * (2u * thid + 2u) - 1u;
            temp0[bank_conflict_free_idx(bi)] += temp0[bank_conflict_free_idx(ai)];
            temp1[bank_conflict_free_idx(bi)] += temp1[bank_conflict_free_idx(ai)];
            temp2[bank_conflict_free_idx(bi)] += temp2[bank_conflict_free_idx(ai)];
            temp3[bank_conflict_free_idx(bi)] += temp3[bank_conflict_free_idx(ai)];
        }
        offset *= 2u;

        workgroupBarrier();
    }

    if thid == 0u {
        temp0[bank_conflict_free_idx(n - 1u)] = 0u;
        temp1[bank_conflict_free_idx(n - 1u)] = 0u;
        temp2[bank_conflict_free_idx(n - 1u)] = 0u;
        temp3[bank_conflict_free_idx(n - 1u)] = 0u;
    }
    workgroupBarrier();

    for (var d: u32 = 1u; d < n; d *= 2u) {
        offset >>= 1u;
        if thid < d {
            var ai: u32 = offset * (2u * thid + 1u) - 1u;
            var bi: u32 = offset * (2u * thid + 2u) - 1u;
            var t: u32 = temp0[bank_conflict_free_idx(ai)];
            temp0[bank_conflict_free_idx(ai)] = temp0[bank_conflict_free_idx(bi)];
            temp0[bank_conflict_free_idx(bi)] += t;


            t = temp1[bank_conflict_free_idx(ai)];
            temp1[bank_conflict_free_idx(ai)] = temp1[bank_conflict_free_idx(bi)];
            temp1[bank_conflict_free_idx(bi)] += t;

            t = temp2[bank_conflict_free_idx(ai)];
            temp2[bank_conflict_free_idx(ai)] = temp2[bank_conflict_free_idx(bi)];
            temp2[bank_conflict_free_idx(bi)] += t;

            t = temp3[bank_conflict_free_idx(ai)];
            temp3[bank_conflict_free_idx(ai)] = temp3[bank_conflict_free_idx(bi)];
            temp3[bank_conflict_free_idx(bi)] += t;
        }
        workgroupBarrier();
    }
    if thid == 0u {
        var count0: u32 = temp0[bank_conflict_free_idx(2u * 255u)];
        var count1: u32 = temp1[bank_conflict_free_idx(2u * 255u)];
        var count2: u32 = temp2[bank_conflict_free_idx(2u * 255u)];
        var count3: u32 = temp3[bank_conflict_free_idx(2u * 255u)];

        var last: u32 = (input[2u * ((WorkgroupID.x + 1u) * 256u - 1u)] & mask) >> (radixMaskId << 1u);
        switch(last) {
              case 0u: {count0 += 1u;}
              case 1u: {count1 += 1u;}
              case 2u: {count2 += 1u;}
              case 3u: {count3 += 1u;}
              default {}
          }

        last = (input[2u * ((WorkgroupID.x + 1u) * 256u - 1u) + 1u] & mask) >> (radixMaskId << 1u);
        switch(last) {
              case 0u: {count0 += 1u;}
              case 1u: {count1 += 1u;}
              case 2u: {count2 += 1u;}
              case 3u: {count3 += 1u;}
              default {}
          }

        sums[WorkgroupID.x * 4u] = count0;
        sums[WorkgroupID.x * 4u + 1u] = count1;
        sums[WorkgroupID.x * 4u + 2u] = count2;
        sums[WorkgroupID.x * 4u + 3u] = count3;
    }
    if thid < (n >> 1u) {
        output[2u * globalThid].x = temp0[bank_conflict_free_idx(2u * thid)];
        output[2u * globalThid + 1u].x = temp0[bank_conflict_free_idx(2u * thid + 1u)];

        output[2u * globalThid].y = temp1[bank_conflict_free_idx(2u * thid)];
        output[2u * globalThid + 1u].y = temp1[bank_conflict_free_idx(2u * thid + 1u)];

        output[2u * globalThid].z = temp2[bank_conflict_free_idx(2u * thid)];
        output[2u * globalThid + 1u].z = temp2[bank_conflict_free_idx(2u * thid + 1u)];

        output[2u * globalThid].w = temp3[bank_conflict_free_idx(2u * thid)];
        output[2u * globalThid + 1u].w = temp3[bank_conflict_free_idx(2u * thid + 1u)];
    }
}
