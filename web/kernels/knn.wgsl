// Brute force nearest neighbour of every query patch against the memory bank.
//
// One workgroup per query patch. Each thread strides over bank rows, keeps a
// local best, then the workgroup tree-reduces to a single minimum. The stride
// loop covers every row for any N, so there is no ceil-division gap, but the
// nn_at_last golden checks that anyway because this is the bug that looks
// correct on a heatmap while silently skipping the tail of the bank.
//
// Distances accumulate squared in f32 and sqrt at the end. Squared is fine for
// argmin because it preserves order. It is NOT fine inside the eq.7 exp, which
// is a separate pass and takes the L2 value this shader writes.

struct Params {
  n_bank : u32,   // rows in the bank
  n_query: u32,   // query patches, 784 for a 28x28 grid
  dim    : u32,   // 1536
  _pad   : u32,
};

@group(0) @binding(0) var<storage, read>       bank    : array<f32>;
@group(0) @binding(1) var<storage, read>       queries : array<f32>;
@group(0) @binding(2) var<storage, read_write> out_min : array<f32>;
@group(0) @binding(3) var<storage, read_write> out_idx : array<u32>;
@group(0) @binding(4) var<uniform>             params  : Params;

const WG : u32 = 256u;

var<workgroup> red_d : array<f32, WG>;
var<workgroup> red_i : array<u32, WG>;

@compute @workgroup_size(256)
fn knn(@builtin(workgroup_id) wg : vec3<u32>,
       @builtin(local_invocation_id) lid : vec3<u32>) {
  let q = wg.x;
  let t = lid.x;
  if (q >= params.n_query) { return; }

  let qbase = q * params.dim;
  var best_d : f32 = 3.4e38;
  var best_i : u32 = 0u;

  var r : u32 = t;
  loop {
    if (r >= params.n_bank) { break; }
    let rbase = r * params.dim;
    var acc : f32 = 0.0;
    for (var k : u32 = 0u; k < params.dim; k = k + 1u) {
      let d = queries[qbase + k] - bank[rbase + k];
      acc = acc + d * d;
    }
    // strict less-than, and on an exact tie prefer the lower row index, so the
    // result does not depend on which thread happened to see it first
    if (acc < best_d || (acc == best_d && r < best_i)) {
      best_d = acc;
      best_i = r;
    }
    r = r + WG;
  }

  red_d[t] = best_d;
  red_i[t] = best_i;
  workgroupBarrier();

  var s : u32 = WG / 2u;
  loop {
    if (s == 0u) { break; }
    if (t < s) {
      let od = red_d[t + s];
      let oi = red_i[t + s];
      if (od < red_d[t] || (od == red_d[t] && oi < red_i[t])) {
        red_d[t] = od;
        red_i[t] = oi;
      }
    }
    workgroupBarrier();
    s = s / 2u;
  }

  if (t == 0u) {
    out_min[q] = sqrt(red_d[0]);   // L2, not squared
    out_idx[q] = red_i[0];
  }
}
