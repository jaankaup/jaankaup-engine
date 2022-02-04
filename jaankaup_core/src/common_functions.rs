pub fn udiv_up_32(x: u32, y: u32) -> u32 {
  (x + y - 1) / y
}

pub fn udiv_up_safe32(x: u32, y: u32) -> u32 {
  if y == 0 { 0 } else { (x + y - 1) / y }
}

pub fn set_bit_to(original_value: u32, pos: u32, value: bool) -> u32 {
    let val = if value { 1 } else { 0 };
    original_value & !(1 << pos) | (val << pos)
}

pub fn get_range(original_value: u32, start: u32, amount: u32) -> u32 {
    (original_value >> start) & ((1 << amount) - 1)
}

pub fn bit_component(x: u32, i: u32) -> u32 {
    (x & (1 << i)) >> i
}

/////////// HILBERT ///////////////////

const N: u32 = 3;

/// Rotate first N bits in place to the right.
fn rotate_right(x: u32, amount: u32) -> u32 {
    let mut d = amount % N;
    let mut out = x >> d;
    for i in 0..d {
        let bit = (x & (1 << i)) >> i;
        out = out | (bit << (N+i-d))
    }
    out
}

/// Rotate first N bits in place to the left.
fn rotate_left(x: u32, amount: u32) -> u32 {
    let mut d = amount % N;
    let mut out = x << d;
    // let excess = out;
    out = out & ((1 << N) - 1);
    for i in 0..d {
        let t = N-d+i; 
        // let bit = (x & (1 << (N-1-d+1+i))) >> (N-1-d+1+i);
        let bit = (x & (1 << t)) >> t;
        out = out | (bit << i)
    }
    out
}
/// Calculate the gray code.
pub fn gc(i: u32) -> u32 {
    i ^ (i >> 1)
}

/// Calculate the entry point of hypercube i.
pub fn e(i: u32) -> u32 {
    //if i == 0 { 0 } else { gc(2*(i-1)/2) }
    if i == 0 { 0 } else { gc(2 * ((i-1) as f64 * 0.5).floor() as u32) }
}

/// Calculate the exit point of hypercube i.
pub fn f(i: u32) -> u32 {
   e((1 << N) - 1 - i) ^ (1 << (N-1))
}

/// Extract the 3d position from a 3-bit integer.
fn i_to_p(i: u32) -> [u32; 3] {
    [bit_component(i, 0), bit_component(i, 1),bit_component(i, 2)] 
}

/// Calculate the inverse gray code.
fn inverse_gc(g: u32) -> u32 {
    let mut i = g; 
    let mut j = 1; 
    while j<N {
        i = i ^ (g >> j);
        j = j + 1;
    }
    i
}

/// Calculate the direction between i and the next one.
pub fn g(i: u32) -> u32 {
    let temp = gc(i) ^ gc(i+1);
    //u32::BITS as u32 - temp.leading_zeros() - 1
    32 - temp.leading_zeros() - 1
    // log2(gc(i) ^ gc(i+1))
}

/// Calculate the direction of the arrow whitin a subcube.
pub fn d(i: u32) -> u32 {
    if i == 0 { 0 }
    else if (i%2) == 0 {
        g(i-1) % N
    }
    else {
        g(i) % N
    }
}

/// Transform b.
pub fn t(e: u32, d: i32, b: u32) -> u32 {
    let out = b ^ e; 
    rotate_right(out, (d+1) as u32)
}

/// Inverse transform.
pub fn t_inv(e: u32, d: u32, b: u32) -> u32 {
    // println!("t(rotate_right({0}, {1}+1), {2}-{1}-2, {3})", e, d, N, b);  
    t(rotate_right(e, d+1), N as i32 - d as i32 - 2, b) 
}

const M: u32 = 3;

/// Return the Hilbert index of point p.
pub fn to_hilbert_index(p: [u32; 3]) -> u32 {

    // The Hilbert index.
    let mut h = 0;

    let mut ve: u32 = 0;
    let mut vd: u32 = 2;

    // for i in (0..(M-1)).rev() {
    for i in (0..(M-1)).rev() {

        // 1. Extract the relevant bits from p.
        // let l = [bit_component(i, p[0]),bit_component(i, p[0]), bit_component(i, p[0])]; 

        // 2. Construct a integer whose bits are given by l.
        let mut l = bit_component(p[0], i) | (bit_component(p[1], i) << 1) | (bit_component(p[2], i) << 2);
        //let mut l = bit_component(p[2], i) | (bit_component(p[1], i) << 1) | (bit_component(p[0], i) << 2);
        //println!("l == {:#006b}", l);

        // Transform l into the current subcube.
        l = t(ve, vd as i32, l);

        // Obtain the gray code ordering from the label l.
        let w = inverse_gc(l);

        ve = ve ^ (rotate_left(e(w), vd+1));
        vd = (vd + d(w) + 1) % N;

        h = (h << N) | w;
    }
    h
}
