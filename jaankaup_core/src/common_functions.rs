use std::cmp;

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

pub fn get_bit(x: u32, i: u32) -> u32 {
    (x & (1 << i)) >> i
}

/// Modulo 3 for u32.
pub fn mod3_32(x: u32) -> u32 {
    let mut a = x;
    a = (a >> 16) + (a & 0xFFFF);
    a = (a >>  8) + (a & 0xFF);
    a = (a >>  4) + (a & 0xF);
    a = (a >>  2) + (a & 0x3);
    a = (a >>  2) + (a & 0x3);
    a = (a >>  2) + (a & 0x3);
    if a > 2 { a = a - 3; }
    a
}

/// Modulo 3 for u64.
pub fn mod3_64(x: u64) -> u64 {
    let mut a = x;
    a = (a >> 32) + (a & 0xFFFFFFFF);
    a = (a >> 16) + (a & 0xFFFF);
    a = (a >>  8) + (a & 0xFF);
    a = (a >>  4) + (a & 0xF);
    a = (a >>  2) + (a & 0x3);
    a = (a >>  2) + (a & 0x3);
    a = (a >>  2) + (a & 0x3);
    if a > 2 { a = a - 3; }
    a
}

const N: u32 = 3;

/// Rotate first N bits in place to the right.
#[allow(dead_code)]
fn rotate_right(x: u32, amount: u32) -> u32 {
    let i = mod3_32(amount);
    (x >> i)^(x << (N-i)) & !(std::u32::MAX<<N)
}

/// Rotate first N bits in place to the left.
#[allow(dead_code)]
fn rotate_left(x: u32, amount: u32) -> u32 {
    let i = mod3_32(amount);
    !(std::u32::MAX<<N) & (x << i) ^ (x >> (N-i))
}

/* HILBERT INDEXING. */

/// Calculate the gray code.
#[allow(dead_code)]
pub fn gc(i: u32) -> u32 {
    i ^ (i >> 1)
}

/// Calculate the entry point of hypercube i.
#[allow(dead_code)]
pub fn e(i: u32) -> u32 {
    if i == 0 { 0 } else { gc(2 * ((i-1) / 2)) }
}

/// Extract the 3d position from a 3-bit integer.
#[allow(dead_code)]
fn i_to_p(i: u32) -> [u32; 3] {
    [get_bit(i, 0), get_bit(i, 1),get_bit(i, 2)] 
}

/// Calculate the inverse gray code.
#[allow(dead_code)]
fn inverse_gc(g: u32) -> u32 {
    (g ^ (g >> 1)) ^ (g >> 2)
}

/// Calculate the direction between i and the next one.
#[allow(dead_code)]
pub fn g(i: u32) -> u32 {
    (!i).trailing_zeros()
}

/// Calculate the direction of the arrow whitin a subcube.
#[allow(dead_code)]
pub fn d(i: u32) -> u32 {
    if i == 0 { 0 }
    else if i&1 == 0 {
        mod3_32(g(i-1)) // % N
    }
    else {
        mod3_32(g(i)) // % N
    }
}

/// Transform b.
#[allow(dead_code)]
pub fn t(e: u32, d: u32, b: u32) -> u32 {
    rotate_right(b^e, d+1)
}

/// Inverse transform.
#[allow(dead_code)]
pub fn t_inv(e: u32, d: u32, b: u32) -> u32 {
    rotate_left(b, d+1) ^ e 
}

// const M: u32 = 3;

/// Calculate the hilbert index from 3d point p.
pub fn to_hilbert_index(p: [u32; 3], m: u32) -> u32 {

    let mut h = 0;
    let mut ve: u32 = 0;
    let mut vd: u32 = 0;

    for i in (0..m).rev() { // TODO: check
        let l = get_bit(p[0], i) | (get_bit(p[1], i) << 1) | (get_bit(p[2], i) << 2);
        let w = inverse_gc(t(l, ve, vd));
        ve = ve ^ (rotate_left(e(w), vd+1));
        vd = mod3_32(vd + d(w) + 1); //% N;
        //vd = mod3_32(vd + d(w) + 1); //% N;
        h = (h << N) | w;
    }
    h
}

/// Calculate 3d point from hilbert index.
pub fn from_hilber_index(h: u32, m: u32) -> [u32; 3] {
    
    let mut ve: u32 = 0;
    let mut vd: u32 = 0;
    let mut p: [u32; 3] = [0, 0, 0];

    for i in (0..m).rev() { // TODO: check
        let w = get_bit(h, i*N) | (get_bit(h, i*N + 1) << 1) | (get_bit(h, i*N + 2) << 2);
        // let mut l = gc(w); 
        let l = t_inv(ve, vd, gc(w)); 
        p[0] = (p[0] << 1) | ((l >> 0) & 1);
        p[1] = (p[1] << 1) | ((l >> 1) & 1);
        p[2] = (p[2] << 1) | ((l >> 2) & 1);
        ve = ve ^ rotate_left(e(w), vd+1);
        vd = mod3_32(vd + d(w) + 1);
    }
    p
}

// pub fn gcr(i: u32, mu: u32) -> u32 {
//     let mut r = 0;
//     for k in (0..N).rev() {
//         if get_bit(mu, k) == 1 {
//             r = (r << 1) | get_bit(i,k);
//         }
//     }
//     r
// }
// 
// pub fn gcr_inv(r: u32, mu: u32, pi: u32) -> u32 {
//     let mut i = 0;
//     let mut g = 0;
//     let mut j: i32 = (get_bit(mu, 0) + get_bit(mu, 1) + get_bit(mu, 2)) as i32 - 1;
//     for k in (0..N).rev() {
//         if get_bit(mu, k) == 1 {
//             i = i | (get_bit(r, j as u32) << k);
//             g = g | (((get_bit(i, k) + get_bit(i, k+1)) % 2) << k);
//             j = j - 1;
//         }
//         else {
//             g = g | (get_bit(pi, k) << k); 
//             i = i | (((get_bit(g, k) + get_bit(i, k+1)) % 2) << k);
//         }
//     }
//     i
// }
// 
// const compact_M: [u32 ; 3] = [4,2,2];
// 
// pub fn extract_mask(i: u32) -> u32 {
//     let mut mu = 0;
//     for j in (0..N).rev() {
//         mu = mu << 1;
//         if compact_M[j as usize] > i {
//             mu = mu | 1;
//         }
//     }
//     mu
// }
// 
// /// Compute the point with compact Hilbert index h. Not working.
// pub fn compact_hilbert_index(p: [u32 ; 3]) -> u32 {
//     let mut h = 0;
//     let mut ve: u32 = 0;
//     let mut vd: u32 = 2;
//     let m = cmp::max(cmp::max(compact_M[0], compact_M[1]), compact_M[2]);  
//     for i in (0..m).rev() {
//         let mut mu = extract_mask(i);
//         let mu_norm = get_bit(mu, 0) + get_bit(mu, 1) + get_bit(mu, 2);
//         mu = rotate_right(mu, vd+1);
//         let mut l = get_bit(p[0], i) | (get_bit(p[1], i) << 1) | (get_bit(p[2], i) << 2);
//         l = t(ve, vd as i32, l);
//         let w = inverse_gc(l);
//         let r = gcr(w, mu);
//         ve = ve ^ rotate_left(e(w), vd+1);
//         vd = (vd + d(w) + 1) % N;
//         h = (h << mu_norm) | r;
//     }
//     h
// }

/// Not working.
// pub fn from_compact_hilbert_index(h: u32) -> [u32; 3] {
//     
//     let mut ve: u32 = 0;
//     let mut vd: u32 = 2;
//     let mut k = 0;
//     let mut p: [u32 ; 3] = [0, 0, 0];
//     let m = cmp::max(cmp::max(compact_M[0], compact_M[1]), compact_M[2]);  
//     let vM = compact_M[0] + compact_M[1] + compact_M[2];
// 
//     for i in (0..m).rev() {
//         // Duplicate code.
//         let mut mu = extract_mask(i);
//         let mu_norm = get_bit(mu, 0) + get_bit(mu, 1) + get_bit(mu, 2);
//         mu = rotate_right(mu, vd+1);
//         let pi = rotate_right(ve, vd+1) & ((!mu) & ((1 << N) - 1)); // ???
// 
//         let mut r = 0;
//         for j in 0..mu_norm {
//             r = r | (get_bit(h, vM - k - (j+1)) << (mu_norm - 1 - j)); // TODO: check.
//         }
//         k = k + mu_norm;
//         let w = gcr_inv(r, mu, pi);
//         let mut l = gc(w);
//         l = t_inv(ve, vd, l);
//         p[0] = p[0] | (get_bit(l, 0) << i);
//         p[1] = p[1] | (get_bit(l, 1) << i);
//         p[2] = p[2] | (get_bit(l, 2) << i);
//         ve = ve ^ rotate_left(e(w), vd+1);
//         vd = (vd + d(w) + 1) % N;
//     }
//     p 
// }

/* Rosenberg-Strong */ 

/// Rosenberg-Strong tuple to index. 
pub fn r2(x: i32, y: i32) -> i32 {
    let max2 = cmp::max(x,y);
    max2 * max2 + max2 + x - y 
}

/// Rosenberg-Strong triple to index. 
pub fn r3(x: i32, y: i32, z: i32) -> i32 {
    let max2 = cmp::max(x,y);
    let max3  = cmp::max(z, max2);
    let max3_2 = max3 * max3;
    let max3_3 = max3 * max3_2;
    max2 * max2 + max2 + x - y + max3_3 + (max3 - z) * (2 * max3 + 1)
}

//#[allow(dead_code)]
fn floored_root(x: i32, n: i32) -> i32 {
    (x as f64).powf(1.0/(n as f64)).floor() as i32
}

/// Rosenberg-Strong index to tuple. 
pub fn r2_reverse(z: i32) -> (i32, i32) {
    let m = floored_root(z, 2);
    let m_powf2 = m * m; 
    let t = z - m_powf2;
    if t < m { (t, m) }
    else {
        (m, m_powf2 + 2 * m - z)
    }
}

/// Rosenberg-Strong index to triple. 
pub fn r3_reverse(z: i32) -> (i32, i32, i32) {
    let m = (z as f64).cbrt().floor() as i32;
    let t = z - m*m*m - m*m;
    let v = if t < 0 { 0 } else { t };
    let x3 = m - (v/((m + 1) * (m + 1) - m*m));
    let (x,y) = r2_reverse(z - m*m*m - (m - x3) * ((m + 1) * (m + 1) - m*m));
    (x as i32, y as i32, x3 as i32)
}
