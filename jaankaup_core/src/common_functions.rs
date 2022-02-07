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

pub fn bit_component(x: u32, i: u32) -> u32 {
    (x & (1 << i)) >> i
}

/////////// HILBERT ///////////////////

pub fn mod3(x: u32) -> u32 {
    let mut a = x;
    a = (a >> 16) + (a & 0xFFFF);
    a = (a >>  8) + (a & 0xFF);
    a = (a >>  4) + (a & 0xF);
    a = (a >>  2) + (a & 0x3);
    a = (a >>  2) + (a & 0x3);
    a = (a >>  2) + (a & 0x3);
    if (a > 2) { a = a - 3; }
    a
}

pub fn mod3_64(x: u64) -> u64 {
    let mut a = x;
    a = (a >> 32) + (a & 0xFFFFFFFF);
    a = (a >> 16) + (a & 0xFFFF);
    a = (a >>  8) + (a & 0xFF);
    a = (a >>  4) + (a & 0xF);
    a = (a >>  2) + (a & 0x3);
    a = (a >>  2) + (a & 0x3);
    a = (a >>  2) + (a & 0x3);
    if (a > 2) { a = a - 3; }
    a
}

const N: u32 = 3;

/// Rotate first N bits in place to the right.
fn rotate_right(x: u32, amount: u32) -> u32 {
    let i = mod3(amount);
    (x >> i)^(x << (N-i)) & !(std::u32::MAX<<N)
}

/// Rotate first N bits in place to the left.
fn rotate_left(x: u32, amount: u32) -> u32 {
    let i = mod3(amount);
    !(std::u32::MAX<<N) & (x << i) ^ (x >> (N-i))
}

/// Calculate the gray code.
pub fn gc(i: u32) -> u32 {
    i ^ (i >> 1)
}

/// Calculate the entry point of hypercube i.
pub fn e(i: u32) -> u32 {
    if i == 0 { 0 } else { gc(2 * ((i-1) / 2)) }
    //if i == 0 { 0 } else { gc(2 * ((i-1) as f64 * 0.5).floor() as u32) }
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
    (g ^ (g >> 1)) ^ (g >> 2)
}

/// Calculate the direction between i and the next one.
pub fn g(i: u32) -> u32 {
    // let temp = gc(i) ^ gc(i+1);
    // 32 - temp.leading_zeros() - 1
    !i.trailing_zeros()
    //(!i).trailing_zeros()
}

/// Calculate the direction of the arrow whitin a subcube.
pub fn d(i: u32) -> u32 {
    if i == 0 { 0 }
    else if i&1 == 0 {
        mod3(g(i-1)) // % N
    }
    else {
        mod3(g(i)) // % N
    }
}

/// Transform b.
pub fn t(e: u32, d: i32, b: u32) -> u32 {
    rotate_right(b^e, (d+1) as u32)
}

/// Inverse transform.
pub fn t_inv(e: u32, d: u32, b: u32) -> u32 {
    rotate_left(b, d+1) ^ e 
}

const M: u32 = 3;

/// Return the Hilbert index of point p.
pub fn to_hilbert_index(p: [u32; 3], m: u32) -> u32 {

    // The Hilbert index.
    let mut h = 0;

    let mut ve: u32 = 0;
    let mut vd: u32 = 2;

    for i in (0..m).rev() { // TODO: check

        // Take the i:th bits from p elements and create a new bit sequence p2[i]p1[i]p0[i].
        let mut l = bit_component(p[0], i) | (bit_component(p[1], i) << 1) | (bit_component(p[2], i) << 2);

        // Transform l into the current subcube.
        //l = t(ve, vd as i32, l);

        // Obtain the gray code ordering from the label l.
        let w = inverse_gc(t(ve, vd as i32, l));

        ve = ve ^ (rotate_left(e(w), vd+1));
        vd = mod3((vd + d(w) + 1)); //% N;

        h = (h << N) | w;
    }
    h
}

pub fn from_hilber_index(h: u32, m: u32) -> [u32; 3] {
    
    let mut ve: u32 = 0;
    let mut vd: u32 = 2;
    let mut p: [u32; 3] = [0, 0, 0];

    for i in (0..m).rev() { // TODO: check
        let mut w = bit_component(h, i*N) | (bit_component(h, i*N + 1) << 1) | (bit_component(h, i*N + 2) << 2);
        let mut l = gc(w); 
        l = t_inv(ve, vd, l); 
        p[0] = (p[0] << 1) | ((l >> 0) & 1) ;; //p[0] | (bit_component(l, 0) << i);
        p[1] = (p[1] << 1) | ((l >> 1) & 1) ;; //p[0] | (bit_component(l, 0) << i);
        p[2] = (p[2] << 1) | ((l >> 2) & 1) ;; //p[0] | (bit_component(l, 0) << i);
        // p[0] = p[0] + bit_component(l, 0) << i;
        // p[1] = p[1] + bit_component(l, 1) << i;
        // p[2] = p[2] + bit_component(l, 2) << i;
        ve = ve ^ rotate_left(e(w), vd+1);
        vd = mod3((vd + d(w) + 1)); // % N;
    }
    p
}

pub fn gcr(i: u32, mu: u32) -> u32 {
    let mut r = 0;
    for k in (0..N).rev() {
        if bit_component(mu, k) == 1 {
            r = (r << 1) | bit_component(i,k);
        }
    }
    r
}

pub fn gcr_inv(r: u32, mu: u32, pi: u32) -> u32 {
    let mut i = 0;
    let mut g = 0;
    let mut j: i32 = (bit_component(mu, 0) + bit_component(mu, 1) + bit_component(mu, 2)) as i32 - 1;
    for k in (0..N).rev() {
        if bit_component(mu, k) == 1 {
            i = i | (bit_component(r, j as u32) << k);
            g = g | (((bit_component(i, k) + bit_component(i, k+1)) % 2) << k);
            j = j - 1;
        }
        else {
            g = g | (bit_component(pi, k) << k); 
            i = i | (((bit_component(g, k) + bit_component(i, k+1)) % 2) << k);
        }
    }
    i
}

const compact_M: [u32 ; 3] = [4,2,2];

pub fn extract_mask(i: u32) -> u32 {
    let mut mu = 0;
    for j in (0..N).rev() {
        mu = mu << 1;
        if compact_M[j as usize] > i {
            mu = mu | 1;
        }
    }
    mu
}

/// Compute the point with compact Hilbert index h.
pub fn compact_hilbert_index(p: [u32 ; 3]) -> u32 {
    let mut h = 0;
    let mut ve: u32 = 0;
    let mut vd: u32 = 2;
    let m = cmp::max(cmp::max(compact_M[0], compact_M[1]), compact_M[2]);  
    //for i in (0..(m-1)).rev() {
    for i in (0..m).rev() {
        let mut mu = extract_mask(i);
        //println!("mu = extract_mask({}) == {}", i, extract_mask(i));
        let mu_norm = bit_component(mu, 0) + bit_component(mu, 1) + bit_component(mu, 2);
        //println!("mu_norm = {}", mu_norm);
        mu = rotate_right(mu, vd+1);
        //println!("mu = {}", mu);
        //let pi = rotate_right(ve, vd+1) & ((!mu) & (1 << (N-1)));
        let mut l = bit_component(p[0], i) | (bit_component(p[1], i) << 1) | (bit_component(p[2], i) << 2);
        l = t(ve, vd as i32, l);
        //println!("l = {}", l);
        let w = inverse_gc(l);
        //println!("w = {}", w);
        let r = gcr(w, mu);
        ve = ve ^ rotate_left(e(w), vd+1);
        vd = (vd + d(w) + 1) % N;
        h = (h << mu_norm) | r;
    }
    h
}

pub fn from_compact_hilbert_index(h: u32) -> [u32; 3] {
    
    let mut ve: u32 = 0;
    let mut vd: u32 = 2;
    let mut k = 0;
    let mut p: [u32 ; 3] = [0, 0, 0];
    let m = cmp::max(cmp::max(compact_M[0], compact_M[1]), compact_M[2]);  
    let vM = compact_M[0] + compact_M[1] + compact_M[2];

    for i in (0..m).rev() {
        // Duplicate code.
        let mut mu = extract_mask(i);
        let mu_norm = bit_component(mu, 0) + bit_component(mu, 1) + bit_component(mu, 2);
        mu = rotate_right(mu, vd+1);
        let pi = rotate_right(ve, vd+1) & ((!mu) & ((1 << N) - 1)); // ???

        let mut r = 0;
        for j in 0..mu_norm {
            r = r | (bit_component(h, vM - k - (j+1)) << (mu_norm - 1 - j)); // TODO: check.
        }
        k = k + mu_norm;
        let w = gcr_inv(r, mu, pi);
        let mut l = gc(w);
        l = t_inv(ve, vd, l);
        p[0] = p[0] | (bit_component(l, 0) << i);
        p[1] = p[1] | (bit_component(l, 1) << i);
        p[2] = p[2] | (bit_component(l, 2) << i);
        ve = ve ^ rotate_left(e(w), vd+1);
        vd = (vd + d(w) + 1) % N;
    }
    p 
}
