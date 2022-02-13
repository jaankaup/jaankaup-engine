// Lookup table for ctz. Maybe this should be a uniform.
var<private> lookup: array<u32, 32> = array<u32, 32>(
    0u, 1u, 28u, 2u, 29u, 14u, 24u, 3u, 30u, 22u, 20u, 15u, 25u, 17u, 4u, 8u,
    31u, 27u, 13u, 23u, 21u, 19u, 16u, 7u, 26u, 12u, 18u, 6u, 11u, 5u, 10u, 9u
);

// Get the bit value from postition i.
fn get_bit(x: u32, i: u32) -> u32 {
    return (x & (1u << i)) >> i;
}

// Modulo 3 for u32.
fn mod3_32(x: u32) -> u32 {
    var a = x;
    a = (a >> 16u) + (a & 0xFFFFu);
    a = (a >>  8u) + (a & 0xFFu);
    a = (a >>  4u) + (a & 0xFu);
    a = (a >>  2u) + (a & 0x3u);
    a = (a >>  2u) + (a & 0x3u);
    a = (a >>  2u) + (a & 0x3u);
    if (a > 2u) { a = a - 3u; }
    return a;
}

// Rotate first N bits in place to the right. This is now tuned for 3-bits.
// For more general function, implement function that has number of bits as parameter.
fn rotate_right(x: u32, amount: u32) -> u32 {
    let i = mod3_32(amount);
    return (x >> i)^(x << (3u-i)) & !(0xFFFFFFFFu<<3u);
}

// Rotate first N bits in place to the left. This is now tuned for 3-bits.
// For more general function, implement function that has number of bits as parameter.
fn rotate_left(x: u32, amount: u32) -> u32 {
    let i = mod3_32(amount);
    return (x << i) ^ (x >> (3u-i)) & !(0xFFFFFFFFu<<3u);
}

// Calculate the gray code.
fn gc(i: u32) -> u32 {
    return i ^ (i >> 1u);
}

// Calculate the inverse gray code.
fn inverse_gc(g: u32) -> u32 {
    return (g ^ (g >> 1u)) ^ (g >> 2u);
}

// TODO: what happend when u32 is a really big number, larger than i32 can hold?
fn countTrailingZeros(x: u32) -> u32 {
    return lookup[
 	u32((i32(x) & (-(i32(x))))) * 0x077CB531u >> 27u
    ];
}

///////////////////////////
////// HILBERT INDEX //////
///////////////////////////

// Calculate the entry point of hypercube i.
fn e(i: u32) -> u32 {
    if (i == 0u) { return 0u; } else { return gc(2u * ((i - 1u) / 2u)); }
}

// Calculate the exit point of hypercube i.
fn f(i: u32) -> u32 {
   return e((1u << 3u) - 1u - i) ^ (1u << 2u);
}

// Calculate the direction between i and the next one.
fn g(i: u32) -> u32 {
    return countTrailingZeros(!i);
}

// Calculate the direction of the arrow whitin a subcube.
fn d(i: u32) -> u32 {
    if (i == 0u) { return 0u; }
    else if ((i & 1u) == 0u) {
        return mod3_32(g(i - 1u));
    }
    else {
        return mod3_32(g(i));
    }
}

// Transform b.
fn t(e: u32, d: u32, b: u32) -> u32 {
    return rotate_right(b^e, d + 1u);
}

// Inverse transform.
fn t_inv(e: u32, d: u32, b: u32) -> u32 {
    return rotate_left(b, d + 1u) ^ e;
}

// Calculate the hilbert index from 3d point p.
// m is the number of bits that represent the value of single coordinate.
// If m is 2, then the compute domain is: m X m X m => 2x2x2
fn to_hilbert_index(p: vec3<u32>, m: u32) -> u32 {

    var h = 0u;

    var ve: u32 = 0u;
    var vd: u32 = 0u;

    var i: u32 = m - 1u;
    loop {
        if (i == 0u) { break; }

        let l = get_bit(p.x, i) | (get_bit(p.y, i) << 1u) | (get_bit(p.z, i) << 2u);
    	let w = inverse_gc(t(l, ve, vd));
    	ve = ve ^ (rotate_left(e(w), vd + 1u));
    	vd = mod3_32(vd + d(w) + 1u);
        h = (h << 3u) | w;

        i = i - 1u;
    }
    return h;
}

// Calculate 3d point from hilbert index.
// m is the number of bits that represent the value of single coordinate.
// If m is 2, then the compute domain is: m X m X m => 2x2x2
fn from_hilber_index(h: u32, m: u32) -> vec3<u32> {
    
    var ve: u32 = 0u;
    var vd: u32 = 0u;
    var p = vec3<u32>(0u, 0u, 0u);

    var i: u32 = m - 1u;
    loop {
        if (i == 0u) { break; }

        let w = get_bit(h, i*3u) | (get_bit(h, i*3u + 1u) << 1u) | (get_bit(h, i*3u + 2u) << 2u);
    	let l = t_inv(ve, vd, gc(w)); 
    	p.x = (p.x << 1u) | ((l >> 0u) & 1u);
    	p.y = (p.y << 1u) | ((l >> 1u) & 1u);
    	p.z = (p.z << 1u) | ((l >> 2u) & 1u);
    	ve = ve ^ rotate_left(e(w), vd + 1u);
    	vd = mod3_32(vd + d(w) + 1u);

        i = i - 1u;
    }
    return p;
}

//////////////////////////////
////// Grid based curve //////
//////////////////////////////

// Map index to 3d coordinate (hexahedron). The x and y dimensions are chosen. The curve goes from left to right, row by row.
// The z direction is "unlimited".
fn index_to_uvec3(index: u32, dim_x: u32, dim_y: u32) -> vec3<u32> {
  var x  = index;
  let wh = dim_x * dim_y;
  let z  = x / wh;
  x  = x - z * wh; // check
  let y  = x / dim_x;
  x  = x - y * dim_x;
  return vec3<u32>(x, y, z);
}
