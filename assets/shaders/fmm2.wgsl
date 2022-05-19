///
/// Fast marching method kernel.
///

// Fmm tags.
let FAR      = 0u;
let BAND     = 1u;
let KNOWN    = 2u;

/// A struct for one fast marching methow cell.
struct FmmCell {
    tag: atomic<u32>,
    value: f32,
    update: atomic<u32>,
    misc: u32,
};

/// A struct that holds information for a single computational domain area.
struct FmmBlock {
    index: u32,
    band_point_count: u32,
};

//
// +---+---+---+---+---+---+---+---+---+---+---+ 
// |   |   |   |   |   |   |   |   |   |   |   |
// +---+---+---+---+---+---+---+---+---+---+---+ 
// |   |   |   |   |   |   |   |   |   |   |   |
// +---+---+---+---+---+---+---+---+---+---+---+ 
// |   |   |   |   |   |   |   |   |   |   |   |
// +---+---+---+---+---+---+---+---+---+---+---+ 
// |   |   |   |   |   |   |   |   |   |   |   |
// +---+---+---+---+---+---+---+---+---+---+---+ 
// |   |   |   |   |   |   |   |   |   |   |   |
// +---+---+---+---+---+---+---+---+---+---+---+ 
// |   |   |   |   |   |   |   |   |   |   |   |
// +---+---+---+---+---+---+---+---+---+---+---+ 
// |   |   |   |   |   |   |   |   |   |   |   |
// +---+---+---+---+---+---+---+---+---+---+---+ 
// |c12|c13|c14|c15|c28|c29|c30|c31|   |   |   |
// +---+---+---+---+---+---+---+---+---+---+---+ 
// | c8| c9|c10|c11|c24|c25|c26|c27|   |   |   |
// +---+---+---+---+---+---+---+---+---+---+---+ 
// | c4| c5| c6| c7|c20|c21|c22|c23|   |   |   |
// +---+---+---+---+---+---+---+---+---+---+---+ 
// | c0| c1| c2| c3|c16|c17|c18|c19|   |   |   |
// +---+---+---+---+---+---+---+---+---+---+---+ 

