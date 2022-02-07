use jaankaup_core::common_functions::{from_hilber_index, to_hilbert_index, g, gc, e, f, t, d, t_inv, from_compact_hilbert_index, compact_hilbert_index, mod3,mod3_64};

fn main() {

    // for i in 0..(4*4*4) {
    //     let point = from_hilber_index(i, 3);
    //     println!("from_hilbert_index({:?}) == {:?}", i, point); 
    // }
    let mut map = Vec::<u32>::new();
    for i in 0..4 {
    for j in 0..4 {
    for k in 0..4 {
        
        let index = to_hilbert_index([i,j,k], 3);
        let point = from_hilber_index(index, 3);
        print!("to_hilbert_index({:?}) == {:?}    ", [i,j,k], index); 
        println!("from_hilbert_index({:?}) == {:?} :: {:?}", index, point, [i,j,k] == point); 
        // println!("to_hilbert_index({},{},{}) == {:?}", i, j, k, index);
        // map.push(index);
    }}};
    map.sort();
    for i in map.iter() {
        println!("{}", i);
    }

    //// const N: u32 = 3;

    //// print!("Only g(i)-th bit changes form gc(i) to gc(i+1)");
    //// for i in 0..(1 << (N-1)) {
    ////     assert!(gc(i) ^ (1 << g(i)) == gc(i+1), "Fail");
    //// }
    //// println!("  OK");

    //// print!("T sends e(i) to 0 and f(i) to 2**(N-1)");
    //// for i in 0..(1 << N) {
    ////     assert!(t(e(i), d(i) as i32, e(i)) == 0, "Fail");
    ////     assert!(t(e(i), d(i) as i32, f(i)) == (1 << (N-1)), "Fail");
    //// }
    //// println!("  OK");

    //// print!("e(i) reflected in direction d(i) is f(i)");
    //// for i in 0..(1 << N) {
    ////     assert!(e(i) ^ (1 << d(i)) == f(i), "Fail");
    //// }
    //// println!("  OK");

    //// print!("t_inv composet with t (and visa versa) is the identity operator.");
    //// for i in 0..(1 << N) {
    //// for b in 0..(1 << N) {
    ////     assert!(t(e(i), d(i) as i32, t_inv(e(i), d(i), b)) == b, "Fail");
    ////     assert!(t_inv(e(i), d(i), t(e(i), d(i) as i32, b)) == b, "Fail");
    //// }};
    //// println!("  OK");

    // pring gc. ok
    // for i in 0..8 {
    //     println!("gc({:?}) == {:#003b}", i, gc(i));
    // }

    // for i in 0..8 {
    //     println!("e({:?}) == {:#003b}", i, e(i));
    // }

    // for i in 0..8 {
    //     println!("f({:?}) == {:#003b}", i, f(i));
    // }
    ////for i in 0..(1 << (3+2+2)) {
    ////    let p = from_compact_hilbert_index(i);
    ////    println!("from_compact_hilbert_index({}) == {:?}", i, p); 
    ////    // compact_hilber_index(p: [u32 ; 3]) -> u32 {
    ////}
    // let mut map = Vec::<[u32; 3]>::new();
    // for i in 0..(8*2*2) {
    //     let h = from_compact_hilbert_index(i);
    //     println!("from_compact_hilbert_index({:?}) == {:?}. ", i, from_compact_hilbert_index(i));
    //     map.push(h);
    // }
    // let mut map = Vec::<u32>::new();
    // for i in 0..1 {
    // for j in 0..2 {
    // for k in 0..3 {
    //     let h = compact_hilbert_index([i,j,k]);
    //     map.push(h);
    //     let index = from_compact_hilbert_index(h);
    //     print!("compact_hilbert_index({:?} == {:?}. ", [i,j,k], compact_hilbert_index([i,j,k]));
    //     print!("from_compact_hilbert_index({:?}) == {:?}. ", h, from_compact_hilbert_index(h));
    //     println!("{:?}", index == [i,j,k]);
    // }}};
    // map.sort();
    // for i in map.iter() {
    //     println!("{:?}", i);
    // }

    // for i in 0..std::u64::MAX {
    //     //if mod3(i) != (i % 3) {
    //     //    println!("mod3({:?} == {:?}. {:?}", i, mod3(i), mod3(i) == (i % 3));
    //     //}
    //     if mod3_64(i) != (i % 3) {
    //         println!("mod3_64({:?}) == {:?}. should be {:?}", i, mod3_64(i), i % 3);
    //     }
    //     //println!("mod3({:?} == {:?}. {:?}", i, mod3(i), mod3(i) == (i % 3));
    // }
}
