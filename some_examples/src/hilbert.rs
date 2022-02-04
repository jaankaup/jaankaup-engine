use jaankaup_core::common_functions::{to_hilbert_index, g, gc, e, f, t, d, t_inv};

fn main() {

    let mut map = Vec::<u32>::new();
    for i in 0..4 {
    for j in 0..4 {
    for k in 0..4 {
        
        let index = to_hilbert_index([i,j,k]);
        println!("to_hilbert_index({},{},{}) == {:?}", i, j, k, index);
        map.push(index);
    }}};
    map.sort();
    for i in map.iter() {
        println!("{}", i);
    }

    // const N: u32 = 3;

    // print!("Only g(i)-th bit changes form gc(i) to gc(i+1)");
    // for i in 0..(1 << (N-1)) {
    //     assert!(gc(i) ^ (1 << g(i)) == gc(i+1), "Fail");
    // }
    // println!("  OK");

    // print!("T sends e(i) to 0 and f(i) to 2**(N-1)");
    // for i in 0..(1 << N) {
    //     assert!(t(e(i), d(i) as i32, e(i)) == 0, "Fail");
    //     assert!(t(e(i), d(i) as i32, f(i)) == (1 << (N-1)), "Fail");
    // }
    // println!("  OK");

    // print!("e(i) reflected in direction d(i) is f(i)");
    // for i in 0..(1 << N) {
    //     assert!(e(i) ^ (1 << d(i)) == f(i), "Fail");
    // }
    // println!("  OK");

    // print!("t_inv composet with t (and visa versa) is the identity operator.");
    // for i in 0..(1 << N) {
    // for b in 0..(1 << N) {
    //     assert!(t(e(i), d(i) as i32, t_inv(e(i), d(i), b)) == b, "Fail");
    //     assert!(t_inv(e(i), d(i), t(e(i), d(i) as i32, b)) == b, "Fail");
    // }};
    // println!("  OK");

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
}
