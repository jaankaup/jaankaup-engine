use jaankaup_core::common_functions::*;

fn main() {
    let arg001 = 4;
    let arg002 = 2;
    println!("udiv_up_safe32({}, {}) == {}", arg001, arg002, udiv_up_safe32(arg001, arg002));

    let arg003 = 3;
    let arg004 = 6;
    let arg005 = true;
    let b001 = set_bit_to(arg003, arg004, arg005);
    println!("set_bit_to({}, {}, {:?} == {};", arg003, arg004, arg005, format!("{:b}, original_value", b001));

    let arg006 = 0b101111;
    let arg007 = 2;
    let arg008 = 4;
    let b002 = get_range(arg006, arg007, arg008);

    println!("get_range({}, {}, {} == {};", format!("{:b}", arg006), arg007, arg008, format!("{:b}, original_value", b002));
}
