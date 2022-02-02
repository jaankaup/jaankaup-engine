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
