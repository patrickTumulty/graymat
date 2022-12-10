
/// Copy string into byte array
///
/// * `s` - string
/// * `bytes` - byte array to copy s into
pub fn copy_string_into_byte_array(s: &str, bytes: &mut [u8]) {
    s.bytes()
     .zip(bytes.iter_mut())
     .for_each(|(b, ptr)| *ptr = b);
}
