use graymat::utilities::math_utils::float_compare;

#[test]
fn relu_test() {
    let test_data = vec![(-10.2, 0.0),
                         (0.0, 0.0),
                         (100.0, 100.0),
                         (-0.112, 0.0),
                         (0.000012, 0.000012)];
    for t in test_data {
        assert_eq!(graymat::utilities::math_utils::relu(t.0), t.1);
    }
}

#[test]
fn relu_prime_test() {
    let test_data = vec![(-10.2, 0.0),
                         (0.0, 0.0),
                         (100.0, 1.0),
                         (-0.112, 0.0),
                         (0.000012, 1.0)];
    for t in test_data {
        assert_eq!(graymat::utilities::math_utils::relu_prime(t.0), t.1);
    }
}

#[test]
fn tanh_test() {
    let test_data = vec![(-2.0, 0.0706508),
                         (-0.5, 0.786448),
                         (0.0, 1.0),
                         (0.5, 0.786448),
                         (1.0, 0.419974)];
    let precision: u8 = 5;
    for t in test_data {
        assert_eq!(float_compare(graymat::utilities::math_utils::tanh_prime(t.0), t.1, precision), true);
    }
}
