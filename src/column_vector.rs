use std::fmt::{Display, Formatter};
use std::ops::{Index, IndexMut, Sub, SubAssign};
use ndarray::{Array2, ArrayView};

/// Create a **[`ColumnVector`]** from a sequence of numbers.
/// This macro automatically casts all input values to floats.
///
/// # Example
/// ```
/// use ndarray::{array, Array2};
/// use graymat::column_vector::ColumnVector;
/// use graymat::cvec;
///
/// let cv1 = cvec![-12, 3.14, 2.71, 42];
/// let cv2 = ColumnVector::from(&Array2::from(array![[-12.0, 3.14, 2.71, 42.0]]));
///
/// assert_eq!(cv1 == cv2, true);
/// ```
#[macro_export]
macro_rules! cvec {
    ($($x:expr),*) => {{
        ColumnVector::from(&array![[$($x as f32,)*]])
    }};
}

pub struct ColumnVector {
    data: Array2<f32>
}

impl ColumnVector {
    pub fn from(data: &Array2<f32>) -> Self {
        let mut data_copy: Array2<f32> = data.to_owned();
        if data_copy.shape()[1] != 1 {
            let rows: usize = data_copy.shape()[0] * data_copy.shape()[1];
            data_copy = data_copy.to_shape((rows, 1)).unwrap().to_owned(); // TODO fix unwrap to handle errors
            return ColumnVector { data: data_copy };
        }
        return Self { data: data_copy };
    }

    pub fn from_vec(data: Vec<f32>) -> Self {
        return ColumnVector::from(&Array2::from_shape_vec((1, data.len()), data).unwrap());
    }

    pub fn empty() -> Self {
        return ColumnVector::zeros(0);
    }

    pub fn zeros(size: usize) -> Self {
        return Self { data: Array2::zeros((size, 1))}
    }

    pub fn ones(size: usize) -> Self {
        return Self { data: Array2::ones((size, 1)) };
    }

    pub fn size(&self) -> usize {
        return self.data.len();
    }

    pub fn push(&mut self, x: f32) {
        self.data.push_row(ArrayView::from(&[x])).unwrap();
    }

    pub fn get_data(&self) -> &Array2<f32> {
        return &self.data;
    }

    pub fn get_data_mut(&mut self) -> &mut Array2<f32> {
        return &mut self.data;
    }

    pub fn sum(self) -> f32 {
        let mut sum = 0.0;
        for element in self.data.iter() {
            sum += *element;
        }
        return sum;
    }
}

impl Index<usize> for ColumnVector {
    type Output = f32;

    fn index(&self, index: usize) -> &Self::Output {
        return &self.data[[index, 0]];
    }
}

impl IndexMut<usize> for ColumnVector {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        return &mut self.data[[index, 0]];
    }
}

impl Display for ColumnVector {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}\n", self.data.to_string())
    }
}

impl SubAssign for ColumnVector {
    fn sub_assign(&mut self, rhs: Self) {
        self.data.sub_assign(rhs.get_data());
    }
}

impl Sub for ColumnVector {
    type Output = ColumnVector;

    fn sub(self, rhs: Self) -> Self::Output {
        return ColumnVector::from(&(self.data.to_owned() - rhs.get_data().to_owned()));
    }
}

impl Clone for ColumnVector {
    fn clone(&self) -> Self {
        return ColumnVector::from(&self.get_data());
    }
}

impl PartialEq<Self> for ColumnVector {
    fn eq(&self, other: &Self) -> bool {
        return self.data == other.get_data();
    }
}

impl Eq for ColumnVector {

}
