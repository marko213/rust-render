use std::ops::{Add, Mul, Sub, SubAssign, AddAssign, MulAssign, Neg, Deref, DerefMut, Index, IndexMut, Div};
use std::fmt;
use std::any::type_name;

// This trait basically captures "yes this is a number that you can do all the math with"
// which should extend to any floating-point implementation
pub trait GoodNum : std::fmt::Display + Copy + Add<Output = Self>
    + AddAssign + Sub<Output = Self> + SubAssign + Mul<Output = Self> 
    + MulAssign + Neg<Output = Self> + Div<Output = Self> {
	const ZERO : Self;
	const ONE : Self;

	fn gn_sqrt(self) -> Self;
    }

impl GoodNum for f32 {
    const ZERO : f32 = 0.0;
    const ONE : f32 = 1.0;

    fn gn_sqrt(self) -> f32 {
	self.sqrt()
    }
}

impl GoodNum for f64 {
    const ZERO : f64 = 0.0;
    const ONE : f64 = 1.0;

    fn gn_sqrt(self) -> f64 {
	self.sqrt()
    }
}

// Help create matrices simply (e.g. mat![[1], [2]])
#[macro_export]
macro_rules! mat {
    [$($x:expr),*$(,)?] => (Mat::from([$($x),*]))
}

/// An N by M matrix
#[derive(Clone, Copy, PartialEq, Eq)]
pub struct Mat<const N : usize, const M : usize, T : GoodNum> {
    data : [[T; M]; N]
}

impl<const N : usize, const M : usize, T : GoodNum> Default for Mat<N, M, T> {
    fn default() -> Mat<N, M, T> {
	Mat::ZERO
    }
}

impl<const N : usize, const M : usize, T : GoodNum> DerefMut for Mat<N, M, T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
	&mut self.data
    }
}

impl<const N : usize, const M : usize, T : GoodNum> Deref for Mat<N, M, T> {
    type Target = [[T; M]; N];

    fn deref(&self) -> &Self::Target {
	&self.data
    }
}

impl<const N : usize, const M : usize, T : GoodNum> Mat<N, M, T> {
    pub const ZERO : Self = Self{data: [[T::ZERO; M]; N]};
    pub const ONE : Self = Self{data: [[T::ONE; M]; N]};

    pub const fn new(data : [[T; M]; N]) -> Self {
	Self { data }
    }

    pub fn slice<const P : usize, const Q : usize>(&self) -> Mat<P, Q, T> {
	let mut r = Mat::<P, Q, T>::ZERO;
	for y in 0..N.min(P) {
	    for x in 0..M.min(Q) {
		r.data[y][x] = self.data[y][x];
	    }
	}
	r
    }

    pub fn transpose(&self) -> Mat<M, N, T> {
	let mut r = Mat::<M, N, T>::ZERO;
	for y in 0..N {
	    for x in 0..M {
		r.data[x][y] = self.data[y][x];
	    }
	}
	r
    }

    pub fn t(&self) -> Mat<M, N, T> {
	self.transpose()
    }
    
    pub fn dot(self, other : Self) -> T {
	let mut sum = T::ZERO;
	for y in 0..N {
	    for x in 0..M {
		sum += self.data[y][x] * other.data[y][x];
	    }
	}
	sum
    }

    pub fn sum(self) -> T {
	let mut sum = T::ZERO;
	for y in 0..N {
	    for x in 0..M {
		sum += self.data[y][x];
	    }
	}
	sum
    }

    pub fn dot_matrix(mut self, other : Mat<N, M, T>) -> Mat<N, M, T> {
	for y in 0..N {
	    for x in 0..M {
		self.data[y][x] *= other.data[y][x];
	    }
	}
	self
    }
}

impl<const N : usize, const M : usize, T : GoodNum> From<[[T; M]; N]> for Mat<N, M, T> {
    fn from(data : [[T; M]; N]) -> Self {
	Self { data }
    }
}

impl<const N : usize, const M : usize, T : GoodNum> From<Mat<N, M, T>> for Vec<[T; M]> {
    fn from(mat : Mat<N, M, T>) -> Vec<[T; M]> {
	mat.data.into_iter().collect()
    }
}

impl<const N : usize, const M : usize, T : GoodNum> std::fmt::Debug for Mat<N, M, T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
	write!(f, "{} {{[", type_name::<Self>().split("::").last().unwrap_or("??"))?;
	if N > 0 && M > 0 {
	    if N > 1 {
		write!(f, "\n")?;
	    }
	    
	    // Gather the formatted numbers
	    let precision = f.precision().unwrap_or(3);
	    let nums : Vec<String>  = self.data.iter().map(|row| row.iter()).flatten()
		.map(|val| format!("{:.precision$}", val)).collect();
	    let width = nums.iter().map(|s| s.len()).max().unwrap() + 1;
	    
	    for y in 0..N {
		write!(f, "  [")?;
		for x in 0..M {
		    write!(f, "{:>width$}", nums[y * M + x])?;
		    if x + 1 < M {
			write!(f, ",")?;
		    }
		}
		write!(f, "]")?;
		if y + 1 < N {
		    write!(f, ",\n")?;
		}
	    }
	}
	write!(f, "]}}")
    }
}


// Operations

// Matrix multiplication
impl<const N : usize, const M : usize, const O : usize, T : GoodNum> Mul<Mat<M, O, T>> for Mat<N, M, T> {
    type Output = Mat<N, O, T>;

    fn mul(self, other : Mat<M, O, T>) -> Self::Output {
	let mut r = Self::Output::ZERO;
	for i in 0..N {
	    for j in 0..O {
		for k in 0..M {
		    r.data[i][j] += self.data[i][k] * other.data[k][j];
		}
	    }
	}
	r
    }
}

// Pointwise addition
impl<const N : usize, const M : usize, T : GoodNum> AddAssign<Mat<N, M, T>> for Mat<N, M, T> {
   fn add_assign(&mut self, other : Self) {
	for i in 0..N {
	    for j in 0..M {
		self.data[i][j] += other.data[i][j];
	    }
	}
    }
}

impl<const N : usize, const M : usize, T : GoodNum> Add<Self> for Mat<N, M, T> {
    type Output = Mat<N, M, T>;
    
    fn add(mut self, other : Self) -> Self::Output {
	self += other;
	self
    }
}

impl<const N : usize, const M : usize, T : GoodNum> SubAssign<Mat<N, M, T>> for Mat<N, M, T> {
   fn sub_assign(&mut self, other : Self) {
	for i in 0..N {
	    for j in 0..M {
		self.data[i][j] -= other.data[i][j];
	    }
	}
    }
}

impl<const N : usize, const M : usize, T : GoodNum> Sub<Self> for Mat<N, M, T> {
    type Output = Mat<N, M, T>;
    
    fn sub(mut self, other : Self) -> Self::Output {
	self -= other;
	self
    }
}

// Scalar multiplication
impl<const N : usize, const M : usize, T : GoodNum> MulAssign<T> for Mat<N, M, T> {
   fn mul_assign(&mut self, v : T) {
	for i in 0..N {
	    for j in 0..M {
		self.data[i][j] *= v;
	    }
	}
    }
}

impl<const N : usize, const M : usize, T : GoodNum> Mul<T> for Mat<N, M, T> {
    type Output = Mat<N, M, T>;

    fn mul(mut self, v : T) -> Self::Output {
	self *= v;
	self
    }
}

impl<const N : usize, const M : usize, T : GoodNum> Neg for Mat<N, M, T> {
    type Output = Mat<N, M, T>;

    fn neg(mut self) -> Self::Output {
	for i in 0..N {
	    for j in 0..M {
		let x = &mut self.data[i][j];
		*x = -*x;
	    }
	}
	self
    }
}

// Square matrices
pub type SqMat<const N : usize, T> = Mat<N, N, T>;

impl<const N : usize, T : GoodNum> SqMat<N, T> {
    
}

pub type V<const N : usize, T> = Mat<N, 1, T>;

impl<const N : usize, T : GoodNum> V<N, T> {
    pub fn sq_magnitude(self) -> T {
	self.dot(self)
    }

    pub fn magnitude(self) -> T {
	self.sq_magnitude().gn_sqrt()
    }
    
    pub fn normalize(self) -> Self {
	self * (T::ONE / self.magnitude())
    }
}

impl<const N : usize, T : GoodNum> From<[T;N]> for V<N, T> {
    fn from(data : [T;N]) -> Self {
	let mut data2 = [[T::ZERO;1];N];
	for i in 0..N {
	    data2[i][0] = data[i];
	}
	Self {data: data2}
    }
}

impl<const N : usize, T : GoodNum> Index<usize> for V<N, T> {
    type Output = T;

    fn index(&self, i : usize) -> &T {
	&self.data[i][0]
    }
}

impl<const N : usize, T : GoodNum> IndexMut<usize> for V<N, T> {
    fn index_mut(&mut self, i : usize) -> &mut T {
	&mut self.data[i][0]
    }
}

pub type V3 = V<3, f32>;

impl V3 {
    pub const DIR_100 : V3 = Mat::new([[1.0], [0.0], [0.0]]);
    pub const DIR_010 : V3 = Mat::new([[0.0], [1.0], [0.0]]);
    pub const DIR_001 : V3 = Mat::new([[0.0], [0.0], [1.0]]);
    
    pub fn cross(self, other : V3) -> V3 {
	mat![
	    self[1] * other[2] - self[2] * other[1],
	    self[2] * other[0] - self[0] * other[2],
	    self[0] * other[1] - self[1] * other[0]
	]
    }
}

// Dynamic matrix
#[derive(Clone)]
pub struct DMat<T> {
    pub data : Vec<T>,
    pub size : [usize; 2]
}

impl<T : Clone> DMat<T> {
    pub fn new(size : [usize; 2], val : T) -> DMat<T> {
	let v = Vec::with_capacity(size[0] * size[1]);
	let mut m = DMat {
	    data: v,
	    size: [0, 0]
	};
	m.resize(size, val);
	m
    }

    pub fn resize(&mut self, size : [usize; 2], val : T) {
	self.data.resize(size[0] * size[1], val);
	self.size = size;
    }

    pub fn fill(&mut self, val : T) {
	self.data.fill(val);
    }
}

impl<T> Index<usize> for DMat<T> {
    type Output = [T];

    fn index(&self, y : usize) -> &[T] {
	let s = self.size[1];
	&self.data[(y*s)..((y + 1) * s)]
    }
}

impl<T> IndexMut<usize> for DMat<T> {
    fn index_mut(&mut self, y : usize) -> &mut [T] {
	let s = self.size[1];
	&mut self.data[(y*s)..((y + 1) * s)]
    }
}

impl<T : AddAssign<T> + Copy> AddAssign<&DMat<T>> for DMat<T> {
    fn add_assign(&mut self, other : &DMat<T>) {
	assert!(self.data.len() == other.data.len());

	for i in 0..self.data.len() {
	    self.data[i] += other.data[i];
	}
    }
}
