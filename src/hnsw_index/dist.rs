//! Some standard distances as L1, L2, Cosine, Jaccard, Hamming
//! and a structure to enable the user to implement its own distances.
//! For the heavily used case (f32) we provide simd avx2 implementation.

/// The trait describing distance.
/// For example for the L1 distance
///
/// pub struct DistL1;
///
/// implement Distance<f32> for DistL1 {
/// }
///
///
/// The L1 and Cosine distance are implemented for u16, i32, i64, f32, f64
use std::os::raw::*;

use num_traits::float::*;
use packed_simd_2::{f32x16, f64x8};

#[allow(unused)]
enum DistKind {
    DistL1(String),
    DistL2(String),
    /// This is the same as Cosine dist but all data L2-normalized to 1.
    DistDot(String),
    DistCosine(String),
    DistHamming(String),
    DistJaccard(String),
    DistHellinger(String),
    DistJeffreys(String),
    DistJensenShannon(String),
    /// To store a distance defined by a C pointer function
    DistCFnPtr,
    /// Distance defined by a closure
    DistFn,
    /// Distance defined by a fn Rust pointer
    DistPtr,
    DistLevenshtein(String),
    /// used only with reloading only graph data from a previous dump
    DistNoDist(String),
}

/// This is the basic Trait describing a distance. The structure Hnsw can be instantiated by
/// anything satisfying this Trait. The crate provides implmentations for L1, L2 , Cosine, Jaccard,
/// Hamming. For other distances implement the trait possibly with the newtype pattern
pub trait Distance<T: Send + Sync> {
    fn eval(&self, va: &[T], vb: &[T]) -> f32;
}

/// Special forbidden computation distance. It is associated to a unit NoData structure
/// This is a special structure used when we want to only reload the graph from a previous
/// computation possibly from an foreign language (and we do not have access to the original type of
/// data from the foreign language).
#[derive(Default)]
pub struct NoDist;

impl<T: Send + Sync> Distance<T> for NoDist {
    fn eval(&self, _va: &[T], _vb: &[T]) -> f32 {
        log::error!("panic error : cannot call eval on NoDist");
        panic!("cannot call distance with NoDist");
    }
} // end impl block for NoDist

/// L1 distance : implemented for i32, f64, i64, u32 , u16 , u8 and with Simd avx2 for f32
#[derive(Default)]
pub struct DistL1;

macro_rules! implementL1Distance (
    ($ty:ty) => (
        impl Distance<$ty> for DistL1 {
            fn eval(&self, va: &[$ty], vb: &[$ty]) -> f32 {
            // RUSTFLAGS = "-C opt-level=3 -C target-cpu=native"
                va.iter().zip(vb.iter()).map(|t| (*t.0 as f32 - *t.1 as f32).abs()).sum()
            } // end of compute
        } // end of impl block
    )  // end of pattern matching
);

macro_rules! simd_l1_distance (
    ($data_type:ident, $simd_type:ident, $size:expr) => {
        #[allow(unreachable_code)]
        impl Distance<$data_type> for DistL1 {
            fn eval(&self, va: &[$data_type], vb: &[$data_type]) -> f32 {
                #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
                {
                    let size: usize = va.len() - (va.len() % $size);

                    let c: $data_type = va
                        .chunks_exact($size)
                        .map($simd_type::from_slice_unaligned)
                        .zip(vb.chunks_exact($size).map($simd_type::from_slice_unaligned))
                        .map(|(a, b)| (a - b).abs())
                        .sum::<$simd_type>()
                        .sum();

                    let d: $data_type = va[size..]
                        .iter()
                        .zip(&vb[size..])
                        .map(|(p, q)| (p - q).abs())
                        .sum();

                    return (c + d) as f32;
                }

                va.iter()
                    .zip(vb.iter())
                    .map(|t| (*t.0 as f32 - *t.1 as f32).abs())
                    .sum()
            }
        }
    }
);

implementL1Distance!(i64);
implementL1Distance!(i32);
implementL1Distance!(u32);
implementL1Distance!(u16);
implementL1Distance!(u8);
implementL1Distance!(i8);
simd_l1_distance!(f64, f64x8, 8);
simd_l1_distance!(f32, f32x16, 16);

//========================================================================

/// L2 distance : implemented for i32, f64, i64, u32 , u16 , u8 and with Simd avx2 for f32
#[derive(Default)]
pub struct DistL2;

macro_rules! implementL2Distance (
    ($ty:ty) => (
        impl Distance<$ty> for DistL2 {
            fn eval(&self, va: &[$ty], vb: &[$ty]) -> f32 {
                let norm: f32 = va.iter().zip(vb.iter()).map(|t| (*t.0 as f32 - *t.1 as f32) * (*t.0 as f32 - *t.1 as f32)).sum();
                norm.sqrt()
            } // end of compute
        } // end of impl block
    )  // end of pattern matching
);

macro_rules! simd_l2_distance (
    ($data_type:ident, $simd_type:ident, $size:expr) => {
        #[allow(unreachable_code)]
        impl Distance<$data_type> for DistL2 {
            fn eval(&self, va: &[$data_type], vb: &[$data_type]) -> f32 {
                #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
                {
                    let size: usize = va.len() - (va.len() % $size);

                    let c: $data_type = va
                        .chunks_exact($size)
                        .map($simd_type::from_slice_unaligned)
                        .zip(vb.chunks_exact($size).map($simd_type::from_slice_unaligned))
                        .map(|(a, b)| {
                            let c = a - b;
                            c * c
                        })
                        .sum::<$simd_type>()
                        .sum();

                    let d: $data_type = va[size..]
                        .iter()
                        .zip(&vb[size..])
                        .map(|(p, q)| (p - q).powi(2))
                        .sum();

                    return (c + d) as f32;
                }

                va.iter()
                    .zip(vb.iter())
                    .map(|(p, q)| (*p as f32 - *q as f32).powi(2))
                    .sum()
            }
        }
    }
);

implementL2Distance!(i64);
implementL2Distance!(i32);
implementL2Distance!(u32);
implementL2Distance!(u16);
implementL2Distance!(u8);
implementL2Distance!(i8);
simd_l2_distance!(f64, f64x8, 8);
simd_l2_distance!(f32, f32x16, 16);

//=========================================================================

/// Cosine distance : implemented for f32, f64, i64, i32 , u16
#[derive(Default)]
pub struct DistCosine;

macro_rules! implementCosDistance(
    ($ty:ty) => (
        impl Distance<$ty> for DistCosine  {
            fn eval(&self, va: &[$ty], vb: &[$ty]) -> f32 {
                let res = va.iter().zip(vb.iter()).map(|t| ((*t.0 * *t.1) as f64, (*t.0 * *t.0) as f64, (*t.1 * *t.1) as f64)).
                    fold((0., 0., 0.), |acc , t| (acc.0 + t.0, acc.1 + t.1, acc.2 + t.2));

                let dist: f32;
                if res.1 > 0.0 && res.2 > 0.0 {
                    let dist_unchecked = 1. - res.0 / (res.1 * res.2).sqrt();
                    assert!(dist_unchecked >= - 0.00002);
                    dist = dist_unchecked.max(0.) as f32;
                }
                else {
                    dist = 0.;
                }

                dist
            } // end of function
        } // end of impl block
    ) // end of matching
);

implementCosDistance!(i64);
implementCosDistance!(i32);
implementCosDistance!(u16);

#[allow(unreachable_code)]
fn dot_f64(va: &[f64], vb: &[f64]) -> f64 {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        let size: usize = va.len() - (va.len() % 8);

        let c: f64 = va
            .chunks_exact(8)
            .map(f64x8::from_slice_unaligned)
            .zip(vb.chunks_exact(8).map(f64x8::from_slice_unaligned))
            .map(|(a, b)| a * b)
            .sum::<f64x8>()
            .sum();

        let d: f64 = va[size..].iter().zip(&vb[size..]).map(|(p, q)| p * q).sum();

        return c + d;
    }

    va.iter().zip(vb).map(|(p, q)| p * q).sum::<f64>()
}

#[allow(unreachable_code)]
fn dot_f32(va: &[f32], vb: &[f32]) -> f32 {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        let size: usize = va.len() - (va.len() % 16);

        let c: f32 = va
            .chunks_exact(16)
            .map(f32x16::from_slice_unaligned)
            .zip(vb.chunks_exact(16).map(f32x16::from_slice_unaligned))
            .map(|(a, b)| a * b)
            .sum::<f32x16>()
            .sum();

        let d: f32 = va[size..].iter().zip(&vb[size..]).map(|(p, q)| p * q).sum();

        return c + d;
    }

    va.iter().zip(vb).map(|(p, q)| p * q).sum::<f32>()
}

#[allow(unreachable_code)]
fn dot_i8(va: &[i8], vb: &[i8]) -> i32 {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    #[target_feature(enable = "avx2")]
    unsafe fn compute_r_dx_dy_avx2(x: &[i8], y: &[i8]) -> i32 {
        compute_r_dx_dy_fallback(x, y)
    }

    #[inline(always)]
    fn compute_r_dx_dy_fallback(x: &[i8], y: &[i8]) -> i32 {
        let mut r: i32 = 0i32;

        for (xi, yi) in x
            .iter()
            .map(|&xi| i32::from(xi))
            .zip(y.iter().map(|&yi| i32::from(yi)))
        {
            r += xi * yi;
        }

        r
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if is_x86_feature_detected!("avx2") {
            return unsafe { compute_r_dx_dy_avx2(va, vb) };
        }
    }

    compute_r_dx_dy_fallback(va, vb)
}

impl Distance<f64> for DistCosine {
    fn eval(&self, va: &[f64], vb: &[f64]) -> f32 {
        let ab: f64 = dot_f64(va, vb);
        let a: f64 = dot_f64(va, va).sqrt();
        let b: f64 = dot_f64(vb, vb).sqrt();

        let dist: f32 = if a > 0.0 && b > 0.0 {
            let d: f64 = 1.0 - ab / (a * b);
            assert!(d >= -0.00002);
            d.max(0.0) as f32
        } else {
            0.0f32
        };
        dist
    }
}

impl Distance<f32> for DistCosine {
    fn eval(&self, va: &[f32], vb: &[f32]) -> f32 {
        let ab: f32 = dot_f32(va, vb);
        let a: f32 = dot_f32(va, va).sqrt();
        let b: f32 = dot_f32(vb, vb).sqrt();

        let dist: f32 = if a > 0.0 && b > 0.0 {
            let d: f32 = 1.0 - ab / (a * b);
            assert!(d >= -0.00002);
            d.max(0.0) as f32
        } else {
            0.0
        };
        dist
    }
}

//=========================================================================

/// This is essentially the Cosine distance but we suppose
/// all vectors (graph construction and request vectors have been l2 normalized to unity
/// BEFORE INSERTING in  HNSW!.   
/// No control is made, so it is the user responsability to send normalized vectors
/// everywhere in inserting and searching.
///
/// In large dimensions (hundreds) this pre-normalization spare cpu time.  
/// At low dimensions (a few ten's there is not a significant gain).  
/// This distance makes sense only for f16, f32 or f64
/// We provide for avx2 implementations for f32 that provides consequent gains
/// in large dimensions

#[derive(Default)]
pub struct DistDot;

#[allow(unused)]
macro_rules! implementDotDistance(
    ($ty:ty) => (
        impl Distance<$ty> for DistDot  {
            fn eval(&self, va: &[$ty], vb: &[$ty]) -> f32 {
                let zero: f32 = 0f32;
                let dot = va.iter().zip(vb.iter()).map(|t| (*t.0 * *t.1) as f32).fold(0., |acc , t| (acc + t));
                assert(dot <= 1.);
                return 1. - dot;
            }  // end of function
        }  // end of impl block
    )  // end of matching
);

impl Distance<f64> for DistDot {
    fn eval(&self, va: &[f64], vb: &[f64]) -> f32 {
        let dot: f64 = 1.0 - dot_f64(va, vb);
        assert!(dot >= -0.000002);
        dot.max(0.) as f32
    } // end of eval
}

impl Distance<f32> for DistDot {
    fn eval(&self, va: &[f32], vb: &[f32]) -> f32 {
        let dot: f32 = 1.0 - dot_f32(va, vb);
        assert!(dot >= -0.000002);
        dot.max(0.) as f32
    } // end of eval
}

impl Distance<i8> for DistDot {
    fn eval(&self, va: &[i8], vb: &[i8]) -> f32 {
        let dot: f32 = 1.0 - (dot_i8(va, vb) as f32);
        dot.max(0.)
    } // end of eval
}

pub fn l2_normalize(va: &mut [f32]) {
    let l2norm: f32 = va.iter().map(|t| (*t * *t) as f32).sum::<f32>().sqrt();
    if l2norm > 0. {
        for i in 0..va.len() {
            va[i] = va[i] / l2norm;
        }
    }
}

//=======================================================================================

/// A structure to compute Hellinger distance between probalilities.
/// Vector must be >= 0 and normalized to 1.
///   
/// The distance computation does not check that
/// and in fact simplifies the expression of distance assuming vectors are positive and L1
/// normalised to 1. The user must enforce these conditions before  inserting otherwise results will
/// be meaningless at best or code will panic!
///
/// For f32 a simd implementation is provided if avx2 is detected.
#[derive(Default)]
pub struct DistHellinger;

// default implementation
macro_rules! implementHellingerDistance (
    ($ty:ty) => (
        impl Distance<$ty> for DistHellinger {
            fn eval(&self, va: &[$ty], vb: &[$ty]) -> f32 {
            // RUSTFLAGS = "-C opt-level=3 -C target-cpu=native"
            // to // by rayon
                let mut dist = va.iter().zip(vb.iter()).map(|t| ((*t.0).sqrt() * (*t.1).sqrt()) as f32).fold(0., |acc , t| (acc + t*t));
                dist = (1. - dist).sqrt();
                dist
            } // end of compute
        } // end of impl block
    )  // end of pattern matching
);

implementHellingerDistance!(f64);

impl Distance<f32> for DistHellinger {
    fn eval(&self, va: &[f32], vb: &[f32]) -> f32 {
        let mut dist: f32 = va
            .iter()
            .zip(vb.iter())
            .map(|t: (&f32, &f32)| ((*t.0).sqrt() * (*t.1).sqrt()) as f32)
            .fold(0., |acc: f32, t: f32| (acc + t));
        // if too far away from >= panic else reset!
        assert!(1. - dist >= -0.000001);
        dist = (1. - dist).max(0.).sqrt();
        dist
    } // end of eval
}

//=======================================================================================

/// A structure to compute Jeffreys divergence between probalilities.
/// If p and q are 2 probability distributions
/// the "distance" is computed as:
///   sum (p\[i\] - q\[i\]) * ln(p\[i\]/q\[i\])
///
/// To take care of null probabilities in the formula we use  max(x\[i\],1.E-30)
/// for x = p and q in the log compuations
///   
/// Vector must be >= 0 and normalized to 1!  
/// The distance computation does not check that.
/// The user must enforce these conditions before inserting in the hnws structure,
/// otherwise results will be meaningless at best or code will panic!
///
/// For f32 a simd implementation is provided if avx2 is detected.
#[derive(Default)]
pub struct DistJeffreys;

const M_MIN: f32 = 1.0e-30;

// default implementation
macro_rules! implementJeffreysDistance (
    ($ty:ty) => (

    impl Distance<$ty> for DistJeffreys {
        fn eval(&self, va: &[$ty], vb: &[$ty]) -> f32 {
        // RUSTFLAGS = "-C opt-level=3 -C target-cpu=native"
        let dist = va.iter().zip(vb.iter()).map(|t| (*t.0 - *t.1) * ((*t.0).max(M_MIN as f64)/ (*t.1).max(M_MIN as f64)).ln() as f64).fold(0., |acc , t| (acc + t*t));
        dist as f32
        } // end of compute
    } // end of impl block
    )  // end of pattern matching
);

implementJeffreysDistance!(f64);

impl Distance<f32> for DistJeffreys {
    fn eval(&self, va: &[f32], vb: &[f32]) -> f32 {
        let dist: f32 = va
            .iter()
            .zip(vb.iter())
            .map(|t: (&f32, &f32)| {
                (*t.0 - *t.1) * ((*t.0).max(M_MIN) / (*t.1).max(M_MIN)).ln() as f32
            })
            .fold(0., |acc, t| (acc + t));
        dist
    } // end of eval
}

//=======================================================================================

/// Jensen-Shannon distance.  
/// It is defined as the **square root** of the  Jensen–Shannon divergence and is a metric.
/// Vector must be >= 0 and normalized to 1!
/// The distance computation does not check that.
#[derive(Default)]
pub struct DistJensenShannon;

macro_rules! implementDistJensenShannon (
    ($ty:ty) => (
        impl Distance<$ty> for DistJensenShannon {
            fn eval(&self, va: &[$ty], vb: &[$ty]) -> f32 {
                let mut dist = 0.;
                //
                assert_eq!(va.len(), vb.len());
                //
                for i in 0..va.len() {
                    let mean_ab = 0.5 * (va[i] + vb[i]);
                    if va[i] > 0. {
                        dist += va[i] * (va[i]/mean_ab).ln();
                    }
                    if vb[i] > 0. {
                        dist += vb[i] * (vb[i]/mean_ab).ln();
                    }
                }
                (0.5 * dist).sqrt() as f32
            } // end eval
        }  // end impl Distance<$ty>
    )  // end of pattern matching on ty
);

implementDistJensenShannon!(f64);
implementDistJensenShannon!(f32);

//=======================================================================================

/// Hamming distance. Implemented for u8, u16, u32, i32 and i16
/// The distance returned is normalized by length of slices, so it is between 0. and 1.  
///
/// A special implementation for f64 is made but exclusively dedicated to SuperMinHash usage in crate [probminhash](https://crates.io/crates/probminhash).  
/// It could be made generic with the PartialEq implementation for f64 and f32 in unsable source of
/// Rust
#[derive(Default)]
pub struct DistHamming;

macro_rules! implementHammingDistance (
    ($ty:ty) => (

    impl Distance<$ty> for DistHamming  {
        fn eval(&self, va: &[$ty], vb: &[$ty]) -> f32 {
        // RUSTFLAGS = "-C opt-level=3 -C target-cpu=native"
            assert_eq!(va.len(), vb.len());
            let norm : f32 = va.iter().zip(vb.iter()).filter(|t| t.0 != t.1).count() as f32;
            norm / va.len() as f32
        } // end of compute
    } // end of impl block
    )  // end of pattern matching
);

impl Distance<i32> for DistHamming {
    fn eval(&self, va: &[i32], vb: &[i32]) -> f32 {
        assert_eq!(va.len(), vb.len());
        let dist: f32 = va
            .iter()
            .zip(vb.iter())
            .filter(|t: &(&i32, &i32)| t.0 != t.1)
            .count() as f32;
        dist / va.len() as f32
    } // end of eval
} // end implementation Distance<i32>

/// This implementation is dedicated to SuperMinHash algorithm in crate [probminhash](https://crates.io/crates/probminhash).  
/// Could be made generic with unstable source as there is implementation of PartialEq for f64
impl Distance<f64> for DistHamming {
    fn eval(&self, va: &[f64], vb: &[f64]) -> f32 {
        assert_eq!(va.len(), vb.len());
        let dist: usize = va
            .iter()
            .zip(vb.iter())
            .filter(|t: &(&f64, &f64)| t.0 != t.1)
            .count();
        (dist as f64 / va.len() as f64) as f32
    } // end of eval
} // end implementation Distance<f64>

/// This implementation is dedicated to SuperMinHash algorithm in crate [probminhash](https://crates.io/crates/probminhash).  
/// Could be made generic with unstable source as there is implementation of PartialEq for f32
impl Distance<f32> for DistHamming {
    fn eval(&self, va: &[f32], vb: &[f32]) -> f32 {
        // in fact simd comparaison seems slower than simple iter
        assert_eq!(va.len(), vb.len());
        let dist: usize = va
            .iter()
            .zip(vb.iter())
            .filter(|t: &(&f32, &f32)| t.0 != t.1)
            .count();
        (dist as f64 / va.len() as f64) as f32
    } // end of eval
} // end implementation Distance<f32>

implementHammingDistance!(u64);
implementHammingDistance!(u32);
implementHammingDistance!(u16);
implementHammingDistance!(u8);
implementHammingDistance!(i16);
implementHammingDistance!(i8);

//====================================================================================
//   Jaccard Distance

/// Jaccard distance. Implemented for u8, u16 , u32.
#[derive(Default)]
pub struct DistJaccard;

// contruct a 2-uple accumulator that has sum of max in first component , and sum of min in 2
// component stay in integer as long as possible
// Note : summing u32 coming from hash values can overflow! We must go up to u64 for additions!
macro_rules! implementJaccardDistance (
    ($ty:ty) => (

    impl Distance<$ty> for DistJaccard  {
        fn eval(&self, va: &[$ty], vb: &[$ty]) -> f32 {
            let (max, min): (u64, u64) = va.iter()
                .zip(vb.iter())
                .fold((0u64, 0u64), |acc, t|
                    if t.0 > t.1 {
                        (acc.0 + *t.0 as u64, acc.1 + *t.1 as u64)
                    }
                    else {
                        (acc.0 + *t.1 as u64, acc.1 + *t.0 as u64)
                    }
                );

            if max > 0 {
                let dist = 1. - (min  as f64) / (max as f64);
                assert!(dist >= 0.);
                dist as f32
            }
            else {
                0.
            }
        } // end of compute
    } // end of impl block
    )  // end of pattern matching
);

implementJaccardDistance!(u32);
implementJaccardDistance!(u16);
implementJaccardDistance!(u8);

// ==========================================================================================

/// Levenshtein distance. Implemented for u16
#[derive(Default)]
pub struct DistLevenshtein;
impl Distance<u16> for DistLevenshtein {
    fn eval(&self, a: &[u16], b: &[u16]) -> f32 {
        let len_a: usize = a.len();
        let len_b: usize = b.len();
        if len_a < len_b {
            return self.eval(b, a);
        }

        // handle special case of 0 length
        if len_a == 0 {
            return len_b as f32;
        } else if len_b == 0 {
            return len_a as f32;
        }

        let len_b: usize = len_b + 1;

        let mut pre: usize;
        let mut tmp: usize;
        let mut cur: Vec<usize> = vec![0; len_b];

        // initialize string b
        for i in 1..len_b {
            cur[i] = i;
        }

        // calculate edit distance
        for (i, ca) in a.iter().enumerate() {
            // get first column for this row
            pre = cur[0];
            cur[0] = i + 1;
            for (j, cb) in b.iter().enumerate() {
                tmp = cur[j + 1];
                cur[j + 1] = std::cmp::min(
                    // deletion
                    tmp + 1,
                    std::cmp::min(
                        // insertion
                        cur[j] + 1,
                        // match or substitution
                        pre + if ca == cb { 0 } else { 1 },
                    ),
                );
                pre = tmp;
            }
        }

        cur[len_b - 1] as f32
    }
}

//=======================================================================================
//   Case of function pointers (cover Trait Fn , FnOnce ...)
// The book (Function item types):  " There is a coercion from function items to function pointers
// with the same signature  " The book (Call trait and coercions): "Non capturing closures can be
// coerced to function pointers with the same signature"

/// This type is for function with a C-API
/// Distances can be computed by such a function. It
/// takes as arguments the two (C, rust, julia) pointers to primitive type vectos and length
/// passed as a unsignedlonlong (64 bits) which is called c_ulonglong in Rust and Culonglong in
/// Julia
type DistCFnPtr<T> = extern "C" fn(*const T, *const T, len: c_ulonglong) -> f32;

/// A structure to implement Distance Api for type DistCFnPtr\<T\>,
/// i.e distance provided by a C function pointer.  
/// It must be noted that this can be used in Julia via the macro @cfunction
/// to define interactiveley a distance function , compile it on the fly and sent it
/// to Rust via the init_hnsw_{f32, i32, u16, u32, u8} function
/// defined in libext
pub struct DistCFFI<T: Copy + Clone + Sized + Send + Sync> {
    dist_function: DistCFnPtr<T>,
}

impl<T: Copy + Clone + Sized + Send + Sync> DistCFFI<T> {
    pub fn new(f: DistCFnPtr<T>) -> Self {
        DistCFFI { dist_function: f }
    }
}

impl<T: Copy + Clone + Sized + Send + Sync> Distance<T> for DistCFFI<T> {
    fn eval(&self, va: &[T], vb: &[T]) -> f32 {
        let len: usize = va.len();
        let ptr_a: *const T = va.as_ptr();
        let ptr_b: *const T = vb.as_ptr();
        let dist: f32 = (self.dist_function)(ptr_a, ptr_b, len as c_ulonglong);
        log::trace!("DistCFFI dist_function_ptr {:?} returning {:?} ", self.dist_function, dist);
        dist
    } // end of compute
} // end of impl block

//========================================================================================================

/// This structure is to let user define their own distance with closures.
pub struct DistFn<T: Copy + Clone + Sized + Send + Sync> {
    dist_function: Box<dyn Fn(&[T], &[T]) -> f32 + Send + Sync>,
}

impl<T: Copy + Clone + Sized + Send + Sync> DistFn<T> {
    /// construction of a DistFn
    pub fn new(f: Box<dyn Fn(&[T], &[T]) -> f32 + Send + Sync>) -> Self {
        DistFn { dist_function: f }
    }
}

impl<T: Copy + Clone + Sized + Send + Sync> Distance<T> for DistFn<T> {
    fn eval(&self, va: &[T], vb: &[T]) -> f32 {
        (self.dist_function)(va, vb)
    }
}

//=======================================================================================

/// This structure uses a Rust function pointer to define the distance.
/// For commodity it can build upon a fonction returning a f64.
/// Beware that if F is f64, the distance converted to f32 can overflow!

#[derive(Copy, Clone)]
pub struct DistPtr<T: Copy + Clone + Sized + Send + Sync, F: Float> {
    dist_function: fn(&[T], &[T]) -> F,
}

impl<T: Copy + Clone + Sized + Send + Sync, F: Float> DistPtr<T, F> {
    /// construction of a DistPtr
    pub fn new(f: fn(&[T], &[T]) -> F) -> Self {
        DistPtr { dist_function: f }
    }
}

/// beware that if F is f64, the distance converted to f32 can overflow!
impl<T: Copy + Clone + Sized + Send + Sync, F: Float> Distance<T> for DistPtr<T, F> {
    fn eval(&self, va: &[T], vb: &[T]) -> f32 {
        (self.dist_function)(va, vb).to_f32().unwrap()
    }
}

//=======================================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hnsw::*;

    fn init_log() -> u64 {
        let mut builder = env_logger::Builder::from_default_env();
        let _ = builder.is_test(true).try_init();
        println!("\n ************** initializing logger *****************\n");
        return 1;
    }

    #[test]
    fn test_access_to_dist_l1() {
        let distl1: DistL1 = DistL1;
        //
        let v1: Vec<i32> = vec![1, 2, 3];
        let v2: Vec<i32> = vec![2, 2, 3];

        let d1: f32 = Distance::eval(&distl1, &v1, &v2);
        assert_eq!(d1, 1 as f32);

        let v3: Vec<f32> = vec![1., 2., 3.];
        let v4: Vec<f32> = vec![2., 2., 3.];
        let d2: f32 = distl1.eval(&v3, &v4);
        assert_eq!(d2, 1 as f32);
    }

    #[test]
    fn have_avx2() {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            if is_x86_feature_detected!("avx2") {
                println!("I have avx2");
            } else {
                println!(" ************ I DO NOT  have avx2  ***************");
            }
        }
    } // end if

    #[test]
    fn have_avx512f() {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            if is_x86_feature_detected!("avx512f") {
                println!("I have avx512f");
            } else {
                println!(" ************ I DO NOT  have avx512f  ***************");
            }
        } // end of have_avx512f
    }

    #[test]
    fn have_sse2() {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            if is_x86_feature_detected!("sse2") {
                println!("I have sse2");
            } else {
                println!(" ************ I DO NOT  have SSE2  ***************");
            }
        }
    } // end of have_sse2

    #[test]
    fn test_access_to_dist_cos() {
        let distcos = DistCosine;
        //
        let v1: Vec<i32> = vec![1, -1, 1];
        let v2: Vec<i32> = vec![2, 1, -1];

        let d1: f32 = Distance::eval(&distcos, &v1, &v2);
        assert_eq!(d1, 1. as f32);
        //
        let v1: Vec<f32> = vec![1.234, -1.678, 1.367];
        let v2: Vec<f32> = vec![4.234, -6.678, 10.367];
        let d1: f32 = Distance::eval(&distcos, &v1, &v2);

        let mut normv1: f32 = 0.;
        let mut normv2: f32 = 0.;
        let mut prod: f32 = 0.;
        for i in 0..v1.len() {
            prod += v1[i] * v2[i];
            normv1 += v1[i] * v1[i];
            normv2 += v2[i] * v2[i];
        }
        let dcos: f32 = 1. - prod / (normv1 * normv2).sqrt();
        println!("dist cos avec macro = {:?} ,  avec for {:?}", d1, dcos);
    }

    #[test]
    fn test_dot_distances() {
        let mut v1: Vec<f32> = vec![1.234, -1.678, 1.367];
        let mut v2: Vec<f32> = vec![4.234, -6.678, 10.367];

        let mut normv1: f32 = 0.;
        let mut normv2: f32 = 0.;
        let mut prod: f32 = 0.;
        for i in 0..v1.len() {
            prod += v1[i] * v2[i];
            normv1 += v1[i] * v1[i];
            normv2 += v2[i] * v2[i];
        }
        let dcos: f32 = 1. - prod / (normv1 * normv2).sqrt();
        //
        l2_normalize(&mut v1);
        l2_normalize(&mut v2);

        println!(" after normalisation v1 = {:?}", v1);

        let dot: f32 = DistDot.eval(&v1, &v2);

        println!("dot  cos avec prenormalisation  = {:?} ,  avec for {:?}", dot, dcos);
    }

    #[test]
    fn test_jaccard_u16() {
        let v1: Vec<u16> = vec![1, 2, 1, 4, 3];
        let v2: Vec<u16> = vec![2, 2, 1, 5, 6];

        let dist: f32 = DistJaccard.eval(&v1, &v2);
        println!("dist jaccard = {:?}", dist);
        assert_eq!(dist, 1. - 11. / 16.);
    } // end of test_jaccard

    #[test]
    fn test_levenshtein() {
        let mut v1: Vec<u16> = vec![1, 2, 3, 4];
        let mut v2: Vec<u16> = vec![1, 2, 3, 3];
        let mut dist: f32 = DistLevenshtein.eval(&v1, &v2);
        println!("dist levenshtein = {:?}", dist);
        assert_eq!(dist, 1.0);
        v1 = vec![1, 2, 3, 4];
        v2 = vec![1, 2, 3, 4];
        dist = DistLevenshtein.eval(&v1, &v2);
        println!("dist levenshtein = {:?}", dist);
        assert_eq!(dist, 0.0);
        v1 = vec![1, 1, 1, 4];
        v2 = vec![1, 2, 3, 4];
        dist = DistLevenshtein.eval(&v1, &v2);
        println!("dist levenshtein = {:?}", dist);
        assert_eq!(dist, 2.0);
        v2 = vec![1, 1, 1, 4];
        v1 = vec![1, 2, 3, 4];
        dist = DistLevenshtein.eval(&v1, &v2);
        println!("dist levenshtein = {:?}", dist);
        assert_eq!(dist, 2.0);
    } // end of test_levenshtein

    extern "C" fn dist_func_float(va: *const f32, vb: *const f32, len: c_ulonglong) -> f32 {
        let mut dist: f32 = 0.;
        let sa: &[f32] = unsafe { std::slice::from_raw_parts(va, len as usize) };
        let sb: &[f32] = unsafe { std::slice::from_raw_parts(vb, len as usize) };

        for i in 0..len {
            dist += (sa[i as usize] - sb[i as usize]).abs().sqrt();
        }
        dist
    }

    #[test]
    fn test_dist_ext_float() {
        let va: Vec<f32> = vec![1., 2., 3.];
        let vb: Vec<f32> = vec![1., 2., 3.];
        println!("in test_dist_ext_float");
        let dist1: f32 = dist_func_float(va.as_ptr(), vb.as_ptr(), va.len() as c_ulonglong);
        println!("test_dist_ext_float computed : {:?}", dist1);

        let mydist = DistCFFI::<f32>::new(dist_func_float);

        let dist2: f32 = mydist.eval(&va, &vb);
        assert_eq!(dist1, dist2);
    } // end test_dist_ext_float

    #[test]

    fn test_my_closure() {
        let weight = vec![0.1, 0.8, 0.1];
        let my_fn = move |va: &[f32], vb: &[f32]| -> f32 {
            // should check that we work with same size for va, vb, and weight...
            let mut dist: f32 = 0.;
            for i in 0..va.len() {
                dist += weight[i] * (va[i] - vb[i]).abs();
            }
            dist
        };
        let my_boxed_f = Box::new(my_fn);
        let my_boxed_dist = DistFn::<f32>::new(my_boxed_f);
        let va: Vec<f32> = vec![1., 2., 3.];
        let vb: Vec<f32> = vec![2., 2., 4.];
        let dist = my_boxed_dist.eval(&va, &vb);
        println!("test_my_closure computed : {:?}", dist);
        // try allocation Hnsw
        let _hnsw = Hnsw::<f32, DistFn<f32>>::new(10, 3, 100, 16, my_boxed_dist);
        //
        assert_eq!(dist, 0.2);
    } // end of test_my_closure

    #[test]
    fn test_hellinger() {
        let length = 9;
        let mut p_data = Vec::with_capacity(length);
        let mut q_data = Vec::with_capacity(length);
        for _ in 0..length {
            p_data.push(1. / length as f32);
            q_data.push(1. / length as f32);
        }
        p_data[0] -= 1. / (2 * length) as f32;
        p_data[1] += 1. / (2 * length) as f32;
        //
        let dist = DistHellinger.eval(&p_data, &q_data);

        let dist_exact_fn = |n: usize| -> f32 {
            let d1: f32 = (4. - (6 as f32).sqrt() - (2 as f32).sqrt()) / n as f32;
            d1.sqrt() / (2 as f32).sqrt()
        };
        let dist_exact: f32 = dist_exact_fn(length);
        //
        log::info!("dist computed {:?} dist exact{:?} ", dist, dist_exact);
        println!("dist computed  {:?} , dist exact {:?} ", dist, dist_exact);
        //
        assert!((dist - dist_exact).abs() < 1.0e-5);
    }

    #[test]

    fn test_jeffreys() {
        // this essentially test av2 implementation for f32
        let length: usize = 19;
        let mut p_data: Vec<f32> = Vec::with_capacity(length);
        let mut q_data: Vec<f32> = Vec::with_capacity(length);
        for _ in 0..length {
            p_data.push(1. / length as f32);
            q_data.push(1. / length as f32);
        }
        p_data[0] -= 1. / (2 * length) as f32;
        p_data[1] += 1. / (2 * length) as f32;
        q_data[10] += 1. / (2 * length) as f32;
        //
        let dist_eval: f32 = DistJeffreys.eval(&p_data, &q_data);
        let mut dist_test: f32 = 0.;
        for i in 0..length {
            dist_test +=
                (p_data[i] - q_data[i]) * (p_data[i].max(M_MIN) / q_data[i].max(M_MIN)).ln();
        }
        //
        log::info!("dist eval {:?} dist test{:?} ", dist_eval, dist_test);
        println!("dist eval  {:?} , dist test {:?} ", dist_eval, dist_test);
        assert!(dist_test >= 0.);
        assert!((dist_eval - dist_test).abs() < 1.0e-5);
    }

    #[test]
    fn test_jensenshannon() {
        init_log();
        //
        let length = 19;
        let mut p_data: Vec<f32> = Vec::with_capacity(length);
        let mut q_data: Vec<f32> = Vec::with_capacity(length);
        for _ in 0..length {
            p_data.push(1. / length as f32);
            q_data.push(1. / length as f32);
        }
        p_data[0] -= 1. / (2 * length) as f32;
        p_data[1] += 1. / (2 * length) as f32;
        q_data[10] += 1. / (2 * length) as f32;
        p_data[12] = 0.;
        q_data[12] = 0.;
        //
        let dist_eval = DistJensenShannon.eval(&p_data, &q_data);
        //
        log::info!("dist eval {:?} ", dist_eval);
        println!("dist eval  {:?} ", dist_eval);
    }

    #[allow(unused)]
    use rand::distributions::{Distribution, Uniform};

    // to be run with and without simdeez_f
    #[test]
    fn test_hamming_f64() {
        init_log();

        let size_test = 500;
        let fmax: f64 = 3.;
        let mut rng = rand::thread_rng();
        for i in 300..size_test {
            // generer 2 va et vb s des vecteurs<i32> de taille i  avec des valeurs entre -imax et +
            // imax et controler les resultat
            let between = Uniform::<f64>::from(-fmax..fmax);
            let va: Vec<f64> = (0..i)
                .into_iter()
                .map(|_| between.sample(&mut rng))
                .collect();
            let mut vb: Vec<f64> = (0..i)
                .into_iter()
                .map(|_| between.sample(&mut rng))
                .collect();
            // reset half of vb to va
            for i in 0..i / 2 {
                vb[i] = va[i];
            }

            let easy_dist: u32 = va
                .iter()
                .zip(vb.iter())
                .map(|(a, b)| if a != b { 1 } else { 0 })
                .sum();
            let h_dist = DistHamming.eval(&va, &vb);
            let easy_dist = easy_dist as f32 / va.len() as f32;
            let j_exact = ((i / 2) as f32) / (i as f32);
            log::debug!(
                "test size {:?}  HammingDist {:.3e} easy : {:.3e} exact : {:.3e} ",
                i,
                h_dist,
                easy_dist,
                j_exact
            );
            if (easy_dist - h_dist).abs() > 1.0e-5 {
                println!(" jhamming = {:?} , jexact = {:?}", h_dist, easy_dist);
                log::debug!("va = {:?}", va);
                log::debug!("vb = {:?}", vb);
                std::process::exit(1);
            }
            if (j_exact - h_dist).abs() > 1. / i as f32 + 1.0E-5 {
                println!(
                    " jhamming = {:?} , jexact = {:?}, j_easy : {:?}",
                    h_dist, j_exact, easy_dist
                );
                log::debug!("va = {:?}", va);
                log::debug!("vb = {:?}", vb);
                std::process::exit(1);
            }
        }
    } // end of test_hamming_f64

    #[test]
    fn test_hamming_f32() {
        init_log();

        let size_test = 500;
        let fmax: f32 = 3.;
        let mut rng = rand::thread_rng();
        for i in 300..size_test {
            // generer 2 va et vb s des vecteurs<i32> de taille i  avec des valeurs entre -imax et +
            // imax et controler les resultat
            let between = Uniform::<f32>::from(-fmax..fmax);
            let va: Vec<f32> = (0..i)
                .into_iter()
                .map(|_| between.sample(&mut rng))
                .collect();
            let mut vb: Vec<f32> = (0..i)
                .into_iter()
                .map(|_| between.sample(&mut rng))
                .collect();
            // reset half of vb to va
            for i in 0..i / 2 {
                vb[i] = va[i];
            }

            let easy_dist: u32 = va
                .iter()
                .zip(vb.iter())
                .map(|(a, b)| if a != b { 1 } else { 0 })
                .sum();
            let h_dist = DistHamming.eval(&va, &vb);
            let easy_dist = easy_dist as f32 / va.len() as f32;
            let j_exact = ((i / 2) as f32) / (i as f32);
            log::debug!(
                "test size {:?}  HammingDist {:.3e} easy : {:.3e} exact : {:.3e} ",
                i,
                h_dist,
                easy_dist,
                j_exact
            );
            if (easy_dist - h_dist).abs() > 1.0e-5 {
                println!(
                    " jhamming = {:?} , jexact = {:?}, j_easy : {:?}",
                    h_dist, j_exact, easy_dist
                );
                log::debug!("va = {:?}", va);
                log::debug!("vb = {:?}", vb);
                std::process::exit(1);
            }
            if (j_exact - h_dist).abs() > 1. / i as f32 + 1.0E-5 {
                println!(
                    " jhamming = {:?} , jexact = {:?}, j_easy : {:?}",
                    h_dist, j_exact, easy_dist
                );
                log::debug!("va = {:?}", va);
                log::debug!("vb = {:?}", vb);
                std::process::exit(1);
            }
        }
    } // end of test_hamming_f32

    //  to run with cargo test --features packed_simd_f -- dist::tests::test_simd_hamming_u32
    #[test]
    fn test_simd_hamming_u32() {
        init_log();
        log::info!("testing test_simd_hamming_u32 with packed_simd_2");
        //
        let size_test: i32 = 500;
        let imax: u32 = 3;
        let mut rng: rand::rngs::ThreadRng = rand::thread_rng();
        for i in 4..size_test {
            // generer 2 va et vb s des vecteurs<i32> de taille i  avec des valeurs entre -imax et +
            // imax et controler les resultat
            let between: Uniform<u32> = Uniform::<u32>::from(0..imax);
            let va: Vec<u32> = (0..i)
                .into_iter()
                .map(|_| between.sample(&mut rng))
                .collect();
            let vb: Vec<u32> = (0..i)
                .into_iter()
                .map(|_| between.sample(&mut rng))
                .collect();
            let simd_dist = distance_jaccard_u32_16_simd(&va, &vb);

            let easy_dist: u32 = va
                .iter()
                .zip(vb.iter())
                .map(|(a, b)| if a != b { 1 } else { 0 })
                .sum();
            let easy_dist: f32 = easy_dist as f32 / va.len() as f32;
            log::debug!("test size {:?} simd  exact = {:?} {:?}", i, simd_dist, easy_dist);
            if (easy_dist - simd_dist).abs() > 1.0e-5 {
                println!(" jsimd = {:?} , jexact = {:?}", simd_dist, easy_dist);
                println!("va = {:?}", va);
                println!("vb = {:?}", vb);
                std::process::exit(1);
            }
        }
    } // end of test_simd_hamming_u32

    #[test]
    fn test_simd_hamming_u64() {
        init_log();
        log::info!("testing test_simd_hamming_u32 with packed_simd_2");
        //
        let size_test: i32 = 500;
        let imax: u64 = 3;
        let mut rng: rand::rngs::ThreadRng = rand::thread_rng();
        for i in 4..size_test {
            // generer 2 va et vb s des vecteurs<i32> de taille i  avec des valeurs entre -imax et +
            // imax et controler les resultat
            let between: Uniform<u64> = Uniform::<u64>::from(0..imax);
            let va: Vec<u64> = (0..i)
                .into_iter()
                .map(|_| between.sample(&mut rng))
                .collect();
            let vb: Vec<u64> = (0..i)
                .into_iter()
                .map(|_| between.sample(&mut rng))
                .collect();
            let simd_dist = distance_jaccard_u64_8_simd(&va, &vb);

            let easy_dist: u64 = va
                .iter()
                .zip(vb.iter())
                .map(|(a, b)| if a != b { 1 } else { 0 })
                .sum();
            let easy_dist: f32 = easy_dist as f32 / va.len() as f32;
            println!("test size {:?} simd  exact = {:?} {:?}", i, simd_dist, easy_dist);
            if (easy_dist - simd_dist).abs() > 1.0e-5 {
                println!(" jsimd = {:?} , jexact = {:?}", simd_dist, easy_dist);
                println!("va = {:?}", va);
                println!("vb = {:?}", vb);
                std::process::exit(1);
            }
        }
    } // end of test_simd_hamming_u64

    #[test]
    fn test_feature_simd() {
        init_log();
        log::info!("I have activated packed_simd_2");
    } // end of test_feature_simd
} // end of module tests
