extern crate ncollide;
extern crate nalgebra;
extern crate noisy_float;
extern crate approx;
extern crate num_traits;
#[macro_use]
extern crate derive_more;
extern crate alga;

use nalgebra::{Point3, Vector3, Isometry3, zero};
use ncollide::world::{CollisionWorld, GeometricQueryType, CollisionGroups};
use ncollide::bounding_volume::*;
use ncollide::shape::*;
use approx::ApproxEq;
use num_traits::ToPrimitive;
use num_traits::Float;

#[derive(Debug, From, Into, Add, AddAssign, Neg, Copy, Clone, Sub, SubAssign, Eq, PartialEq, PartialOrd)]
pub struct F(noisy_float::types::N32);

impl From<f32> for F {
    fn from(v: f32) -> Self {
        F(n32(v))
    }
}

use noisy_float::types::n32;

impl alga::general::SubsetOf<F> for F {
    fn to_superset(&self) -> F {
        self.clone()
    }

    unsafe fn from_superset_unchecked(element: &F) -> Self {
        element.clone()
    }

    fn is_in_subset(_element: &F) -> bool {
        true
    }
}

impl alga::general::SubsetOf<F> for f64 {
    fn to_superset(&self) -> F {
        F(n32(*self as f32))
    }

    unsafe fn from_superset_unchecked(element: &F) -> Self {
        element.0.to_f64().unwrap()
    }

    fn is_in_subset(_element: &F) -> bool {
        true
    }
}

impl alga::general::Lattice for F {
}

impl alga::general::MeetSemilattice for F {
    fn meet(&self, other: &Self) -> Self {
        F(n32(self.0.to_f32().unwrap().meet(&other.0.to_f32().unwrap())))
    }
}

impl alga::general::JoinSemilattice for F {
    fn join(&self, other: &Self) -> Self {
        F(n32(self.0.to_f32().unwrap().join(&other.0.to_f32().unwrap())))
    }
}

impl alga::general::AbstractRingCommutative for F {
    fn prop_mul_is_commutative_approx(args: (Self, Self)) -> bool {
        f32::prop_mul_is_commutative_approx(((args.0).0.to_f32().unwrap(), (args.1).0.to_f32().unwrap()))
    }
}

impl alga::general::AbstractField for F { }
impl alga::general::AbstractRing for F { }
impl alga::general::AbstractGroup<alga::general::Additive> for F { }
impl alga::general::AbstractGroup<alga::general::Multiplicative> for F { }
impl alga::general::AbstractGroupAbelian<alga::general::Additive> for F { }
impl alga::general::AbstractGroupAbelian<alga::general::Multiplicative> for F { }
impl alga::general::AbstractMonoid<alga::general::Multiplicative> for F { }
impl alga::general::AbstractMonoid<alga::general::Additive> for F { }
impl alga::general::AbstractLoop<alga::general::Additive> for F { }
impl alga::general::AbstractLoop<alga::general::Multiplicative> for F { }
impl alga::general::AbstractSemigroup<alga::general::Multiplicative> for F { }
impl alga::general::AbstractSemigroup<alga::general::Additive> for F { }
impl alga::general::AbstractQuasigroup<alga::general::Multiplicative> for F { }
impl alga::general::AbstractQuasigroup<alga::general::Additive> for F { }
impl alga::general::AbstractMagma<alga::general::Multiplicative> for F {
    fn operate(&self, right: &Self) -> Self {
        F(self.0 * right.0)
    }
}
impl alga::general::AbstractMagma<alga::general::Additive> for F {
    fn operate(&self, right: &Self) -> Self {
        F(self.0 + right.0)
    }
}

impl alga::general::Inverse<alga::general::Multiplicative> for F {
    fn inverse(&self) -> Self {
        F(n32(1.0) / self.0)
    }
}
impl alga::general::Inverse<alga::general::Additive> for F {
    fn inverse(&self) -> Self {
        -self.clone()
    }
}
impl alga::general::Identity<alga::general::Multiplicative> for F {
    fn identity() -> Self {
        F(n32(1.0))
    }
}
impl alga::general::Identity<alga::general::Additive> for F {
    fn identity() -> Self {
        F(n32(0.0))
    }
}

impl ::std::fmt::Display for F {
    fn fmt(&self, f: &mut ::std::fmt::Formatter) -> Result<(), ::std::fmt::Error> {
        self.0.fmt(f)
    }
}

impl ::std::ops::Rem for F {
    type Output = Self;
    fn rem(self, rhs: Self) -> Self::Output {
        F(self.0.rem(rhs.0))
    }
}

impl num_traits::One for F {
    fn one() -> Self {
        F(n32(f32::one()))
    }
}

impl num_traits::Zero for F {
    fn zero() -> Self {
        F(n32(f32::zero()))
    }
    fn is_zero(&self) -> bool {
        self.0.is_zero()
    }
}

impl ::std::ops::Div for F {
    type Output = F;
    fn div(self, rhs: Self) -> Self::Output {
        F(self.0/rhs.0)
    }
}

impl ::std::ops::Mul for F {
    type Output = F;
    fn mul(self, rhs: Self) -> Self::Output {
        F(self.0*rhs.0)
    }
}

impl ::std::ops::DivAssign for F {
    fn div_assign(&mut self, rhs: Self) {
        self.0 = self.0 / rhs.0;
    }
}

impl ::std::ops::MulAssign for F {
    fn mul_assign(&mut self, rhs: Self) {
        self.0 = self.0 * rhs.0;
    }
}

impl num_traits::Num for F {
    type FromStrRadixErr = num_traits::ParseFloatError;
    fn from_str_radix(
        str: &str,
        radix: u32
    ) -> Result<Self, Self::FromStrRadixErr> {
        Ok(F(n32(f32::from_str_radix(str, radix)?)))
    }
}

impl num_traits::FromPrimitive for F {
    fn from_i64(n: i64) -> Option<Self> {
        Some(F(n32(n as f32)))
    }
    fn from_u64(n: u64) -> Option<Self> {
        Some(F(n32(n as f32)))
    }
}

impl num_traits::Bounded for F {
    fn min_value() -> Self {
        F(n32(<f32 as num_traits::Bounded>::min_value()))
    }
    fn max_value() -> Self {
        F(n32(<f32 as num_traits::Bounded>::max_value()))
    }
}

impl num_traits::Signed for F {
    fn abs(&self) -> Self {
        F(self.0.abs())
    }
    fn abs_sub(&self, other: &Self) -> Self {
        F(self.0.abs_sub(other.0))
    }
    fn signum(&self) -> Self {
        F(self.0.signum())
    }
    fn is_positive(&self) -> bool {
        self.0.to_f32().unwrap().is_positive()
    }
    fn is_negative(&self) -> bool {
        self.0.to_f32().unwrap().is_negative()
    }
}

impl ApproxEq for F {
    type Epsilon = F;
    fn default_epsilon() -> Self::Epsilon {
        F(n32(f32::default_epsilon()))
    }
    fn default_max_relative() -> Self::Epsilon {
        F(n32(f32::default_max_relative()))
    }
    fn default_max_ulps() -> u32 {
        f32::default_max_ulps()
    }
    fn relative_eq(
        &self,
        other: &Self,
        epsilon: Self::Epsilon,
        max_relative: Self::Epsilon
    ) -> bool {
        self.0.to_f32().unwrap().relative_eq(
            &other.0.to_f32().unwrap(),
            epsilon.0.to_f32().unwrap(),
            max_relative.0.to_f32().unwrap(),
        )
    }
    fn ulps_eq(
        &self,
        other: &Self,
        epsilon: Self::Epsilon,
        max_ulps: u32
    ) -> bool {
        self.0.to_f32().unwrap().ulps_eq(
            &other.0.to_f32().unwrap(),
            epsilon.0.to_f32().unwrap(),
            max_ulps,
        )
    }
}

impl nalgebra::Real for F {
    fn floor(self) -> Self {
        F(self.0.floor())
    }
    fn ceil(self) -> Self {
        F(self.0.ceil())
    }
    fn round(self) -> Self {
        F(self.0.round())
    }
    fn trunc(self) -> Self {
        F(self.0.trunc())
    }
    fn fract(self) -> Self {
        F(self.0.fract())
    }
    fn abs(self) -> Self {
        F(self.0.abs())
    }
    fn signum(self) -> Self {
        F(self.0.signum())
    }
    fn is_sign_positive(self) -> bool {
        self.0.is_sign_positive()
    }
    fn is_sign_negative(self) -> bool {
        self.0.is_sign_negative()
    }
    fn mul_add(self, a: Self, b: Self) -> Self {
        F(self.0.mul_add(a.0, b.0))
    }
    fn recip(self) -> Self {
        F(self.0.recip())
    }
    fn powi(self, n: i32) -> Self {
        F(self.0.powi(n))
    }
    fn powf(self, n: Self) -> Self {
        F(self.0.powf(n.0))
    }
    fn sqrt(self) -> Self {
        F(self.0.sqrt())
    }
    fn exp(self) -> Self {
        F(self.0.exp())
    }
    fn exp2(self) -> Self {
        F(self.0.exp2())
    }
    fn ln(self) -> Self {
        F(self.0.ln())
    }
    fn log(self, base: Self) -> Self {
        F(self.0.log(base.0))
    }
    fn log2(self) -> Self {
        F(self.0.log2())
    }
    fn log10(self) -> Self {
        F(self.0.log10())
    }
    fn max(self, other: Self) -> Self {
        F(self.0.max(other.0))
    }
    fn min(self, other: Self) -> Self {
        F(self.0.min(other.0))
    }
    fn cbrt(self) -> Self {
        F(self.0.cbrt())
    }
    fn hypot(self, other: Self) -> Self {
        F(self.0.hypot(other.0))
    }
    fn sin(self) -> Self {
        F(self.0.sin())
    }
    fn cos(self) -> Self {
        F(self.0.cos())
    }
    fn tan(self) -> Self {
        F(self.0.tan())
    }
    fn asin(self) -> Self {
        F(self.0.asin())
    }
    fn acos(self) -> Self {
        F(self.0.acos())
    }
    fn atan(self) -> Self {
        F(self.0.atan())
    }
    fn atan2(self, other: Self) -> Self {
        F(self.0.atan2(other.0))
    }
    fn sin_cos(self) -> (Self, Self) {
        let r = self.0.sin_cos();
        (F(r.0), F(r.1))
    }
    fn exp_m1(self) -> Self {
        F(self.0.exp_m1())
    }
    fn ln_1p(self) -> Self {
        F(self.0.ln_1p())
    }
    fn sinh(self) -> Self {
        F(self.0.sinh())
    }
    fn cosh(self) -> Self {
        F(self.0.cosh())
    }
    fn tanh(self) -> Self {
        F(self.0.tanh())
    }
    fn asinh(self) -> Self {
        F(self.0.asinh())
    }
    fn acosh(self) -> Self {
        F(self.0.acosh())
    }
    fn atanh(self) -> Self {
        F(self.0.atanh())
    }
    fn pi() -> Self {
        F(n32(::std::f32::consts::PI))
    }
    fn two_pi() -> Self {
        F(n32(2.0*::std::f32::consts::PI))
    }
    fn frac_pi_2() -> Self {
        F(n32(::std::f32::consts::FRAC_PI_2))
    }
    fn frac_pi_3() -> Self {
        F(n32(::std::f32::consts::FRAC_PI_3))
    }
    fn frac_pi_4() -> Self {
        F(n32(::std::f32::consts::FRAC_PI_4))
    }
    fn frac_pi_6() -> Self {
        F(n32(::std::f32::consts::FRAC_PI_6))
    }
    fn frac_pi_8() -> Self {
        F(n32(::std::f32::consts::FRAC_PI_8))
    }
    fn frac_1_pi() -> Self {
        F(n32(::std::f32::consts::FRAC_1_PI))
    }
    fn frac_2_pi() -> Self {
        F(n32(::std::f32::consts::FRAC_2_PI))
    }
    fn frac_2_sqrt_pi() -> Self {
        F(n32(::std::f32::consts::FRAC_2_SQRT_PI))
    }
    fn e() -> Self {
        F(n32(::std::f32::consts::E))
    }
    fn log2_e() -> Self {
        F(n32(::std::f32::consts::LOG2_E))
    }
    fn log10_e() -> Self {
        F(n32(::std::f32::consts::LOG10_E))
    }
    fn ln_2() -> Self {
        F(n32(::std::f32::consts::LN_2))
    }
    fn ln_10() -> Self {
        F(n32(::std::f32::consts::LN_10))
    }
}

#[test]
fn main() {
    let mut world: CollisionWorld<Point3<F>, Isometry3<F>, ()> = CollisionWorld::new(0.02f32.into());
    let groups = CollisionGroups::new();
    let contacts_query = GeometricQueryType::Contacts(0.0.into(), 0.0.into());

    let min = Point3::new((-0.5).into(), (-0.5).into(), (-0.1).into());
    let max = Point3::new(0.5.into(), 0.5.into(), 0.1.into());
    let bounding = AABB::new(min, max);
    let cuboid = Cuboid::new(bounding.half_extents());
    let shape = ShapeHandle::new(cuboid);

    let iso1 = Isometry3::new(Vector3::new(0.0.into(), 0.0.into(), 0.0.into()), zero());
    world.add(iso1, shape.clone(), groups, contacts_query, ());

    let iso2 = Isometry3::new(Vector3::new(0.0.into(), 1.0.into(), 0.0.into()), zero());
    world.add(iso2, shape, groups, contacts_query, ());

    world.update();
}
