extern crate maths_traits;
extern crate numerical_integration;

use numerical_integration::{VelIntegrator, VelocityVerlet};

fn main() {
    use maths_traits::algebra::*;
    // use maths_traits::analysis::*;
    // use maths_traits::analysis::Exponential;

    //
    //A simple 2D vector impl
    //

    #[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
    pub struct Vec2<T> {
        pub x: T,
        pub y: T,
    }

    // impl<T> Vec2<T> {
    //     #[inline]
    //     pub fn new(x: T, y: T) -> Self {
    //         Vec2 { x: x, y: y }
    //     }
    // }

    impl<T: Add<Output = T>> Add for Vec2<T> {
        type Output = Self;
        #[inline]
        fn add(self, rhs: Self) -> Self {
            Vec2 {
                x: self.x + rhs.x,
                y: self.y + rhs.y,
            }
        }
    }

    impl<T: AddAssign> AddAssign for Vec2<T> {
        #[inline]
        fn add_assign(&mut self, rhs: Self) {
            self.x += rhs.x;
            self.y += rhs.y;
        }
    }

    impl<T: Sub<Output = T>> Sub for Vec2<T> {
        type Output = Self;
        #[inline]
        fn sub(self, rhs: Self) -> Self {
            Vec2 {
                x: self.x - rhs.x,
                y: self.y - rhs.y,
            }
        }
    }

    impl<T: SubAssign> SubAssign for Vec2<T> {
        #[inline]
        fn sub_assign(&mut self, rhs: Self) {
            self.x -= rhs.x;
            self.y -= rhs.y;
        }
    }

    impl<T: Neg<Output = T>> Neg for Vec2<T> {
        type Output = Self;
        #[inline]
        fn neg(self) -> Self {
            Vec2 {
                x: -self.x,
                y: -self.y,
            }
        }
    }

    impl<T: Zero> Zero for Vec2<T> {
        #[inline]
        fn zero() -> Self {
            Vec2 {
                x: T::zero(),
                y: T::zero(),
            }
        }
        #[inline]
        fn is_zero(&self) -> bool {
            self.x.is_zero() && self.y.is_zero()
        }
    }

    impl<K: Clone, T: Mul<K, Output = T>> Mul<K> for Vec2<T> {
        type Output = Self;
        #[inline]
        fn mul(self, rhs: K) -> Self {
            Vec2 {
                x: self.x * rhs.clone(),
                y: self.y * rhs,
            }
        }
    }

    impl<K: Clone, T: Clone + MulAssign<K>> MulAssign<K> for Vec2<T> {
        #[inline]
        fn mul_assign(&mut self, rhs: K) {
            self.x *= rhs.clone();
            self.y *= rhs;
        }
    }

    impl<K: Clone, T: Div<K, Output = T>> Div<K> for Vec2<T> {
        type Output = Self;
        #[inline]
        fn div(self, rhs: K) -> Self {
            Vec2 {
                x: self.x / rhs.clone(),
                y: self.y / rhs,
            }
        }
    }

    impl<K: Clone, T: Clone + DivAssign<K>> DivAssign<K> for Vec2<T> {
        #[inline]
        fn div_assign(&mut self, rhs: K) {
            self.x /= rhs.clone();
            self.y /= rhs;
        }
    }

    impl<T> AddAssociative for Vec2<T> {}
    impl<T> MulAssociative for Vec2<T> {}
    impl<T> AddCommutative for Vec2<T> {}
    impl<T> MulCommutative for Vec2<T> {}
    impl<T> Distributive<T> for Vec2<T> {}

    //
    //A simple harmonic oscillator simulation
    //
    //we use a vec2 for the state vector, where the x coord is the position
    //and the y coord is the velocity
    //

    //the x-coord of the derivative is y since it is the velocity
    //and the y coord of the derivative is -x as dv/dt = -x
    fn f(_t: f64, y: Vec2<f64>) -> ((), Vec2<f64>) {
        ((), Vec2 { x: y.y, y: -y.x })
    }

    //shifts the velocity into x and sets y to be 0
    fn v(_t: f64, y: Vec2<f64>) -> ((), Vec2<f64>) {
        ((), Vec2 { x: y.y, y: 0.0 })
    }

    //the time-step
    let dt = 0.1;

    //init
    let mut t = 0.0;
    let y0 = Vec2 { x: 1.0, y: 0.0 };
    let mut y1 = VelocityVerlet.init_with_vel(y0, dt, v, f);

    // let mut y2 = RK4.init(y0, dt, &f);
    // let mut y3 = RK4.init(y0, dt, &f);

    for _ in 0..100 {
        let yi = VelocityVerlet.step_with_vel(t, y1.as_mut(), dt, v, f);
        t += dt;

        println!(
            "{:>6.3} | {:>10.7} {:>10.7} | {:>10.7} {:>10.6} ",
            t,
            yi.1.x,
            yi.1.y,
            t.cos(),
            -t.sin()
        );
    }
}
