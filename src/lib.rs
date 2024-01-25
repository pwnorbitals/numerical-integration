//!  # Numerical Integration
//!
//!  Algorithms and traits for numerical approximation in the Rust language
//!
//!  The purpose of this crate is to provide a selection of common algorithms
//!  for approximating differential equations and to unify them under a common API
//!  in order to make the process of using and testing different integration schemes
//!  as easy as possible.
//!
//!  # How to use
//!
//!  The primary entry point into the API is through the `Integrator`,
//!  `VelIntegrator`, and `AdaptiveIntegrator` traits. Each one has an `init()`
//!  method and `step()`. They are all designed in such a way so that all values
//!  and state of the integrator are stored *externally* and passed in through method
//!  arguments exclusively.
//!
//!  Each trait represents a different class of algorithms requiring different
//!  kinds of data. The `Integrator` trait takes in the current state and time,
//!  the timestep, and a closure/function that can compute the derivative of the
//!  state vector. `VelIntegrator` does the same, but also requires a function
//!  that computes the velocity in an admittedly convoluted way. And
//!  `AdaptiveIntegrator` takes in a minimum error value instead of a time-step.
//!
//!  In addition to these traits are traits that are like the above but adapted to
//!  not include generics in the function signature so that it can be used as in
//!  `dyn` types.
//!
//!  To use, you can either work with the traits generally and pass in a particular
//!  implementor, or you can just use the various algorithms directly.
//!
//!  At the moment, this crate includes Velocity Verlet and methods in the
//!  Runge-Kutta family (including Euler and RK4), but others (such as the linear
//!  multistep methods) could be added if enough people want them.
//!
//!  # Current state of the project
//!
//!  This project is currently in hiatus for now, and it will probably remain as such
//!  unless enough people express interest in it to be continued. If that *does* happen
//!  though, expect the API to have breaking changes, as I am not quite
//!  satisfied with the current design and certain features I wish to add require
//!  a slight redesign.

extern crate maths_traits;

use maths_traits::algebra::module_like::*;
use maths_traits::analysis::metric::*;
use maths_traits::analysis::real::*;

type Eval<'a, R, D, S> = &'a dyn Fn(R, S) -> (D, S);

pub trait Integrator {
    fn init<R: Real, D: Clone + Default, S: VectorSpace<R>, F: Fn(R, S) -> (D, S)>(
        &self,
        state: S,
        _dt: R,
        _force: F,
    ) -> Box<[(D, S)]> {
        Box::new([(Default::default(), state)])
    }
    fn step<R: Real, D: Clone + Default, S: VectorSpace<R>, F: Fn(R, S) -> (D, S)>(
        &self,
        time: R,
        state: &mut [(D, S)],
        dt: R,
        force: F,
    ) -> (D, S);
}

pub trait Integrates<R: Real, D: Clone + Default, S: VectorSpace<R>> {
    fn init(&self, state: S, _dt: R, _force: Eval<R, D, S>) -> Box<[(D, S)]> {
        Box::new([(Default::default(), state)])
    }
    fn step(&self, time: R, state: &mut [(D, S)], dt: R, force: Eval<R, D, S>) -> (D, S);
}

impl<I: Integrator, R: Real, D: Clone + Default, S: VectorSpace<R>> Integrates<R, D, S> for I {
    fn init(&self, state: S, dt: R, force: Eval<R, D, S>) -> Box<[(D, S)]> {
        Integrator::init(self, state, dt, force)
    }
    fn step(&self, time: R, state: &mut [(D, S)], dt: R, force: Eval<R, D, S>) -> (D, S) {
        Integrator::step(self, time, state, dt, force)
    }
}

pub trait VelIntegrator {
    fn init_with_vel<
        R: Real,
        D: Clone + Default,
        S: VectorSpace<R>,
        V: Fn(R, S) -> (D, S),
        F: Fn(R, S) -> (D, S),
    >(
        &self,
        state: S,
        _dt: R,
        _vel: V,
        _force: F,
    ) -> Box<[(D, S)]> {
        Box::new([(Default::default(), state)])
    }

    fn step_with_vel<
        R: Real,
        D: Clone + Default,
        S: VectorSpace<R>,
        V: Fn(R, S) -> (D, S),
        F: Fn(R, S) -> (D, S),
    >(
        &self,
        time: R,
        state: &mut [(D, S)],
        dt: R,
        velocity: V,
        force: F,
    ) -> (D, S);
}

pub trait VelIntegrates<R: Real, D: Clone + Default, S: VectorSpace<R>> {
    fn init_with_vel(
        &self,
        state: S,
        _dt: R,
        _vel: Eval<R, D, S>,
        _force: Eval<R, D, S>,
    ) -> Box<[(D, S)]> {
        Box::new([(Default::default(), state)])
    }
    fn step_with_vel(
        &self,
        time: R,
        state: &mut [(D, S)],
        dt: R,
        vel: Eval<R, D, S>,
        force: Eval<R, D, S>,
    ) -> (D, S);
}

impl<I: VelIntegrator, R: Real, D: Clone + Default, S: VectorSpace<R>> VelIntegrates<R, D, S>
    for I
{
    fn init_with_vel(
        &self,
        state: S,
        dt: R,
        vel: Eval<R, D, S>,
        force: Eval<R, D, S>,
    ) -> Box<[(D, S)]> {
        VelIntegrator::init_with_vel(self, state, dt, vel, force)
    }
    fn step_with_vel(
        &self,
        time: R,
        state: &mut [(D, S)],
        dt: R,
        vel: Eval<R, D, S>,
        force: Eval<R, D, S>,
    ) -> (D, S) {
        VelIntegrator::step_with_vel(self, time, state, dt, vel, force)
    }
}

pub trait AdaptiveIntegrator {
    fn adaptive_init<
        R: Real,
        D: Clone + Default,
        S: VectorSpace<R>,
        M: Metric<S, R>,
        F: Fn(R, S) -> (D, S),
    >(
        &self,
        t0: R,
        state: S,
        _ds: R,
        _force: F,
        _d: M,
    ) -> Box<[(R, D, S)]> {
        Box::new([(t0, Default::default(), state)])
    }
    fn adaptive_step<
        R: Real,
        D: Clone + Default,
        S: VectorSpace<R>,
        M: Metric<S, R>,
        F: Fn(R, S) -> (D, S),
    >(
        &self,
        state: &mut [(R, D, S)],
        ds: R,
        force: F,
        d: M,
    ) -> (R, D, S);
}

pub use runge_kutta::*;
pub mod runge_kutta;

#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
pub struct VelocityVerlet;

impl VelIntegrator for VelocityVerlet {
    fn init_with_vel<
        R: Real,
        D: Clone + Default,
        S: VectorSpace<R>,
        V: Fn(R, S) -> (D, S),
        F: Fn(R, S) -> (D, S),
    >(
        &self,
        state: S,
        _dt: R,
        _vel: V,
        _force: F,
    ) -> Box<[(D, S)]> {
        Box::new([(Default::default(), state), (Default::default(), S::zero())])
    }

    fn step_with_vel<
        R: Real,
        D: Clone + Default,
        S: VectorSpace<R>,
        V: Fn(R, S) -> (D, S),
        F: Fn(R, S) -> (D, S),
    >(
        &self,
        time: R,
        state: &mut [(D, S)],
        dt: R,
        velocity: V,
        force: F,
    ) -> (D, S) {
        let (s1, rest) = state.split_first_mut().unwrap();
        let (a1, _) = rest.split_first_mut().unwrap();

        let mid = s1.clone().1
            + a1.clone().1 * dt.clone()
            + velocity(time.clone(), a1.clone().1).1 * (dt.clone() * dt.clone() * R::repr(0.5));

        s1.1 += a1.clone().1 * (dt.clone() * R::repr(0.5));
        a1.1 = force(time.clone() + dt.clone(), mid).1;
        s1.1 += a1.clone().1 * (dt.clone() * R::repr(0.5));

        s1.clone()
    }
}
