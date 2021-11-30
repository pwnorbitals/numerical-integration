
# Numerical Integration

Algorithms and traits for numerical approximation in the Rust language

The purpose of this crate is to provide a selection of common algorithms
for approximating differential equations and to unify them under a common API
in order to make the process of using and testing different integration schemes
as easy as possible.

# How to use

The primary entry point into the API is through the `Integrator`,
`VelIntegrator`, and `AdaptiveIntegrator` traits. Each one has an `init()`
method and `step()`. They are all designed in such a way so that all values
and state of the integrator are stored *externally* and passed in through method
arguments exclusively.

Each trait represents a different class of algorithms requiring different
kinds of data. The `Integrator` trait takes in the current state and time,
the timestep, and a closure/function that can compute the derivative of the
state vector. `VelIntegrator` does the same, but also requires a function
that computes the velocity in an admittedly convoluted way. And
`AdaptiveIntegrator` takes in a minimum error value instead of a time-step.

In addition to these traits are traits that are like the above but adapted to
not include generics in the function signature so that it can be used as in
`dyn` types.

To use, you can either work with the traits generally and pass in a particular
implementor, or you can just use the various algorithms directly.

At the moment, this crate includes Velocity Verlet and methods in the
Runge-Kutta family (including Euler and RK4), but others (such as the linear
multistep methods) could be added if enough people want them.

# Current state of the project

This project is currently in hiatus for now, and it will probably remain as such
unless enough people express interest in it to be continued. If that *does* happen
though, expect the API to have breaking changes, as I am not quite
satisfied with the current design and certain features I wish to add require
a slight redesign.
