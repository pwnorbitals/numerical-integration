extern crate numerical_integration;

use numerical_integration::{Integrator, EULER, RK4};

fn main() {
    use std::f64::consts::*;

    //
    //Computes pi by numerically approximating the integral of e^(-x^2)
    //

    //the function e^(-t^2) that we are trying to integrate
    fn f(t: f64, _y: f64) -> ((), f64) {
        ((), (-t * t).exp())
    }

    let n = 1000; //the number of steps
    let l = 100.0; //the size of the interval to integrate over
    let dt = l / (n as f64);

    //init
    let mut t = -l / 2.0;
    let mut y1 = EULER.init(0.0, dt, f);
    let mut y2 = RK4.init(0.0, dt, f);

    for _ in 0..n {
        println!(
            "{} {} {}",
            EULER.step(t, y1.as_mut(), dt, f).1,
            RK4.step(t, y2.as_mut(), dt, f).1,
            PI.sqrt()
        );
        t += dt;
    }

    //the result should be sqrt(PI)
    println!("{} {} {}", y1[0].1 * y1[0].1, y2[0].1 * y2[0].1, PI);
}
