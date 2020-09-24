extern crate numerical_integration;

use numerical_integration::{Integrator, EULER, RK4};


fn main() {

    use std::f64::consts::*;

    fn f(t: f64, y: f64) -> f64 { (-t*t).exp() }

    let n = 2000;
    let dT = 10.0;
    let dt = dT / (n as f64);
    let mut t = -dT/2.0;
    let mut y1 = EULER.init(0.0, dt, &f);
    let mut y2 = RK4.init(0.0, dt, &f);

    for _ in 0..n {
        println!("{} {} {}", EULER.step(t, y1.as_mut(), dt, &f), RK4.step(t, y2.as_mut(), dt, &f), PI.sqrt());
        t += dt;
    }

    println!("{} {} {}", y1[0]*y1[0], y2[0]*y2[0], PI);


}
