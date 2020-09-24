
extern crate maths_traits;
extern crate numerical_integration;

use numerical_integration::{Integrator, EULER, RK4};


fn main() {

    fn f(t: f64, y: f64) -> f64 { y }

    let dt = 0.125;
    let mut t = 0.0;
    let mut y1 = EULER.init(1.0, dt, &f);
    let mut y2 = RK4.init(1.0, dt, &f);

    for i in 0..100 {
        println!("{} {} {}", EULER.step(t, y1.as_mut(), dt, &f), RK4.step(t, y2.as_mut(), dt, &f), (t+dt).exp());
        t += dt;
    }


}
