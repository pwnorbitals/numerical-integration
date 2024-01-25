extern crate maths_traits;
extern crate numerical_integration;

use numerical_integration::{Integrator, EULER, RK4};

fn main() {
    //
    //A comparison between Euler, RK4, and intrinsics for computing the exponential of a real number
    //

    //the derivative of the exponential is itself
    fn f(_t: f64, y: f64) -> ((), f64) {
        ((), y)
    }

    //the time-step
    let dt = 0.125;

    //the initial time and values
    let mut t = 0.0;
    let mut y1 = EULER.init(1.0, dt, f);
    let mut y2 = RK4.init(1.0, dt, f);

    //table column lables
    for _ in 0..(9 + 11 * 3) {
        print!("_");
    }
    println!();
    println!("|      t|     Euler|       RK4|f64::exp()|");

    for _ in 0..100 {
        //compute the next step and print
        println!(
            "|{: >7.3}|{: >10.2}|{: >10.2}|{: >10.2}|",
            t + dt,
            EULER.step(t, y1.as_mut(), dt, f).1,
            RK4.step(t, y2.as_mut(), dt, f).1,
            (t + dt).exp()
        );
        t += dt;
    }

    for _ in 0..(9 + 11 * 3) {
        print!("_");
    }
    println!();
}
