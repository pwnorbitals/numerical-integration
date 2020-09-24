
extern crate maths_traits;
extern crate numerical_integration;

use numerical_integration::*;
use maths_traits::analysis::metric::InnerProductMetric;

fn main() {

    fn f(t: f64, y: f64) -> f64 { y }

    let ds = 0.5;
    let mut s1 = EULER_HEUN.adaptive_init(0.0, 1.0, ds, &f, InnerProductMetric);
    let mut s2 = BOGACKI_SHAMPINE.adaptive_init(0.0, 1.0, ds, &f, InnerProductMetric);
    let mut s3 = RK_FELBERG.adaptive_init(0.0, 1.0, ds, &f, InnerProductMetric);
    let mut s4 = DORMAND_PRINCE.adaptive_init(0.0, 1.0, ds, &f, InnerProductMetric);

    for i in 0..100 {
        // let (t, y) = EULER_HEUN.adaptive_step(s1.as_mut(), ds, &f, InnerProductMetric);
        // let (t, y) = BOGACKI_SHAMPINE.adaptive_step(s3.as_mut(), ds, &f, InnerProductMetric);
        // let (t, y) = RK_FELBERG.adaptive_step(s3.as_mut(), ds, &f, InnerProductMetric);
        let (t, y) = DORMAND_PRINCE.adaptive_step(s4.as_mut(), ds, &f, InnerProductMetric);
        println!("t={} y={} exp(t)={}", t, y, t.exp());
        // println!("{:?} ", RK_FELBERG.adaptive_step(s2.as_mut(), ds, &f));
    }


}
