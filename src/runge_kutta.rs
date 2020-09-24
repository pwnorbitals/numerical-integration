use super::*;

use std::fmt::{Debug, Formatter};

#[derive(Clone, Copy, PartialEq)]
pub enum RKError {
    EmptyTableau,
    JaggedTableau,
    TooManyColumns(usize, usize),
    NonSquareTableau(usize, usize),
    UnsupportedImplicit
}

impl Debug for RKError {
    fn fmt(&self, f: &mut Formatter) -> ::std::fmt::Result {
        match self {
            RKError::EmptyTableau =>
                write!(f, "Zero-length Runge-Kutta matrix"),
            RKError::JaggedTableau =>
                write!(f, "Tableau is non-rectangular"),
            RKError::TooManyColumns(r, c) =>
                write!(f, "Tableau has {} rows but {} columns", r, c),
            RKError::NonSquareTableau(r, c) =>
                write!(f, "Non-square tableau; number of rows is {} but there is a row of length {}", r, c),
            RKError::UnsupportedImplicit =>
                write!(f, "Implicit Runge-Kutta not supported")
        }
    }
}

#[derive(Clone, Copy, PartialEq, Debug)]
pub enum ButcherTableau<'a> {
    Fixed(&'a[&'a[f64]]),
    Adaptive(&'a[&'a[f64]]),
    Implicit(&'a[&'a[f64]]),
    AdaptiveImplicit(&'a[&'a[f64]])
}

impl<'a> ButcherTableau<'a> {
    fn new(table: &'a[&'a[f64]]) -> Result<Self, RKError> {
        use ButcherTableau::*;
        use RKError::*;

        //make sure the tableau is non-empty
        if table.len()==0 {
            Err(EmptyTableau)
        } else {
            let rows = table.len();
            let columns = table[0].len();

            //make sure we have enough rows
            if columns>rows { return Err(TooManyColumns(rows, columns)); }

            //check if the tableau is of an implict method and make sure we have a non-jagged array
            let mut implicit = false;
            for i in 0..rows {
                if table[i].len()!=columns { return Err(JaggedTableau); }
                for j in i..columns {
                    if table[i][j] != 0.0 {
                        implicit = true;
                        break;
                    }
                }
            }

            Ok(match (rows>columns, implicit) {
                (false, false) => Fixed(table),
                (true, false) => Adaptive(table),
                (false, true) => Implicit(table),
                (true, true) => AdaptiveImplicit(table),
            })

        }
    }
}


#[derive(Clone, Copy, PartialEq, Debug)]
pub struct RungeKutta<'a>(&'a[&'a[f64]]);

#[derive(Clone, Copy, PartialEq, Debug)]
pub struct AdaptiveRungeKutta<'a>(&'a[&'a[f64]]);

pub const EULER: RungeKutta = RK1;
pub const MIDPOINT: RungeKutta = RK2;
pub const RK1: RungeKutta = RungeKutta(
    &[&[0.0,0.0],
      &[0.0,1.0]]
);
pub const RK2: RungeKutta = RungeKutta(
    &[&[0.0,0.0,0.0],
      &[0.5,0.5,0.0],
      &[0.0,0.0,1.0]]
);
pub const HEUN2: RungeKutta = RungeKutta(
    &[&[0.0,0.0,0.0],
      &[1.0,1.0,0.0],
      &[0.0,0.5,0.5]]
);
pub const RALSTON: RungeKutta = RungeKutta(
    &[&[0.0,    0.0,    0.0 ],
      &[2.0/3.0,2.0/3.0,0.0 ],
      &[0.0,    0.25,   0.75]]
);
pub const RK3: RungeKutta = RungeKutta(
    &[&[0.0, 0.0,     0.0,     0.0],
      &[0.5, 0.5,     0.0,     0.0],
      &[1.0,-1.0,     2.0,     0.0],
      &[0.0, 1.0/6.0, 2.0/3.0, 1.0/6.0]]
);
pub const HEUN3: RungeKutta = RungeKutta(
    &[&[0.0,    0.0,    0.0,    0.0],
      &[1.0/3.0,1.0/3.0,0.0,    0.0],
      &[2.0/3.0,0.0,    2.0/3.0,0.0],
      &[0.0,    0.25,   0.0,    0.75]]
);
pub const RK4: RungeKutta = RungeKutta(
    &[&[0.0, 0.0,     0.0,     0.0,     0.0],
      &[0.5, 0.5,     0.0,     0.0,     0.0],
      &[0.5, 0.0,     0.5,     0.0,     0.0],
      &[1.0, 0.0,     0.0,     1.0,     0.0],
      &[0.0, 1.0/6.0, 1.0/3.0, 1.0/3.0, 1.0/6.0]]
);
pub const RK_3_8: RungeKutta = RungeKutta(
    &[&[0.0,     0.0,      0.0,   0.0,   0.0],
      &[1.0/3.0, 1.0/3.0,  0.0,   0.0,   0.0],
      &[2.0/3.0, -1.0/3.0, 1.0,   0.0,   0.0],
      &[1.0,     1.0,      -1.0,  1.0,   0.0],
      &[0.0,     0.125,    0.375, 0.375, 0.125]]
);

pub const EULER_HEUN: AdaptiveRungeKutta = AdaptiveRungeKutta(
    &[&[0.0, 0.0, 0.0],
      &[1.0, 1.0, 0.0],
      &[0.0, 0.5, 0.5],
      &[0.0, 1.0, 0.0]]
);

pub const BOGACKI_SHAMPINE: AdaptiveRungeKutta = AdaptiveRungeKutta(
    &[&[0.0,  0.0,      0.0,     0.0,     0.0],
      &[0.5,  0.5,      0.0,     0.0,     0.0],
      &[0.75, 0.0,      0.75,    0.0,     0.0],
      &[1.0,  2.0/9.0,  1.0/3.0, 4.0/9.0, 0.0],
      &[0.0,  2.0/9.0,  1.0/3.0, 4.0/9.0, 0.0],
      &[0.0,  7.0/24.0, 0.25,    1.0/3.0, 0.125]]
);

pub const RK_FELBERG: AdaptiveRungeKutta = AdaptiveRungeKutta(
    &[&[0.0,       0.0,            0.0,            0.0,            0.0,              0.0,       0.0],
      &[0.25,      0.25,           0.0,            0.0,            0.0,              0.0,       0.0],
      &[0.375,     3.0/32.0,       9.0/32.0,       0.0,            0.0,              0.0,       0.0],
      &[12.0/13.0, 1932.0/2197.0, -7200.0/2197.0,  7296.0/2197.0,  0.0,              0.0,       0.0],
      &[1.0,       439.0/216.0,   -8.0,            3680.0/513.0,  -845.0/4104.0,     0.0,       0.0],
      &[0.5,      -8.0/27.0,       2.0,           -3544.0/2565.0,  1859.0/4104.0,   -11.0/40.0, 0.0],
      &[0.0,       16.0/135.0,     0.0,            6656.0/12825.0, 28561.0/56430.0, -9.0/50.0,  2.0/55.0],
      &[0.0,       25.0/216.0,     0.0,            1408.0/2565.0,  2197.0/4104.0,   -1.0/5.0,   0.0]]
);

pub const DORMAND_PRINCE: AdaptiveRungeKutta = AdaptiveRungeKutta(
    &[&[0.0,     0.0,             0.0,            0.0,             0.0,          0.0,              0.0,          0.0],
      &[0.2,     0.2,             0.0,            0.0,             0.0,          0.0,              0.0,          0.0],
      &[0.3,     3.0/40.0,        9.0/40.0,       0.0,             0.0,          0.0,              0.0,          0.0],
      &[0.4,     44.0/45.0,      -56.0/15.0,      32.0/9.0,        0.0,          0.0,              0.0,          0.0],
      &[8.0/9.0, 19372.0/6561.0, -25360.0/2187.0, 64448.0/6561.0, -212.0/729.0,  0.0,              0.0,          0.0],
      &[1.0,     9017.0/3168.0,  -355.0/33.0,     46732.0/5247.0,  49.0/176.0,  -5103.0/18656.0,   0.0,          0.0],
      &[1.0,     35.0/384.0,      0.0,            500.0/1113.0,    125.0/192.0, -2187.0/6784.0,    11.0/84.0,    0.0],
      &[0.0,     35.0/384.0,      0.0,            500.0/1113.0,    125.0/192.0, -2187.0/6784.0,    11.0/84.0,    0.0],
      &[0.0,     5179.0/57600.0,  0.0,            7571.0/16695.0,  393.0/640.0, -92097.0/339200.0, 187.0/2100.0, 1.0/4.0]]
);


impl<'a> RungeKutta<'a> {
    pub fn order(&self) -> usize {(self.0.len()-1)}
    pub fn from_matrix(rk_matrix: &'a[&'a[f64]]) -> Result<Self, RKError> {
        match ButcherTableau::new(rk_matrix)? {
            ButcherTableau::Fixed(t) => Ok(RungeKutta(t)),
            ButcherTableau::Implicit(_) => Err(RKError::UnsupportedImplicit),
            _ => Err(RKError::NonSquareTableau(rk_matrix.len(), rk_matrix[0].len()))
        }
    }
}

impl<'a> AdaptiveRungeKutta<'a> {
    pub fn order(&self) -> usize {(self.0[0].len()-1)}
    pub fn from_matrix(rk_matrix: &'a[&'a[f64]]) -> Result<Self, RKError> {
        match ButcherTableau::new(rk_matrix)? {
            ButcherTableau::Adaptive(t) => Ok(AdaptiveRungeKutta(t)),
            ButcherTableau::Fixed(t) => Err(RKError::TooManyColumns(t.len(), t[0].len())),
            _ => Err(RKError::UnsupportedImplicit),
        }
    }
}

impl<'a> VelIntegrator for RungeKutta<'a> {
    fn step_with_vel<R:Real, S:VectorSpace<R>, V:Fn(R,S)->S, F:Fn(R,S)->S>(&self, time:R, state: &mut [S], dt:R, _:V, force:F) -> S {
        Integrator::step(self, time, state, dt, force)
    }
}

fn compute_k<R:Real, S:VectorSpace<R>, F:Fn(R, S) -> S>(tableau: &[&[f64]], time:R, state:&S, dt:R, force: F) -> Vec<S> {
    let order = tableau[0].len()-1;
    let mut k:Vec<S> = Vec::with_capacity(order);

    for i in 0..order {
        let t = time.clone() + dt.clone() * R::repr(tableau[i][0]);
        let mut y_i = state.clone();
        for j in 1..=i {
            if tableau[i][j]!=0.0 {
                y_i += k[j-1].clone() * (dt.clone() * R::repr(tableau[i][j]));
            }
        }
        k.push(force(t, y_i));
    }

    k
}

impl<'a> Integrator for RungeKutta<'a> {
    fn step<R:Real, S:VectorSpace<R>, F:Fn(R, S) -> S>(&self, time:R, state: &mut [S], dt:R, force: F) -> S {

        let order = self.order();
        let k:Vec<S> = compute_k(self.0, time, &state[0], dt.clone(), force);

        let mut j = 1;
        for k_j in k {
            if self.0[order][j]!=0.0 { state[0] += k_j * (dt.clone()*R::repr(self.0[order][j]));}
            j += 1;
        }

        state[0].clone()
    }
}

impl<'a> AdaptiveIntegrator for AdaptiveRungeKutta<'a> {
    fn adaptive_init<R:Real, S:VectorSpace<R>, M:Metric<S,R>, F:Fn(R, S) -> S>(&self, t0:R, state: S, ds:R, _force:F, _d:M) -> Box<[(R,S)]>{
        Box::new([(t0, state.clone()), (ds, state.clone())])
    }

    fn adaptive_step<R:Real, S:VectorSpace<R>, M:Metric<S,R>, F:Fn(R, S) -> S>(&self, state: &mut [(R,S)], ds:R, force:F, d:M) -> (R,S) {
        let order = self.order();
        let mut dt = state[1].0.clone();
        let time = state[0].0.clone();

        loop {
            let k:Vec<S> = compute_k(self.0, time.clone(), &state[0].1, dt.clone(), &force);

            let mut est1 = state[0].1.clone();
            let mut est2 = state[0].1.clone();

            let mut j = 1;
            for k_j in k {
                if self.0[order][j]!=0.0 { est1 += k_j.clone() * (dt.clone()*R::repr(self.0[order][j]));}
                if self.0[order+1][j]!=0.0 { est2 += k_j * (dt.clone()*R::repr(self.0[order+1][j]));}
                j += 1;
            }

            let err = d.distance(est1.clone(), est2.clone());

            if err < ds {
                let next_dt = dt.clone() * R::repr(1.5);
                state[0].0 += dt;
                state[0].1 = est1;
                state[1] = (next_dt, est2);
                return state[0].clone();
            } else {
                dt *= R::repr(0.5);
            }
        }

    }
}
