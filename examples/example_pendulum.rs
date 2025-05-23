use clf_cbf;
use nalgebra::{SMatrix, SVector};
use plotters::prelude::*;

/* Plots */
pub fn make_plot(
    signal: &Vec<f64>,
    time: &Vec<f64>,
    signal_name: &'static str,
) -> Result<(), Box<dyn std::error::Error>> {
    let signal_min = signal.iter().cloned().fold(f64::INFINITY, f64::min) - 0.1;
    let signal_max = signal.iter().cloned().fold(f64::NEG_INFINITY, f64::max) + 0.1;

    let filename = format!("examples/figures/pendulum_{}.png", signal_name);
    let description = format!("{} vs time", signal_name);

    let root_u = BitMapBackend::new(&filename, (800, 600)).into_drawing_area();
    root_u.fill(&WHITE)?;
    let mut chart_u = ChartBuilder::on(&root_u)
        .caption(description, ("sans-serif", 30))
        .margin(20)
        .x_label_area_size(40)
        .y_label_area_size(40)
        .build_cartesian_2d(0.0..time[time.len() - 1], signal_min..signal_max)?;
    chart_u.configure_mesh().draw()?;

    chart_u.draw_series(LineSeries::new(
        time.iter().cloned().zip(signal.iter().cloned()),
        &GREEN,
    ))?;

    Ok(())
}

fn draw_ellipse_from_constraint(
    x_1: &Vec<f64>,
    x_2: &Vec<f64>,
    a: f64,
    b: f64,
) -> Result<(), Box<dyn std::error::Error>> {
    use nalgebra::{Matrix2, Vector2};
    use plotters::prelude::*;

    let m = Matrix2::new(
        1.0 / (a * a),
        1.0 / (2.0 * a * b),
        1.0 / (2.0 * a * b),
        1.0 / (b * b),
    );

    let eig = m.symmetric_eigen();
    let axes = Vector2::new(
        1.0 / eig.eigenvalues[0].sqrt(),
        1.0 / eig.eigenvalues[1].sqrt(),
    );
    let q = eig.eigenvectors;

    let points: Vec<(f64, f64)> = (0..=100)
        .map(|i| {
            let theta = 2.0 * std::f64::consts::PI * (i as f64) / 100.0;
            let unit = Vector2::new(theta.cos(), theta.sin());
            let scaled = axes.component_mul(&unit);
            let rotated = q * scaled;
            (rotated[0], rotated[1])
        })
        .collect();

    let root = BitMapBackend::new(
        "examples/figures/pendulum_ellipse_constraint.png",
        (800, 600),
    )
    .into_drawing_area();
    root.fill(&WHITE)?;
    let mut chart = ChartBuilder::on(&root)
        .caption("Trajectory wth safe space", ("sans-serif", 30))
        .margin(20)
        .x_label_area_size(40)
        .y_label_area_size(40)
        .build_cartesian_2d(-1.0..1.0, -1.0..1.0)?;
    chart.configure_mesh().draw()?;

    chart.draw_series(LineSeries::new(points, &RED))?;
    chart.draw_series(LineSeries::new(
        x_1.iter().cloned().zip(x_2.iter().cloned()),
        &GREEN,
    ))?;

    Ok(())
}

/* Constants */
const M: f64 = 2.0;
const G: f64 = 9.81;
const L: f64 = 1.0;
const A: f64 = 0.25;
const B: f64 = 0.5;

/* Model Simulator */
struct ModelSimuator<const N: usize> {
    f_: clf_cbf::ModelFn<f64, N>,
    g_: clf_cbf::ModelFn<f64, N>,
    x_: SVector<f64, N>,
}

impl<const N: usize> ModelSimuator<N> {
    pub fn new(
        f: clf_cbf::ModelFn<f64, N>,
        g: clf_cbf::ModelFn<f64, N>,
        x: SVector<f64, N>,
    ) -> Self {
        Self {
            f_: f,
            g_: g,
            x_: x,
        }
    }

    pub fn make_step(&mut self, u: f64) {
        let dt = 0.01;
        let dx = (self.f_)(&self.x_) * dt + (self.g_)(&self.x_) * u * dt;
        self.x_ += dx;
    }

    pub fn get_x(&self) -> SVector<f64, N> {
        self.x_
    }
}

// Example - Pendulum
// This example simulates a pendulum using the CLF-CBF control approach.
// The pendulum is modeled as a second-order system with a control input.
// The goal is to keep the pendulum within a certain region of the state space.
pub fn main() -> Result<(), Box<dyn std::error::Error>> {
    const N: usize = 2;

    /* Define model*/
    fn f(x: &SVector<f64, N>) -> SVector<f64, N> {
        SVector::<f64, N>::from([x[1], G * x[0].sin() / L])
    }
    fn g(_x: &SVector<f64, N>) -> SVector<f64, N> {
        SVector::<f64, N>::from([0.0, 1.0 / (M * L * L)])
    }

    let x_0 = SVector::<f64, N>::from([-0.2, 0.4]);
    let mut model = ModelSimuator::<N>::new(f, g, x_0);

    // Define control functions
    fn h(x: &SVector<f64, N>) -> f64 {
        let p = x[0];
        let v = x[1];
        1.0 - p.powi(2) / A.powi(2) - v.powi(2) / B.powi(2) - p * v / (A * B)
    }
    fn h_dot(x: &SVector<f64, N>) -> SVector<f64, N> {
        let p = x[0];
        let v = x[1];
        SVector::<f64, N>::from([
            -2.0 * p / A.powi(2) - v / (A * B),
            -2.0 * v / A.powi(2) - p / (A * B),
        ])
    }
    fn v(x: &SVector<f64, N>) -> f64 {
        let p = x[0];
        let v = x[1];
        0.5 * M * L.powi(2) * v.powi(2) + M * G * L * (1.0 - p.cos())
    }
    fn v_dot(x: &SVector<f64, N>) -> SVector<f64, N> {
        let p = x[0];
        let v = x[1];
        SVector::<f64, N>::from([M * G * L * p.sin(), M * L.powi(2) * v])
    }

    // Create controller
    let controller = clf_cbf::ClfCbfControl::<N>::new(
        f,
        g,
        h,
        h_dot,
        v,
        v_dot,
        5.0,
        5.0,
        SMatrix::<f64, 2, 2>::from([[7.346189164370983e-07, 0.0], [0.0, 0.2]]),
        SVector::<f64, 2>::from([0.0, 0.0]),
    );

    // Simulate
    let mut x_1: Vec<f64> = Vec::new();
    let mut x_2: Vec<f64> = Vec::new();
    let mut u: Vec<f64> = Vec::new();
    let mut v_fun: Vec<f64> = Vec::new();
    let mut h_fun: Vec<f64> = Vec::new();

    for _ in 0..1000 {
        let x = model.get_x();
        let control = controller.get_control(&x);
        model.make_step(control);

        let (v_sample, h_sample) = controller.get_control_functions_value(&x);

        let x = model.get_x();
        x_1.push(x[0]);
        x_2.push(x[1]);
        u.push(control);

        h_fun.push(h_sample);
        v_fun.push(v_sample);
    }

    // Plot
    let time: Vec<f64> = (0..u.len()).map(|i| i as f64 * 0.01).collect();

    make_plot(&x_1, &time, "x_1")?;
    make_plot(&x_2, &time, "x_2")?;
    make_plot(&u, &time, "u")?;
    draw_ellipse_from_constraint(&x_1, &x_2, A, B)?;

    make_plot(&h_fun, &time, "h")?;
    make_plot(&v_fun, &time, "v")?;

    Ok(())
}
