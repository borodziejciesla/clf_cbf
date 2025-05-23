use clf_cbf;
use nalgebra::{SMatrix, SVector};
use plotters::prelude::*;

/* Plots */
pub fn make_plot(
    signal: &Vec<f64>,
    time: &Vec<f64>,
    signal_name: &'static str,
) -> Result<(), Box<dyn std::error::Error>> {
    let signal_min = signal.iter().cloned().fold(f64::INFINITY, f64::min);
    let signal_max = signal.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

    let filename = format!("examples/figures/acc_{}.png", signal_name);
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

// Example: Car following model
// This example simulates a car following model using the CLF-CBF approach.
// The model consists of three states: x1 (position), x2 (velocity), and x3 (headway).
// The control input is the acceleration of the car.
// The goal is to maintain a safe distance from the car in front while also achieving a desired velocity.
// The CLF-CBF approach is used to ensure that the car follows the desired trajectory while avoiding collisions.
// The model is defined by the following equations:
// x1_dot = x2
// x2_dot = -f(x2) / M
// x3_dot = V0 - x2
// where f(x2) is the friction force, M is the mass of the car, and V0 is the desired velocity.
// The control input is computed using the CLF-CBF approach, which ensures that the car follows the desired trajectory while avoiding collisions.
// The model is simulated for a certain number of steps, and the results are plotted using the plotters library.
// The plots show the position, velocity, and control input over time.
// The results can be used to analyze the performance of the CLF-CBF approach in maintaining a safe distance from the car in front while achieving the desired velocity.

pub fn main() -> Result<(), Box<dyn std::error::Error>> {
    const V0: f64 = 15.0;
    const VD: f64 = 25.0;
    const M: f64 = 1650.0;
    const G: f64 = 9.81;
    const F0: f64 = 0.1;
    const F1: f64 = 5.0;
    const F2: f64 = 0.25;
    const CD: f64 = 0.3;
    const T: f64 = 1.0;

    const N: usize = 3;

    /* Define model*/
    fn f(x: &SVector<f64, N>) -> SVector<f64, N> {
        let fr = F0 + F1 * x[1] + F2 * x[1].powi(2);
        SVector::<f64, N>::from([x[1], -fr / M, V0 - x[1]])
    }
    fn g(_x: &SVector<f64, N>) -> SVector<f64, N> {
        SVector::<f64, N>::from([0.0, 1.0 / M, 0.0])
    }

    let x_0 = SVector::<f64, N>::from([0.0, 20.0, 100.0]);
    let mut model = ModelSimuator::<N>::new(f, g, x_0);

    // Define control functions
    fn h(x: &SVector<f64, N>) -> f64 {
        let d = x[2];
        d - T * x[1] - 0.5 * (x[1] - V0).powi(2) / (CD * G)
    }
    fn h_dot(x: &SVector<f64, N>) -> SVector<f64, N> {
        SVector::<f64, N>::from([0.0, -T - (x[1] - V0) / (CD * G), 1.0])
    }
    fn v(x: &SVector<f64, N>) -> f64 {
        (x[1] - VD).powi(2)
    }
    fn v_dot(x: &SVector<f64, N>) -> SVector<f64, N> {
        SVector::<f64, N>::from([0.0, 2.0 * (x[1] - VD), 0.0])
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
        SMatrix::<f64, 2, 2>::from([[7.346189164370983e-07, 0.0], [0.0, 0.02]]),
        SVector::<f64, 2>::from([0.0, 0.0]),
    );

    // Simulate
    let mut x_1: Vec<f64> = Vec::new();
    let mut x_2: Vec<f64> = Vec::new();
    let mut x_3: Vec<f64> = Vec::new();
    let mut u: Vec<f64> = Vec::new();
    let mut headway: Vec<f64> = Vec::new();

    for _ in 0..2000 {
        let x = model.get_x();
        let control = controller.get_control(&x);
        model.make_step(control);

        let x = model.get_x();
        x_1.push(x[0]);
        x_2.push(x[1]);
        x_3.push(x[2]);
        u.push(control);
        headway.push(x[2] / x[1]);
    }

    // Plot
    let time: Vec<f64> = (0..u.len()).map(|i| i as f64 * 0.01).collect();
    make_plot(&x_2, &time, "x_2")?;
    make_plot(&x_3, &time, "x_3")?;
    make_plot(&u, &time, "u")?;
    make_plot(&headway, &time, "headway")?;

    Ok(())
}
