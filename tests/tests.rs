use clf_cbf;
use nalgebra::{SMatrix, SVector};

#[cfg(test)]
mod tests {
    use super::*;

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

    /* Define model - double integrator */
    fn f_di(x: &SVector<f64, 2>) -> SVector<f64, 2> {
        SVector::<f64, 2>::from([x[1], 0.0])
    }
    fn g_di(_x: &SVector<f64, 2>) -> SVector<f64, 2> {
        SVector::<f64, 2>::from([0.0, 1.0])
    }

    #[test]
    fn test_double_integrator() {
        let x_0 = SVector::<f64, 2>::from([0.5, 0.5]);
        let mut model = ModelSimuator::<2>::new(f_di, g_di, x_0);

        fn h<const N: usize>(x: &SVector<f64, N>) -> f64 {
            1.0 - x.dot(&x)
        }
        fn h_dot(x: &SVector<f64, 2>) -> SVector<f64, 2> {
            SVector::<f64, 2>::from([-2.0 * x[0], -2.0 * x[1]])
        }
        fn v(x: &SVector<f64, 2>) -> f64 {
            x.dot(&x)
        }
        fn v_dot(x: &SVector<f64, 2>) -> SVector<f64, 2> {
            SVector::<f64, 2>::from([2.0 * x[0], 2.0 * x[1]])
        }

        let control = clf_cbf::ClfCbfControl::<2>::new(
            f_di,
            g_di,
            h,
            h_dot,
            v,
            v_dot,
            5.0,
            5.0,
            SMatrix::<f64, 2, 2>::from([[7.346189164370983e-07, 0.0], [0.0, 0.02]]),
            SVector::<f64, 2>::from([0.0, 0.0]),
        );
        let x = model.get_x();
        let result = control.get_control(&x);
        assert_ne!(result, 0.0);
        model.make_step(result);
    }

    #[test]
    fn test_dummy2() {
        fn f<const N: usize>(x: &SVector<f64, N>) -> SVector<f64, N> {
            *x
        }
        fn g<const N: usize>(x: &SVector<f64, N>) -> SVector<f64, N> {
            *x
        }
        fn h<const N: usize>(x: &SVector<f64, N>) -> f64 {
            x.dot(&x)
        }
        fn h_dot<const N: usize>(x: &SVector<f64, N>) -> SVector<f64, N> {
            *x
        }
        fn v<const N: usize>(x: &SVector<f64, N>) -> f64 {
            x.dot(&x)
        }
        fn v_dot<const N: usize>(x: &SVector<f64, N>) -> SVector<f64, N> {
            *x
        }

        let control = clf_cbf::ClfCbfControl::<2>::new(
            f::<2>,
            g::<2>,
            h::<2>,
            h_dot::<2>,
            v::<2>,
            v_dot::<2>,
            5.0,
            5.0,
            SMatrix::<f64, 2, 2>::from([[7.346189164370983e-07, 0.0], [0.0, 0.02]]),
            SVector::<f64, 2>::from([0.0, 0.0]),
        );
        let x = SVector::<f64, 2>::from([0.0, 0.0]);
        let result = control.get_control(&x);
        assert_eq!(result, 0.0);
    }
}
