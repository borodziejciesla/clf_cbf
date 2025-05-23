use nalgebra::{SMatrix, SVector};
use quadprog::solve_qp;

pub type ModelFn<T, const N: usize> = fn(&SVector<T, N>) -> SVector<T, N>;
pub type LieDerivative<T, const N: usize> = fn(&SVector<T, N>) -> SVector<T, N>;
pub type Function<T, const N: usize> = fn(&SVector<T, N>) -> T;

pub struct ClfCbfControl<const N: usize> {
    f_: ModelFn<f64, N>,
    g_: ModelFn<f64, N>,

    h_: Function<f64, N>,
    h_dot_: LieDerivative<f64, N>,

    v_: Function<f64, N>,
    v_dot_: LieDerivative<f64, N>,

    alpha_: f64,
    gamma_: f64,
    q_: SMatrix<f64, 2, 2>,
    c_: SVector<f64, 2>,
}

impl<const N: usize> ClfCbfControl<N> {
    pub fn new(
        f: ModelFn<f64, N>,
        g: ModelFn<f64, N>,
        h: Function<f64, N>,
        h_dot: LieDerivative<f64, N>,
        v: Function<f64, N>,
        v_dot: LieDerivative<f64, N>,
        alpha: f64,
        gamma: f64,
        q: SMatrix<f64, 2, 2>,
        c: SVector<f64, 2>,
    ) -> Self {
        Self {
            f_: f,
            g_: g,
            h_: h,
            h_dot_: h_dot,
            v_: v,
            v_dot_: v_dot,
            alpha_: alpha,
            gamma_: gamma,
            q_: q,
            c_: c,
        }
    }

    pub fn get_control(&self, x: &SVector<f64, N>) -> f64 {
        // Compute the control input using the CLF-CBF approach
        let (v, h) = self.get_control_functions_value(x);
        let (v_dot, h_dot) = self.get_functions_values(x);
        let (f, g) = self.get_model_values(x);

        // Define the QP problem:
        let (mut q, a, b) = self.get_parameters_for_qp(v_dot, v, h_dot, h, f, g);

        // Solve QP problem
        let solution = solve_qp(
            q.as_mut_slice(),
            self.c_.as_slice(),
            a.as_slice(),
            b.as_slice(),
            0,
            false,
        );

        match solution {
            Ok(solution) => {
                let control = solution.sol;
                control[0]
            }
            Err(e) => {
                println!("QP Solution error: {:?}", e);
                0.0
            }
        }
    }

    pub fn get_control_functions_value(&self, x: &SVector<f64, N>) -> (f64, f64) {
        // 1. Compute the function h
        let h = (self.h_)(&x);
        // 2. Compute the function v
        let v = (self.v_)(&x);

        (v, h)
    }

    fn get_functions_values(&self, x: &SVector<f64, N>) -> (SVector<f64, N>, SVector<f64, N>) {
        // 3. Compute the Lie derivative of the function h
        let h_dot = (self.h_dot_)(&x);
        // 4. Compute the Lie derivative of the function v
        let v_dot = (self.v_dot_)(&x);

        (v_dot, h_dot)
    }

    fn get_model_values(&self, x: &SVector<f64, N>) -> (SVector<f64, N>, SVector<f64, N>) {
        // 5. Compute the function g
        let g = (self.g_)(&x);
        // 6. Compute the function f
        let f = (self.f_)(&x);

        (f, g)
    }

    fn get_parameters_for_qp(
        &self,
        v_dot: SVector<f64, N>,
        v: f64,
        h_dot: SVector<f64, N>,
        h: f64,
        f: SVector<f64, N>,
        g: SVector<f64, N>,
    ) -> (SMatrix<f64, 2, 2>, SMatrix<f64, 2, 2>, SMatrix<f64, 2, 1>) {
        let lfv = v_dot.dot(&f);
        let lgv = v_dot.dot(&g);
        let lfh = h_dot.dot(&f);
        let lgh = h_dot.dot(&g);

        // minimize 0.5 * u^T * Q * u + c^T * u
        // subject to A * u <= bs
        // where Q = identity matrix, c = 0, A = [h_dot'*g; -v_dot'*g], b = [-alpha*h - h_dot'*f; gamma*v + v_dot'*f]
        //let q = SMatrix::<f64, 2, 2>::from([[7.346189164370983e-07, 0.0], [0.0, 0.02]]);
        let q = self.q_;
        // let c = SVector::<f64, 2>::zeros();
        let a = SMatrix::<f64, 2, 2>::from([[lgv, -1.0], [-lgh, 0.0]]);
        let b = SVector::<f64, 2>::from([-lfv - self.alpha_ * v, lfh + self.gamma_ * h]);

        (q, a, b)
    }
}

/********************** Test Private Methods **********************/
#[cfg(test)]
mod tests {
    use super::*;

    /* Define model - double integration */
    fn f_di(x: &SVector<f64, 2>) -> SVector<f64, 2> {
        SVector::<f64, 2>::from([x[1], 0.0])
    }
    fn g_di(_x: &SVector<f64, 2>) -> SVector<f64, 2> {
        SVector::<f64, 2>::from([0.0, 1.0])
    }
    /* Define control functions */
    fn h_di<const N: usize>(x: &SVector<f64, N>) -> f64 {
        1.0 - x.dot(&x)
    }
    fn h_dot_di(x: &SVector<f64, 2>) -> SVector<f64, 2> {
        SVector::<f64, 2>::from([-2.0 * x[0], -2.0 * x[1]])
    }
    fn v_di(x: &SVector<f64, 2>) -> f64 {
        x.dot(&x)
    }
    fn v_dot_di(x: &SVector<f64, 2>) -> SVector<f64, 2> {
        SVector::<f64, 2>::from([2.0 * x[0], 2.0 * x[1]])
    }

    /* Tests */
    #[test]
    fn test_model_values() {
        let controller = ClfCbfControl::<2>::new(
            f_di,
            g_di,
            h_di,
            h_dot_di,
            v_di,
            v_dot_di,
            5.0,
            5.0,
            SMatrix::<f64, 2, 2>::identity(),
            SVector::<f64, 2>::zeros(),
        );
        let x = SVector::<f64, 2>::from([1.0, 1.0]);
        let (f, g) = controller.get_model_values(&x);

        // Check if empty
        assert!(!g.is_empty());
        assert!(!f.is_empty());

        // Check value
        assert_eq!(g, g_di(&x));
        assert_eq!(f, f_di(&x));
    }

    #[test]
    fn test_function_values() {
        let controller = ClfCbfControl::<2>::new(
            f_di,
            g_di,
            h_di,
            h_dot_di,
            v_di,
            v_dot_di,
            5.0,
            5.0,
            SMatrix::<f64, 2, 2>::identity(),
            SVector::<f64, 2>::zeros(),
        );
        let x = SVector::<f64, 2>::from([1.0, 1.0]);
        let (v, h) = controller.get_control_functions_value(&x);
        let (v_dot, h_dot) = controller.get_functions_values(&x);

        // Check if empty
        assert!(!h_dot.is_empty());
        assert!(!v_dot.is_empty());

        // Check value
        assert_eq!(h, h_di(&x));
        assert_eq!(v, v_di(&x));
        assert_eq!(h_dot, h_dot_di(&x));
        assert_eq!(v_dot, v_dot_di(&x));
    }

    #[test]
    fn test_qp_parameters() {
        let controller = ClfCbfControl::<2>::new(
            f_di,
            g_di,
            h_di,
            h_dot_di,
            v_di,
            v_dot_di,
            5.0,
            5.0,
            SMatrix::<f64, 2, 2>::identity(),
            SVector::<f64, 2>::zeros(),
        );
        let x_1 = 0.5;
        let x_2 = 0.5;
        let x = SVector::<f64, 2>::from([x_1, x_2]);
        let (f, g) = controller.get_model_values(&x);
        let (v, h) = controller.get_control_functions_value(&x);
        let (v_dot, h_dot) = controller.get_functions_values(&x);
        let (_, a, b) = controller.get_parameters_for_qp(v_dot, v, h_dot, h, f, g);

        // Check if empty
        assert!(!a.is_empty());
        assert!(!b.is_empty());

        // Check value
        let a_expected = SMatrix::<f64, 2, 2>::from([[2.0 * x_2, -1.0], [2.0 * x_2, 0.0]]);
        assert_eq!(a, a_expected);
        let alpha = 5.0;
        let gamma = 5.0;
        let b_expected = SVector::<f64, 2>::from([
            -alpha * v - (2.0 * x_1 * x_2),
            gamma * h + (-2.0 * x_1 * x_2),
        ]);
        assert_eq!(b, b_expected);
    }
}
