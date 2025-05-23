# CLF-CBF Controller
Dynamical system:

$
 \dot{x} = f(x) + g(x)u, f(\cdot): \mathbb{R}^{n} \rightarrow \mathbb{R}^{n}, g(\cdot): \mathbb{R}^{n} \rightarrow \mathbb{R}^{n}, u \in \mathbb{R} 
$

## Controller
Controller minimize:

$
    \begin{bmatrix} u & p \end{bmatrix} Q \begin{bmatrix} u \\ p \end{bmatrix}
$

such that:

$
    \begin{array}{l}
        L_{f}V + L_{g}Vu \leq -\alpha V - \delta p\\
        L_{f}h + L_{g}hu \geq -\alpha h
    \end{array}
$

## Examples

### ACC

$
    \left\{
    \begin{array}{l}
        \dot{x}_{1} = x_{2}\\
        \dot{x}_{2} = -\frac{1}{m}(f_{0} + f_{1}x_{2} + f_{2}x_{2}^{2}) + \frac{1}{m}u\\
        \dot{x}_{3} = v_{0} - x_{2}
    \end{array}
    \right\}
$

Velociy of host vehicle:

![Opis alternatywny](examples/figures/acc_x_2.png)

Distance between host and leading vehicle:

![Opis alternatywny](examples/figures/acc_x_3.png)

Headway time:

![Opis alternatywny](examples/figures/acc_headway.png)

Control signal:

![Opis alternatywny](examples/figures/acc_u.png)