# CLF-CBF Controller
Dynamical system:

$\dot{x} = f(x) + g(x)u, f(\cdot): \mathbb{R}^{n} \rightarrow \mathbb{R}^{n}, g(\cdot): \mathbb{R}^{n} \rightarrow \mathbb{R}^{n}, u \in \mathbb{R}$

## Controller
Controller minimize:

![quality](figs/q.png)

such that:

![quality](figs/st.png)

## Examples

### ACC

![quality](figs/model_acc.png)

Velociy of host vehicle:

![Opis alternatywny](examples/figures/acc_x_2.png)

Distance between host and leading vehicle:

![Opis alternatywny](examples/figures/acc_x_3.png)

Headway time:

![Opis alternatywny](examples/figures/acc_headway.png)

Control signal:

![Opis alternatywny](examples/figures/acc_u.png)