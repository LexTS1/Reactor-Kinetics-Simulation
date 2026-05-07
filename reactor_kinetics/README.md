# Reactor Kinetics Physics MVP

This repository contains a Python physics MVP for a six-group point-kinetics
reactor transient simulator. It intentionally has no GUI, web app, dashboard,
3D model, or animation.

## Point Kinetics

Point kinetics models the time evolution of neutron population when spatial
effects are collapsed into one representative reactor amplitude. The neutron
population `n` is proportional to reactor power. Delayed neutron precursor
concentrations `C_i` represent radioactive fission products that later emit
delayed neutrons.

The six-group point-kinetics equations are:
```text
dn/dt = ((rho(t) - beta_eff) / Lambda) n + sum(lambda_i C_i)

dC_i/dt = (beta_i / Lambda) n - lambda_i C_i
```

where `rho` is dimensionless reactivity, `beta_i` are delayed neutron fractions,
`beta_eff = sum(beta_i)`, `Lambda` is the prompt neutron generation time in
seconds, and `lambda_i` are precursor decay constants in `1/s`.

## Reactivity Regimes

At critical steady state, `rho = 0` and power is steady. For `n0 = 1`, the
critical precursor inventory is initialized as:
```text
C_i0 = beta_i / (Lambda lambda_i) n0
```

The regimes demonstrated by the MVP are:

- Subcritical: `rho < 0`; power decays.
- Critical: `rho = 0`; power is steady.
- Delayed supercritical: `0 < rho < beta_eff`; power rises on a delayed-neutron
  timescale.
- Prompt supercritical: `rho > beta_eff`; power rises on the much faster prompt
  neutron timescale.

## Thermal Feedback

The optional lumped thermal model tracks fuel and moderator temperatures:
```text
Cf dTf/dt = P(t) - Hfm (Tf - Tm)

Cm dTm/dt = Hfm (Tf - Tm) - Hmc (Tm - Tc)
```

Temperature feedback is coupled into reactivity as:
```text
rho_feedback = alpha_f (Tf - Tf0) + alpha_m (Tm - Tm0)
```

Negative `alpha_f` and `alpha_m` demonstrate qualitative stabilisation: a power
rise heats the fuel and moderator, which inserts negative reactivity.

## Running The MVP

Install the required Python packages if needed:
```bash
pip install numpy scipy matplotlib
```

Run all demonstration cases:
```bash
python examples/run_mvp.py
```

The script runs:
- delayed-supercritical positive step insertion,
- negative step insertion,
- prompt-supercritical insertion,
- ramp insertion,
- positive step insertion with negative thermal feedback.

Each case produces Matplotlib summary plots showing relative power, logarithmic
relative power, delayed neutron precursor concentrations, and reactivity. The
thermal-feedback case also shows fuel and moderator temperatures.

To save plots instead of only showing them:
```bash
python examples/run_mvp.py --save-dir examples/output
```
