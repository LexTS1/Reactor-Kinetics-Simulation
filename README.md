# Reactor Kinetics Simulator

Educational six-group point-kinetics reactor transient simulator. The project
has no GUI, web app, dashboard, 3D model, or animation.

## Running

```bash
python examples/run_mvp.py --no-show
python examples/run_validation.py
python examples/run_thermal_feedback.py
pytest
```

## Reactivity Components

Phase 3 replaces a single generic reactivity input with composable,
physically interpretable reactivity channels:

```text
rho_total =
    rho_external
  + rho_control_rods
  + rho_boron
  + rho_fuel_temperature
  + rho_moderator_temperature
```

All reactivity values are dimensionless delta-k/k. Control rod insertion is
defined as a fraction where `1.0` is fully inserted and `0.0` is fully
withdrawn. Control rods are neutron absorbers, so inserting rods adds negative
reactivity and withdrawing rods adds positive reactivity relative to the
reference rod position.

Soluble boron is also a neutron absorber. Increasing boron concentration adds
negative reactivity; boron dilution adds positive reactivity. Fuel and
moderator temperature feedback are linearized around reference temperatures:

```text
rho_fuel_temperature = alpha_fuel (Tf - Tf_ref)
rho_moderator_temperature = alpha_moderator (Tm - Tm_ref)
```

Negative temperature coefficients give stabilising feedback: temperature
increases insert negative reactivity. `rho_external` remains a generic
user-defined perturbation for scenarios that are not yet represented by a
specific physical channel.

Run the Phase 3 examples from the repository root:

```bash
python examples/run_reactivity_components.py
```

Plots are saved automatically to:

```text
reactivity_outputs/
```

The coefficients in `configs/pwr_reference.yaml` are representative educational
parameters, not plant-specific design or safety data.

## Thermal Feedback Model

Phase 4 adds an explicit lumped two-node fuel-moderator thermal model coupled
to the six-group point kinetics. The coupled state is:

```text
y = [n, C1, C2, C3, C4, C5, C6, Tf, Tm]
```

The thermal equations are:

```text
Cf dTf/dt = P(t) - Hfm (Tf - Tm)

Cm dTm/dt = Hfm (Tf - Tm) - Hmc (Tm - Tc)
```

with `P(t) = power_scale n(t)`. Temperature feedback is evaluated by the
reactivity component model:

```text
rho_feedback = alpha_f (Tf - Tf_ref) + alpha_m (Tm - Tm_ref)
```

The neutron kinetics and thermal model are coupled because power affects
temperature, and temperature affects reactivity. With negative fuel and
moderator temperature coefficients, rising temperatures insert negative
reactivity and reduce the net reactivity.

Run the Phase 4 thermal examples from the repository root:

```bash
python examples/run_thermal_feedback.py
```

Plots are saved automatically to:

```text
thermal_outputs/
```
