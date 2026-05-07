# Validation

## 1. Scope

This validation checks whether the numerical point-kinetics implementation
reproduces expected qualitative and semi-quantitative behaviour. It covers the
six-group point-kinetics model, numerical transient response, prompt-jump
behaviour, period extraction, and a basic Inhour-equation comparison.

## 2. Parameters

The file `configs/pwr_reference.yaml` contains representative PWR-like U-235
point-kinetics parameters and illustrative lumped thermal parameters.

These parameters are not plant-specific licensed data and should not be used for
safety analysis.

The kinetics values are used for validation. The thermal values are included so
the same YAML file can configure the educational simulator, but Phase 2
validation focuses on the kinetics-only reference cases.

## 3. Validation Cases

- Zero reactivity steady-state: a critical initial condition is simulated with
  `rho = 0`. Relative power should remain constant.
- Negative reactivity decay: a step to `rho = -0.5 beta_eff` should reduce the
  neutron population.
- Delayed-supercritical growth: a step to `rho = 0.5 beta_eff` should increase
  the neutron population on a delayed-neutron timescale.
- Prompt-supercritical growth: a step to `rho = 1.2 beta_eff` should grow much
  faster than the delayed-supercritical case.
- Prompt jump approximation: a small sub-prompt-critical step is compared with
  `n_plus / n_minus ~= beta_eff / (beta_eff - rho)`.
- Reactor period extraction: a linear fit of `log(n)` versus time estimates the
  inverse period in exponential-growth regions.
- Inhour equation comparison: the extracted numerical inverse period for
  `rho = 0.5 beta_eff` is compared with the positive root of the Inhour equation.

## 4. Results

The validation script applies these numerical criteria:

- zero reactivity: `max(abs(n(t) - n0)) / n0 < 1e-5`;
- negative reactivity: `n_final < 0.8 n_initial`;
- delayed-supercritical reactivity: `n_final > n_initial`;
- prompt-supercritical reactivity: extracted prompt growth rate is greater than
  the extracted delayed-supercritical growth rate;
- prompt jump: relative error below 10 percent for the selected extraction
  point;
- Inhour comparison: relative inverse-period error below 10 percent with a
  high-quality exponential fit.

Run:

```bash
python examples/run_validation.py
pytest
```

The validation script writes PNG plots to `validation/outputs/`.

## 5. Limitations

- Point kinetics ignores spatial flux-shape effects.
- The reference parameters are representative only.
- Thermal feedback is lumped and illustrative.
- No neutron transport calculation is performed.
- No core design, operational, safety, or licensing claim is made.
- Validation is behavioural and educational, not licensing-grade.
- Prompt-jump extraction is approximate because the physical jump is represented
  by a fast numerical transient over a finite time grid.

## 6. Conclusion

The model reproduces the expected point-kinetics behaviours for critical,
subcritical, delayed-supercritical, and prompt-supercritical regimes. The
numerical solver is therefore considered validated for educational and
demonstrative point-kinetics studies within the assumptions stated above.
