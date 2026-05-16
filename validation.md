# Validation

The main validation notes and Phase 2 plots live in `validation/validation.md`
and `validation/outputs/`.

## Phase 3 Reactivity Components

Phase 3 does not add licensing-grade validation. It adds physically
interpretable reactivity channels and validates their sign conventions and
qualitative transient response.

The control rod worth, boron worth, and temperature coefficients are
representative educational parameters, not plant-specific design data.

Automated tests check pcm conversion, control rod and boron absorber signs,
linear temperature feedback signs, total reactivity summation, and qualitative
power response for rod motion and boron concentration changes.

## Thermal Feedback Validation

Phase 4 validates the expected qualitative coupled behaviour:

```text
positive reactivity insertion
→ power rise
→ fuel/moderator temperature rise
→ negative feedback
→ reduced net reactivity
```

Automated tests exercise the lumped thermal derivatives, temperature rise under
power, negative fuel and moderator feedback signs, coupled positive insertion
behaviour, and reduced power rise relative to the same insertion without
thermal feedback.

The thermal model is a lumped two-node educational model. It is not a
thermal-hydraulic system code, does not resolve coolant channels, does not
model boiling, does not model spatial power distributions, and is not suitable
for safety analysis.

The thermal parameters are representative educational values selected for
stable demonstrative transients. They are not plant-specific design, operating,
licensing, or safety data.
