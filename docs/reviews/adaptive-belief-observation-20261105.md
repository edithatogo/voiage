# Adaptive trial belief observation — 5 November 2026

On source candidate `fc50e7f15955f820e973b4dd63335adb67eed02c`,
`voiage/methods/adaptive.py::_calculate_net_benefits` accepts updated treatment
effects and control rates but its returned benefit array depends on observed
outcome counts, patient costs, adjusted sample size and random noise; clipped
belief values are not used in the result. CodeQL alert #864 points to one
redundant assignment in that path.

A Python 3.15.1 probe called the helper twice with identical counts, costs,
sample size and NumPy seed, changing treatment effects from 0 to 0.8 and
control rates from 0.1 to 0.9. `numpy.array_equal` returned true. This is a
behavioral observation, not a complete scientific validation or proof of the
correct economic model. Before changing the calculation, identify the intended
units and relationship between posterior beliefs, observed outcomes, costs and
net benefit; freeze an independent analytic or published reference and add a
witness that would fail on the current behavior. Do not treat the CodeQL alert
as fixed or dismiss it.
