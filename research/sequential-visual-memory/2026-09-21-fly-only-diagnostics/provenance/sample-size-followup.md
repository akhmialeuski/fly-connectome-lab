### Sample-size limitation and follow-up requested during execution

The user asked whether 280 training examples are too few. Yes: the smoke study has only **14 photographs per identity across 20 identities**, so it is a diagnostic cohort, not an adequate standalone demonstration of generalization. The main cohort increases the class count to 100 and the total training size to 1,400 but still has **14 photographs per identity**; it is not a controlled learning-curve experiment.

The observed smoke persistent readout fits 280/280 training examples but only 4/60 validation examples. This demonstrates training-set memorization and weak observed generalization. It does not distinguish limited independent training images from insufficient identity signal at the neural readout.

Add a separate, preregistered learning-curve study after the current diagnostic queue: keep identities, validation membership, representation, and classifier-selection rules fixed, then use nested balanced training subsets (e.g. 2, 4, 8, 14 photographs per identity) with repeated deterministic subset seeds. Fit every scaler/PCA within its training/CV subset and report identity-cluster uncertainty. Compare encoded-input and neural representations to see whether additional examples help both stages similarly.

The existing cohort cannot support an arbitrary increase beyond 14 independent training photographs per identity while preserving all historical splits and reserves. Audit source availability before defining any larger-data cohort; a changed cohort must be reported separately, not spliced into the same learning curve. Do not borrow validation, historical-test, or reserved photographs, and do not count augmented views as independent photographs. No specialized or pretrained vision models are proposed.

Current registered attempts continue unchanged. This is a follow-up hypothesis and protocol direction, not a result that more data has already solved recognition.
