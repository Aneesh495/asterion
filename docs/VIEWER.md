# Retained molecular observations

`make demo` exports standalone `viewer.html` and `failure-viewer.html` files from the run's
actual held-out predictions and trajectories. The primary viewer uses the first equivariant
model at the campaign's largest label budget. Each file embeds data and the shipped local
canvas renderer; it requires no server or external assets.

Held-out observations show learned and reference forces with a shared arrow scale, geometry
force MAE and energy error. Generated trajectory frames have no reference labels. Up to 1000
frames plus the final frame are retained for animation; complete energy, thermostat heat and
uncertainty histories remain embedded. A rejected initial evaluation has no physical energy,
force or uncertainty observation and displays its retained failure reason at zero steps.

When acquisition observations are present, export first independently checks every member's
training, saved-weight probes, separate calibration roles, committee arrays, reveal decisions,
cost clocks, selected-coordinate diversity and paired policy outcomes. The molecule, trial,
policy and round controls select a real reveal snapshot. The coordinate view shows the exact
released geometries in that snapshot and distinguishes newly acquired rows from earlier rows.
It attaches no force or energy observation to those coordinates. The force layers are disabled.

The budget plots compare uniform, diversity and uncertainty-plus-diversity policies on the
same held-out identities. Horizontal positions use the actual distinct-label budgets. The table
includes geometry force MAE, tied-rank error correlation, observed coverage, mean calibrated
bound, measured round seconds and nearest descriptor distance. Constant scores or errors make
the ranking unavailable. A paired interval uses the recorded original-time blocks; it does not
establish independent geometries or a useful acquisition advantage.

Round identities include the source digest, author dataset hash, split, reveal hash, three
selected checkpoint identities, and hashes of the raw uncertainty, cost and paired observations.
Missing acquisition is explicitly unavailable in a diagnostic viewer. Full-profile export
requires the declared acquisition population. The full verifier checks both embedded viewers
against raw predictions, trajectory frames and histories, acquisition observations and the
frozen executable template. Altering embedded observations while rehashing the surrounding
bundle does not make the viewer pass that check.

The local browser check covers the actual exported artifacts, policy/round changes, acquired
coordinates, disabled unavailable overlays, narrow-layout overflow and console errors. Passing
that check establishes the viewer path for the measured run, not full research acceptance.
