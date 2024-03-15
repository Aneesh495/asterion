# Freeze roles before label reveals

Status: implemented. See [data ownership](../DATA.md) and [acquisition walkthrough](../WALKTHROUGHS.md).

Frames from one molecular trajectory are correlated. Random membership does not establish
independent conformations. The primary comparison preserves the declared published memberships;
a separate temporal-block protocol applies explicit guards. Candidate coordinates may be used
for allowed selection, while candidate labels remain inaccessible until a reveal. Evaluation,
validation and calibration roles are frozen before model selection.

A single freely accessible array is convenient, but it makes accidental test normalization and
candidate-label ranking hard to detect. Copying each role into separate files consumes storage
and can obscure original row identities. The chosen access layer retains author arrays externally
and validates role-specific access using immutable row identities and split records.

Every acquired row enters a hash-chained reveal generation exactly once. Training normalization
fits only the currently revealed training rows, so changing a ledger requires corresponding
normalizer and checkpoint lineage changes. Shared initial pools and paired policy trials permit
comparisons on identical evaluation identities. Retraining cost and every revealed label count
remain visible, including earlier rounds.

The checks prevent accidental boundary violations; Python objects are not a sandbox against
malicious code. A hash proves identity rather than statistical independence. Nearby conformations
across allowed memberships remain a documented limitation, and a temporal slice evaluated by a
random-pool learner must not be called guarded training. Independent verification reconstructs
membership, reveal order, normalization denominators and target author rows. Altered ledgers,
repeated rows, hidden labels and mismatched split digests reject the bundle.
