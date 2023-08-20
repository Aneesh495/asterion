# Publish tensor generations atomically

Status: implemented.

Pickle loading can execute serialized code. Safetensors stores weights and optimizer tensors; JSON stores container structure and provenance. Atomic directory publication avoids partial generations. Resume requires identical source, data, split, configuration and reveal lineage. This costs additional files but makes interrupted state independently inspectable.

The consequences are exercised through the standard model, training, simulation or verifier path. Full scientific acceptance remains a separate execution obligation.
