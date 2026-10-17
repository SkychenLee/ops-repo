# MoE LoRA direct-launch kernels

Migrated from committed vllm-ascend `misha` at
`72b7912616e0e296e64f0ca0003afcc2ef7ce6bc`. This directory contains seven
direct-launch AscendC kernels, not an msopgen OPP operator. It intentionally
has no `operator.json`. Source hashes and changes are in `provenance.json`.

Build with `OPS_REPO_BUILD_LORA=1 OPS_REPO_SOC=ascend910_93` through the root
Python packaging entry point. Only A2/A3 are configured. Device execution
has not been validated as part of this migration.

See [migration, build and API documentation](../../../docs/moe_lora.md).
