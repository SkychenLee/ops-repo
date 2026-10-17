"""Source namespace for future Triton Ascend kernels.

Packaged as ``ops_repo.ops.triton.triton_core`` by the repository's root
setup.py. No executable kernel is implemented here. Importing the installed
package path first runs ops_repo's root AscendC initialization, so the complete
AscendC runtime is required even though this source file contains no imports.
"""

__all__ = []
