#ifndef OPS_REPO_ADD_TILING_H
#define OPS_REPO_ADD_TILING_H

#include "register/tilingdata_base.h"

namespace optiling {

BEGIN_TILING_DATA_DEF(OpsRepoAddTilingData)
    TILING_DATA_FIELD_DEF(uint64_t, totalLength);
    TILING_DATA_FIELD_DEF(uint32_t, tileLength);
    TILING_DATA_FIELD_DEF(uint32_t, blockDim);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(OpsRepoAdd, OpsRepoAddTilingData)

}  // namespace optiling

#endif  // OPS_REPO_ADD_TILING_H
