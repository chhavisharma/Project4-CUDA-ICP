set(SOURCE_FILES
    "common.h"
    "common.cu"
    "picp.h"
    "picp.cu"
    )

cuda_add_library(PICP
    ${SOURCE_FILES}
    OPTIONS -arch=sm_61
    )
