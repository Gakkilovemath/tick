set(TICK_HAWKES_INCLUDE_DIR ${PROJECT_SOURCE_DIR}/include/tick/hawkes/model)

add_library(tick_hawkes_model EXCLUDE_FROM_ALL
    ${TICK_HAWKES_INCLUDE_DIR}/base/hawkes_model.h
    ${TICK_HAWKES_INCLUDE_DIR}/base/hawkes_fixed_kern_loglik.h
    ${TICK_HAWKES_INCLUDE_DIR}/hawkes_fixed_expkern_loglik.h
    ${TICK_HAWKES_INCLUDE_DIR}/hawkes_fixed_sumexpkern_loglik.h
    ${TICK_HAWKES_INCLUDE_DIR}/hawkes_fixed_expkern_leastsq.h
    ${TICK_HAWKES_INCLUDE_DIR}/hawkes_fixed_sumexpkern_leastsq.h
    ${TICK_HAWKES_INCLUDE_DIR}/hawkes_utils.h
    ${TICK_HAWKES_INCLUDE_DIR}/base/hawkes_list.h
    ${TICK_HAWKES_INCLUDE_DIR}/variants/hawkes_leastsq_list.h
    ${TICK_HAWKES_INCLUDE_DIR}/variants/hawkes_fixed_expkern_leastsq_list.h
    ${TICK_HAWKES_INCLUDE_DIR}/variants/hawkes_fixed_expkern_loglik_list.h
    ${TICK_HAWKES_INCLUDE_DIR}/variants/hawkes_fixed_kern_loglik_list.h
    ${TICK_HAWKES_INCLUDE_DIR}/base/hawkes_single.h
    ${TICK_HAWKES_INCLUDE_DIR}/variants/hawkes_fixed_sumexpkern_leastsq_list.h
    ${TICK_HAWKES_INCLUDE_DIR}/variants/hawkes_fixed_sumexpkern_loglik_list.h
    variants/hawkes_fixed_sumexpkern_loglik_list.cpp
    variants/hawkes_fixed_sumexpkern_leastsq_list.cpp
    variants/hawkes_fixed_kern_loglik_list.cpp
    variants/hawkes_fixed_expkern_loglik_list.cpp
    variants/hawkes_leastsq_list.cpp
    variants/hawkes_fixed_expkern_leastsq_list.cpp
    base/hawkes_fixed_kern_loglik.cpp
    base/hawkes_single.cpp
    base/hawkes_model.cpp
    base/hawkes_list.cpp
    hawkes_fixed_sumexpkern_leastsq.cpp
    hawkes_fixed_expkern_leastsq.cpp
    hawkes_fixed_sumexpkern_loglik.cpp
    hawkes_fixed_expkern_loglik.cpp
    hawkes_utils.cpp
)

target_link_libraries(tick_hawkes_model
        ${TICK_LIB_BASE}
        ${TICK_LIB_ARRAY}
        ${TICK_LIB_LINEAR_MODEL}
        ${TICK_LIB_CRANDOM}
)