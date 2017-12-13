set(TICK_HAWKES_SIMULATION_INCLUDE_DIR ${PROJECT_SOURCE_DIR}/include/tick/hawkes/simulation)

add_library(tick_hawkes_simulation EXCLUDE_FROM_ALL
    ${TICK_HAWKES_SIMULATION_INCLUDE_DIR}/pp.h
    ${TICK_HAWKES_SIMULATION_INCLUDE_DIR}/poisson.h
    ${TICK_HAWKES_SIMULATION_INCLUDE_DIR}/hawkes.h
    ${TICK_HAWKES_SIMULATION_INCLUDE_DIR}/inhomogeneous_poisson.h
    ${TICK_HAWKES_SIMULATION_INCLUDE_DIR}/hawkes_kernels/hawkes_kernel.h
    ${TICK_HAWKES_SIMULATION_INCLUDE_DIR}/hawkes_kernels/hawkes_kernel_exp.h
    ${TICK_HAWKES_SIMULATION_INCLUDE_DIR}/hawkes_kernels/hawkes_kernel_sum_exp.h
    ${TICK_HAWKES_SIMULATION_INCLUDE_DIR}/hawkes_kernels/hawkes_kernel_power_law.h
    ${TICK_HAWKES_SIMULATION_INCLUDE_DIR}/hawkes_kernels/hawkes_kernel_0.h
    ${TICK_HAWKES_SIMULATION_INCLUDE_DIR}/hawkes_kernels/hawkes_kernel_time_func.h
    ${TICK_HAWKES_SIMULATION_INCLUDE_DIR}/hawkes_baselines/baseline.h
    ${TICK_HAWKES_SIMULATION_INCLUDE_DIR}/hawkes_baselines/constant_baseline.h
    ${TICK_HAWKES_SIMULATION_INCLUDE_DIR}/hawkes_baselines/timefunction_baseline.h
    hawkes.cpp
    poisson.cpp
    inhomogeneous_poisson.cpp
    hawkes_baselines/timefunction_baseline.cpp
    hawkes_kernels/hawkes_kernel_power_law.cpp
    hawkes_kernels/hawkes_kernel_exp.cpp
    hawkes_kernels/hawkes_kernel_time_func.cpp
    hawkes_kernels/hawkes_kernel_sum_exp.cpp
    hawkes_kernels/hawkes_kernel.cpp
    hawkes_baselines/constant_baseline.cpp
    pp.cpp
)

target_link_libraries(tick_hawkes_simulation
        ${TICK_LIB_BASE}
        ${TICK_LIB_ARRAY}
        ${TICK_LIB_HAWKES_MODEL}
        ${TICK_LIB_CRANDOM})