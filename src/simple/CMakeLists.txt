function(add_simple name)
    add_executable(${name}
        src/${name}.cpp
    )
    target_link_libraries(${name} ${OpenCV_LIBS})
endfunction()

add_simple(edge_detection)
add_simple(effects)
add_simple(stereo)
add_simple(features)
add_simple(calibration)
