add_executable(edge_detection
    src/main.cpp
)
target_link_libraries(edge_detection
    ${OpenCV_LIBS}
)
