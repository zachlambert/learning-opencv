add_executable(features
    src/main.cpp
)
target_link_libraries(features
    ${OpenCV_LIBS}
)
