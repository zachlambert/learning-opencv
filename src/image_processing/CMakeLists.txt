add_executable(image_processing
    src/main.cpp
    src/trackbar_data.cpp
)
target_include_directories(image_processing
    PRIVATE include
)
target_link_libraries(image_processing
    ${OpenCV_LIBS}
)
