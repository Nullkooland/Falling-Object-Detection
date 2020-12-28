# Copyright - Benjamin Laugraud <blaugraud@ulg.ac.be> - 2016
# Copyright - Marc Van Droogenbroeck <m.vandroogenbroeck@ulg.ac.be> - 2016
#
# ViBe is covered by a patent (see http://www.telecom.ulg.ac.be/research/vibe).
#
# Permission to use ViBe without payment of fee is granted for nonprofit
# educational and research purposes only.
#
# This work may not be copied or reproduced in whole or in part for any purpose.
#
# Copying, reproduction, or republishing for any purpose shall require a
# license. Please contact the authors in such cases. All the code is provided
# without any guarantee.

set(TARGET_NAME vibe++)

file(
  GLOB_RECURSE
  ${TARGET_NAME}_SRCS
  ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp
)

add_library(${TARGET_NAME} STATIC
  ${${TARGET_NAME}_SRCS}
)

target_include_directories(${TARGET_NAME} PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/common
    ${CMAKE_CURRENT_SOURCE_DIR}/metaprograms
)