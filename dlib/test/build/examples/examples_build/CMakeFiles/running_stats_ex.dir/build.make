# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 2.8

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list

# Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/chin/Git/DLIB/dlib/test

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/chin/Git/DLIB/dlib/test/build

# Include any dependencies generated for this target.
include examples/examples_build/CMakeFiles/running_stats_ex.dir/depend.make

# Include the progress variables for this target.
include examples/examples_build/CMakeFiles/running_stats_ex.dir/progress.make

# Include the compile flags for this target's objects.
include examples/examples_build/CMakeFiles/running_stats_ex.dir/flags.make

examples/examples_build/CMakeFiles/running_stats_ex.dir/running_stats_ex.cpp.o: examples/examples_build/CMakeFiles/running_stats_ex.dir/flags.make
examples/examples_build/CMakeFiles/running_stats_ex.dir/running_stats_ex.cpp.o: /home/chin/Git/DLIB/examples/running_stats_ex.cpp
	$(CMAKE_COMMAND) -E cmake_progress_report /home/chin/Git/DLIB/dlib/test/build/CMakeFiles $(CMAKE_PROGRESS_1)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object examples/examples_build/CMakeFiles/running_stats_ex.dir/running_stats_ex.cpp.o"
	cd /home/chin/Git/DLIB/dlib/test/build/examples/examples_build && /usr/bin/c++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/running_stats_ex.dir/running_stats_ex.cpp.o -c /home/chin/Git/DLIB/examples/running_stats_ex.cpp

examples/examples_build/CMakeFiles/running_stats_ex.dir/running_stats_ex.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/running_stats_ex.dir/running_stats_ex.cpp.i"
	cd /home/chin/Git/DLIB/dlib/test/build/examples/examples_build && /usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -E /home/chin/Git/DLIB/examples/running_stats_ex.cpp > CMakeFiles/running_stats_ex.dir/running_stats_ex.cpp.i

examples/examples_build/CMakeFiles/running_stats_ex.dir/running_stats_ex.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/running_stats_ex.dir/running_stats_ex.cpp.s"
	cd /home/chin/Git/DLIB/dlib/test/build/examples/examples_build && /usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -S /home/chin/Git/DLIB/examples/running_stats_ex.cpp -o CMakeFiles/running_stats_ex.dir/running_stats_ex.cpp.s

examples/examples_build/CMakeFiles/running_stats_ex.dir/running_stats_ex.cpp.o.requires:
.PHONY : examples/examples_build/CMakeFiles/running_stats_ex.dir/running_stats_ex.cpp.o.requires

examples/examples_build/CMakeFiles/running_stats_ex.dir/running_stats_ex.cpp.o.provides: examples/examples_build/CMakeFiles/running_stats_ex.dir/running_stats_ex.cpp.o.requires
	$(MAKE) -f examples/examples_build/CMakeFiles/running_stats_ex.dir/build.make examples/examples_build/CMakeFiles/running_stats_ex.dir/running_stats_ex.cpp.o.provides.build
.PHONY : examples/examples_build/CMakeFiles/running_stats_ex.dir/running_stats_ex.cpp.o.provides

examples/examples_build/CMakeFiles/running_stats_ex.dir/running_stats_ex.cpp.o.provides.build: examples/examples_build/CMakeFiles/running_stats_ex.dir/running_stats_ex.cpp.o

# Object files for target running_stats_ex
running_stats_ex_OBJECTS = \
"CMakeFiles/running_stats_ex.dir/running_stats_ex.cpp.o"

# External object files for target running_stats_ex
running_stats_ex_EXTERNAL_OBJECTS =

examples/examples_build/running_stats_ex: examples/examples_build/CMakeFiles/running_stats_ex.dir/running_stats_ex.cpp.o
examples/examples_build/running_stats_ex: examples/examples_build/CMakeFiles/running_stats_ex.dir/build.make
examples/examples_build/running_stats_ex: dlib_build/libdlib.a
examples/examples_build/running_stats_ex: /usr/lib/x86_64-linux-gnu/libnsl.so
examples/examples_build/running_stats_ex: /usr/lib/x86_64-linux-gnu/libSM.so
examples/examples_build/running_stats_ex: /usr/lib/x86_64-linux-gnu/libICE.so
examples/examples_build/running_stats_ex: /usr/lib/x86_64-linux-gnu/libX11.so
examples/examples_build/running_stats_ex: /usr/lib/x86_64-linux-gnu/libXext.so
examples/examples_build/running_stats_ex: /usr/lib/x86_64-linux-gnu/libpng.so
examples/examples_build/running_stats_ex: examples/examples_build/CMakeFiles/running_stats_ex.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --red --bold "Linking CXX executable running_stats_ex"
	cd /home/chin/Git/DLIB/dlib/test/build/examples/examples_build && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/running_stats_ex.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
examples/examples_build/CMakeFiles/running_stats_ex.dir/build: examples/examples_build/running_stats_ex
.PHONY : examples/examples_build/CMakeFiles/running_stats_ex.dir/build

examples/examples_build/CMakeFiles/running_stats_ex.dir/requires: examples/examples_build/CMakeFiles/running_stats_ex.dir/running_stats_ex.cpp.o.requires
.PHONY : examples/examples_build/CMakeFiles/running_stats_ex.dir/requires

examples/examples_build/CMakeFiles/running_stats_ex.dir/clean:
	cd /home/chin/Git/DLIB/dlib/test/build/examples/examples_build && $(CMAKE_COMMAND) -P CMakeFiles/running_stats_ex.dir/cmake_clean.cmake
.PHONY : examples/examples_build/CMakeFiles/running_stats_ex.dir/clean

examples/examples_build/CMakeFiles/running_stats_ex.dir/depend:
	cd /home/chin/Git/DLIB/dlib/test/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/chin/Git/DLIB/dlib/test /home/chin/Git/DLIB/examples /home/chin/Git/DLIB/dlib/test/build /home/chin/Git/DLIB/dlib/test/build/examples/examples_build /home/chin/Git/DLIB/dlib/test/build/examples/examples_build/CMakeFiles/running_stats_ex.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : examples/examples_build/CMakeFiles/running_stats_ex.dir/depend

