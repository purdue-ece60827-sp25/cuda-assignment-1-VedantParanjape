import subprocess
import os
import csv

SRC_ROOT = "/home/paranjav/ece60827/cuda-assignment-1-VedantParanjape"
BUILD_ROOT = "/home/paranjav/ece60827/cuda-assignment-1-VedantParanjape/build"
OUT_FILE1 = "run.part1.out"
OUT_FILE2 = "run.part2.out"

# Change to the build directory
os.chdir(BUILD_ROOT)

# Create or clear the output file
with open(OUT_FILE1, "w") as out_file1:
    out_file1.write("")

# Create or clear the output file
with open(OUT_FILE2, "w") as out_file2:
    out_file2.write("")

# Loop over the range from 15 to 25
for i in range(15, 21):

    # Edit the .cuh file
    subprocess.call(f"sed -i 's/#define VECTOR_SIZE (1 << [0-9]*)/#define VECTOR_SIZE (1 << {i})/' {SRC_ROOT}/lab1.cuh", shell=True)

    # Clean the previous build and run the build commands
    subprocess.run(["make", "clean"])

    # Compile with parallel jobs
    subprocess.run(["make", "-j"])

    # Run nvprof and process the output
    nvprof_command = [
        "nvprof", "--print-gpu-trace", "--log-file", "output.part1.log", "--csv", "./lab1"
    ]
    subprocess.run(nvprof_command, input="2", text=True)

    # Extract the GPU trace and compute the total time
    with open("output.part1.log", "r") as log_file:
        log_data = list(csv.reader(log_file))
        total = 0
        time = 0
        if log_data[4][1] == "ms":
            time = 2
        elif log_data[4][1] == "us":
            time = 1

        for row in log_data[5:]:
            print(row[1])
            total += float(row[1])

        if time == 1:
            print(f"{total} us")
        elif time == 2:
            total = total * 1000
            print(f"{total} us")

    # Append the total time to the output file
    with open(OUT_FILE1, "a") as out_file:
        out_file.write(f"{1 << int(i)}, {total}\n")

# Loop over the range from 15 to 25
for i in range(10, 16):

    # Edit the .cuh file
    subprocess.call(f"sed -i 's/#define GENERATE_BLOCKS\s*[0-9]*/#define GENERATE_BLOCKS\t\t{1 << i}/' {SRC_ROOT}/lab1.cuh", shell=True)

    # Clean the previous build and run the build commands
    subprocess.run(["make", "clean"])

    # Compile with parallel jobs
    subprocess.run(["make", "-j"])

    # Run nvprof and process the output
    nvprof_command = [
        "nvprof", "--print-gpu-trace", "--log-file", "output.part2.log", "--csv", "./lab1"
    ]
    subprocess.run(nvprof_command, input="4", text=True)

    # Extract the GPU trace and compute the total time
    with open("output.part2.log", "r") as log_file:
        log_data = list(csv.reader(log_file))
        total = 0
        time = 0
        if log_data[4][1] == "ms":
            time = 2
        elif log_data[4][1] == "us":
            time = 1

        for row in log_data[5:]:
            print(row[1])
            total += float(row[1])

        if time == 1:
            print(f"{total} us")
        elif time == 2:
            total = total * 1000
            print(f"{total} us")

    # Append the total time to the output file
    with open(OUT_FILE2, "a") as out_file:
        out_file.write(f"{1 << int(i)}, {total}\n")
