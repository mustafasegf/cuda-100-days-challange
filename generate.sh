#!/usr/bin/env bash

# Read and increment day
last_day=$(<.lastday)
day=$((last_day + 1))
padded=$(printf "%03d" "$day")

# Ask for program name if not given
if [ -z "$1" ]; then
  # read -p "Enter program name (without .cu): " prog
  prog="main"
else
  prog="$1"
fi

folder="day-${padded}"
mkdir -p "$folder"

# Create Makefile
cat > "$folder/Makefile" <<EOF
NVCC = nvcc
NVCCFLAGS = -std=c++11 -O3
TARGET = ${prog}
SOURCE = ${prog}.cu

all: \$(TARGET)

\$(TARGET): \$(SOURCE)
	\$(NVCC) \$(NVCCFLAGS) -o \$@ \$<

clean:
	rm -f \$(TARGET)

run:
	./\$(TARGET)

.PHONY: all clean run
EOF

# Create README.md
cat > "$folder/README.md" <<EOF
# Day ${day}

File: [${prog}.cu](https://github.com/mustafasegf/cuda-100-days-challange/blob/master/${folder}/${prog}.cu)
EOF


# Create .gitignore
cat > "$folder/.gitignore" <<EOF
${prog}
EOF

# Create empty .cu file
touch "$folder/${prog}.cu"

# Update .lastday
echo "$day" > .lastday

echo "Created $folder with $prog.cu, Makefile, .gitignore and README.md"

