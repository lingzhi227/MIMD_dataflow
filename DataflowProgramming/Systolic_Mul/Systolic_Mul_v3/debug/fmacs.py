import os
import re
import struct

# Define a function to check if a string is a valid hexadecimal number
def is_hex(s):
    try:
        int(s, 16)
        return True
    except ValueError:
        return False

# Define a function to convert a hexadecimal string to float32
def hex_to_float32(hex_str):
    int_value = int(hex_str, 16)
    return struct.unpack('!f', int_value.to_bytes(4, byteorder='big'))[0]

# Create a debug folder if it does not exist
if not os.path.exists("debug"):
    os.makedirs("debug")

file_path = "fmacs.txt"
output_file_path = "debug/fmacs.txt"
output_lines = []

try:
    # Open the sim.log file to read content and process it
    with open(file_path, "r") as infile:
        for line in infile:
            if re.search(r"\bFMACS\b", line):
                # Extract fields like Src0, Src1, Src2, Dest containing hexadecimal numbers
                parts = line.split()
                for i, part in enumerate(parts):
                    if part.startswith("Dest:") or part.startswith("Src0:") or part.startswith("Src1:") or part.startswith("Src2:"):
                        hex_value = part.split(":")[1]
                        # Check if the value is a valid hexadecimal
                        if is_hex(hex_value):
                            # Convert the hexadecimal to float and replace
                            float_value = hex_to_float32(hex_value)
                            parts[i] = f"{part.split(':')[0]}:{float_value:.6f}"

                # Rejoin the processed line
                converted_line = " ".join(parts)
                output_lines.append(converted_line + "\n")

    # Write the converted content to debug/fmacs.txt file
    with open(output_file_path, "w") as outfile:
        outfile.writelines(output_lines)

    print(f"Successfully wrote the converted 'FMACS' data to {output_file_path}")

except FileNotFoundError:
    print(f"Error: File not found at path {file_path}.")