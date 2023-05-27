import os

eng_to_gr = {
    "a": "α",
    "b": "β",
    "c": "γ",
    "d": "δ",
    "e": "ε",
    "z": "ζ",
    "h": "η",
    "q": "θ",
    "i": "ι",
    "k": "κ",
    "l": "λ",
    "m": "μ",
    "n": "ν",
    "j": "ξ",
    "o": "ο",
    "p": "π",
    "r": "ρ",
    "s": "σ",
    "t": "τ",
    "u": "υ",
    "f": "φ",
    "x": "χ",
    "y": "ψ",
    "w": "ω"
}



files_to_move = []
# list all files in /home/Public and save in list
for root, dirs, files in os.walk('/home/Public'):
    for f in files:
        files_to_move.append(os.path.join(root, f))

for f in files_to_move:
    # print(f)
    filename = f.split("/")[-1]
    label = eng_to_gr[filename[0]]
    if "Accelerometer" in f:
        destination = "~/Documents/GitHub/Handwriting-Sensor-Recognition/data/" + label + "/accel"
    else:
        destination = "~/Documents/GitHub/Handwriting-Sensor-Recognition/data/" + label + "/gyro"
    new_filename = destination + "/" + filename
    os.system("mv " + f + " " + destination)