import os
import subprocess

class RunScript:
    def __init__(self, args, filename="main.cpp"):
        self.wkDir = os.getcwd()
        self.filename = os.path.join(self.wkDir, filename)
        
        # Path to ggml (CHANGE if needed)
        self.ggml_path = "/mnt/c/Users/shiva/Desktop/IISC/temp/ExampleTemp/GGML/ggml"   

        # Includes
        includes = [
            f"-I{self.ggml_path}/include"
        ]

        # Library paths
        # Library paths
        lib_dirs = [
            f"-L{self.ggml_path}/build/src",
            f"-L{self.ggml_path}/build/src/ggml-cuda", # <-- Adds the specific CUDA build folder
            "-L/usr/local/cuda/lib64",                 # <-- Tells it where the NVIDIA math libraries live
            f"-Wl,-rpath,{self.ggml_path}/build/src",
            f"-Wl,-rpath,{self.ggml_path}/build/src/ggml-cuda" # <-- Allows it to run after compiling
        ]

        # Libraries
        libs = [
            "-lggml",
            "-lggml-base",
            "-lggml-cpu",
            "-lggml-cuda",
            "-lcudart",
            "-lcublas"
        ]

        # Final command
        self.args = (
            ["g++", "-std=c++17", self.filename]
            + includes
            + self.GetAllDir()
            + lib_dirs
            + ["-o", "main"]
            + libs
        )

    def GetAllDir(self):
        dirs = []
        includeDir = [self.wkDir]
        while includeDir:
            folder = includeDir.pop()
            for roots, dire, files in os.walk(folder):
                dirs.append(roots)
        return ["-I" + i for i in dirs]

    def PrintStatus(self, result):
        if result.returncode == 0:
            print("[*] Compilation successful!")
            subprocess.run(["./main"])
        else:
            print("[#] Compilation failed")
            print(result.stderr.decode())
            print("Command used:", " ".join(self.args))

    def Run(self):
        print("Compiling...")
        result = subprocess.run(self.args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        self.PrintStatus(result)

if __name__ == "__main__":
    script = RunScript([])
    script.Run()