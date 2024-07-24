# Python_Ultrasound_Reconstruction
A Python library for 3D free-hand ultra-sound measurements, volumetric reconstruction, and segmentation

## Installation
This package uses ```python3.8```. Other versions might result in issues. Tested on Windows 10.

To get started, clone the repository (as well as any submodules):
```
git clone https://github.com/Neuro-Mechatronics-Interfaces/Python_Ultrasound_Reconstruction.git
```

Navigate to the package directory, create a new virtual python environment (installing packages locally in a virtual environment instead of the PC is optional but highly recommended):
```
cd Python_Ultrasound_Recunstruction
python -m venv py_us
```
#### Note that you should always activate the virtual environment before attempting to run any files in the package.

Activate the environment. How you activate your virtual environment depends on the OS you’re using.

 - To activate your venv on Windows, you need to run a script that gets installed by venv. If you created your venv in a directory called myenv, the command would be:
   ```
   # In cmd.exe
   py_us\Scripts\activate.bat
   # In PowerShell
   py_us\Scripts\Activate.ps1
   ```
 - On Linux and MacOS, we activate our virtual environment with the source command. If you created your venv in the myvenv directory, the command would be:
   ```
   $ source myvenv/bin/activate
   ```
 - When you're done, to deactivate the virtual environment, you can run the `deactivate` command:
   ```
   deactivate
   ```
Install the python packages required for the project in the environment:
```
pip install -r requirements.txt
```
  - If it fails, try to upgrade pip:
    ```
    python -m pip install --upgrade pip
    ```
