# Capacitive Lickometry System
### Created for the Timme Lab at University of Cincinnati
### Author: Christopher Parker

This repository contains the Python code for running our capacitive lickometry system.

## Installation

First, if you're on Windows (which is what we are using to make things easy on undergrads/grad students
who need to run the system), you'll have to install drivers for the FT232H boards. The recommended 
method is Zadig (https://zadig.akeo.ie/), and we roughly followed the steps outlined here: https://learn.adafruit.com/circuitpython-on-any-computer-with-ft232h/windows

Clone this repository (or download the .zip and extract), then run the following commands from inside the new directory.

The easiest installation option is to use miniforge: https://github.com/conda-forge/miniforge/releases

After installation on Windows, use the Miniforge Prompt start menu entry and navigate to the directory where you've cloned this repo. On Mac/Linux it should automatically add conda to your path, and you can use your normal terminal. Then enter the commands:
```
conda env create --file environment.yml
conda activate cliqr
jupyter-lab
```

Alternatively (on Unix-based systems), I manage my Python environments with pyenv-virtualenv on the command line: https://github.com/pyenv/pyenv-virtualenv

To install the environment, I run:
```
pyenv virtualenv 3.13 cliqr # creates a new environment called lickometry on Python 3.13
pyenv activate cliqr
pip install -r requirements.txt
```
Once the environment is configured and active, you'll need to set serial numbers for the FT232H boards
so we can tell which is which even if the USB plugs get shuffled (which would change the ftdi://... address,
and could cause confusion). I've included the script set_ft232h_serial.py for this purpose. The FT232H boards
should be plugged in ONE AT A TIME and the script can be run as follows:
```
python set_ft232h_serial.py FT232H0 # FT232H0 is the new serial number for the board
```
I have used the serial numbers FT232H0 through FT232H3 in our lab (and hence in the DataRecording.ipynb notebook),
so you can either use the same serials or modify the notebook. To change the serial numbers used, you'll need to
modify the serial_number_sensor_map dictionary to tell the system which cages go with which board.

Then to run the system, just use the command jupyter-lab and navigate to the DataRecording.ipynb notebook.

On Windows, a desktop shortcut can be created with the following link (to easily start the system for those not
comfortable with the command line):
```
C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe -c "cd ~/CLiQR; jupyter-lab"
```
The path "~/CLiQR" will need to match the directory where you cloned the repository.

## Tips
In the JupyterLab settings, it's probably a good idea to rebind any shortcuts that do not have multiple keystrokes.
For instance, by default, pressing 1 while having a cell selected (but not in the text editing mode), will change it
to be a markdown heading. The same is true for other digits up to 6, I think. Also, pressing 'm' changes it to plain markdown.
While it has only happened once, we did have to restart the GUI once because of these shortcuts, and that messed up the last
section of data. I just went through and rebound them to require CTRL + 'key'.
