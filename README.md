# NITHwebcamWrapper
_A python script to turn a normal webcam into a sensor capable to detect some movement features and output them through NITH protocol_

<p align="center">
  <img src="https://raw.githubusercontent.com/LIMUNIMI/NITHwebcamWrapper/main/Readme_images/NITHwebcamWrapper.png" width="50%" />
</p>

## Overview

NITHwebcamWrapper is part of NITH, a framework for building software tools for accessibility for people with quadriplegic or other kind of motor disabilities which hinder hand movements.
It's Python script which extracts facial movement features in real-time by analizing the video stream of a webcam. It's part of the [NithSensors](https://github.com/LIMUNIMI/NITHsensors) collection, and it's built to be used for interaction purposes.

The currently extracted facial movement features are:
- Head rotation (yaw and pitch)
- Opening of the left and right eyes
- Mouth opening

Detected features are relayed locally over IP "127.0.0.1" on UDP port 20100, using the standard NithSensors communication protocol (see NithSensors [Readme](https://github.com/LIMUNIMI/NITHsensors)) This allows applications designed to work with NithSensors to receive and incorporate these detections.

Landmarks extraction is provided by the [MediaPipe](https://google.github.io/mediapipe/) library.

## Installation Guide

### Step 1: Installing Python

 is built with Python, meaning you'll need Python installed on your computer to run it. If you're new to Python, follow this steps:

1. Visit the [official Python website](https://www.python.org/downloads/).
2. Download the latest version of Python for your operating system (Windows, macOS, Linux).
3. Run the downloaded installer and follow the on-screen instructions. Make sure to check the box that says "Add Python X.X to PATH" during installation to make Python accessible from your command line or terminal.

For a more detailed guide, you can check out this beginner-friendly [Python installation tutorial](https://realpython.com/installing-python/).

### Step 2: Download NITHwebcamWrapper

You can download NITHwebcamWrapper by cloning the repository from GitHub:

```bash
git clone https://github.com/LIMUNIMI/NITHwebcamWrapper.git
```

### Step 3: Installing Dependencies

With Python installed, open your command line or terminal and install the required dependencies by running:

```bash
pip install opencv-python numpy mediapipe python-osc
```

These dependencies include packages for computer vision operations, mathematics, data handling, extracting facial landmarks, and sending data over UDP.

## Running NITHwebcamWrapper
To run NITHwebcamWrapper, ensure you're in the NITHwebcamWrapper directory in your command line or terminal. Then execute:

```bash
python main.py
```
If you're on Windows, you can alternatively simply run `NITHwebcamWrapper.bat`.

Make sure your webcam is connected and properly setup as NITHwebcamWrapper will immediately begin to capture and analyze facial movements.

## Contributions and Support
NITHwebcamWrapper is licensed through a GNU GPL-v3 Free Open-Source software license. Feel free to contribute!

You can open an Issue for any request regarding this code, or contact me (the developer) directly via email at *nicola.davanzo@unimi.it*.

## Acknowledgements
Thanks to one of my thesis students ([@Ale-Gr](https://github.com/Ale-Gr/)), which contributed to this project with his thesis work. 🙏 Check out [his code](https://github.com/LIMUNIMI/TT-Alessandro_Grasso-2024-975295), which is a similar application which sends a similar set of features formatted using the OSC (Open Sound Control) protocol, provides a graphical user interface, a calibration system, and can be used to emulate the mouse cursor (for accessibility purposes).

The head rotation analysis function is based on a Python script provided by [Nicolai Nielsen in a tutorial on YouTube](https://youtu.be/-toNMaS4SeQ?si=gcpJSJcbziH-V4DN). Thank you 🙏
