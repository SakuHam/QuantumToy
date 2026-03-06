# QuantumToy

QuantumToy is a modular Python simulation environment for exploring quantum dynamics and alternative quantum interpretations.

The project implements several wave equation models and allows experimentation with different theoretical extensions such as:

o Schrodinger dynamics  
o Continuous measurement models  
o Dirac equation  
o Thick reality front models  
o Hybrid Dirac plus Thick Front theories  

The simulator includes tools for:

o wavefunction propagation  
o double slit experiments  
o Bohmian trajectory overlays  
o ridge tracking of probability flow  
o retrodictive weighting (Emix)  
o animated visualization  

The goal is to provide a playground for quantum dynamics experiments and theoretical exploration.

---

## Features

The simulator supports:

### Wave dynamics

o Schrodinger equation using FFT split operator  
o Dirac equation using a two spinor relativistic model  

### Experimental models

o double slit barriers  
o absorbing boundaries  
o detection screens  

### Analysis tools

o ridge tracking of probability flow  
o velocity field visualization  
o Bohmian trajectory integration  
o divergence diagnostics  
o retrodictive weighting (Emix)  

### Visualization

o animated density maps  
o ridge trajectory overlay  
o Bohmian paths  
o flow arrows  
o mp4 video export  

---

## Example simulation

Typical experiment simulated by the code:
wave packet -> double slit -> interference -> detection screen

The simulation computes:

o forward wave evolution  
o backward click conditioned propagation  
o ridge trajectory through the probability landscape  
o optional Bohmian trajectories  

---

## Installation

Clone the repository:
git clone https://github.com/YOURNAME/quantumtoy.git
cd quantumtoy

Create a virtual environment:
python -m venv venv
source venv/bin/activate

On Windows:
venv\Scripts\activate

Install dependencies:
pip install -r requirements.txt

---

## Requirements

Main Python dependencies:
numpy
matplotlib
scipy

Video export requires ffmpeg.

Linux:
sudo apt install ffmpeg

Mac:
brew install ffmpeg

---

## Running a simulation

Run the main simulation:
python3 main.py

The simulation will generate a video file such as:
schrodinger_modular_api.mp4
and open a visualization window.

---

## Configuration

Simulation parameters are controlled through:
config.py

Example configuration:
THEORY_NAME = "schrodinger_measurement"

---

| Theory                 | Equation                 | State       | Collapse   | Relativistic |
| ---------------------- | ------------------------ | ----------- | ---------- | ------------ |
| Schrödinger            | (i\hbar \partial_t \psi) | scalar      | optional   | no           |
| SchrödingerMeasurement | stochastic               | scalar      | continuous | no           |
| Dirac                  | relativistic spinor      | 2-component | no         | yes          |
| ThickFront             | modified wave evolution  | scalar      | emergent   | no           |
| DiracThickFront        | hybrid                   | spinor      | emergent   | yes          |

---

## Author

Project by:

Saku Hamalainen

---

## Acknowledgements

The simulator builds on standard numerical methods used in quantum simulation, including:

o split operator FFT methods  
o Dirac spinor propagation  
o Bohmian trajectory integration  
o retrodictive weighting methods