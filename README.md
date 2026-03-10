# QuantumToy

QuantumToy is a modular Python simulation environment for exploring quantum dynamics and alternative quantum interpretations.

The project implements several wave equation models and allows experimentation with different theoretical extensions such as:

* Schrodinger dynamics  
* Continuous measurement models  
* Dirac equation  
* Thick reality front models  
* Hybrid Dirac plus Thick Front theories  

The simulator includes tools for:

* wavefunction propagation  
* double slit experiments  
* Bohmian trajectory overlays  
* ridge tracking of probability flow  
* retrodictive weighting (Emix)  
* animated visualization  

The goal is to provide a playground for quantum dynamics experiments and theoretical exploration.

---

## Features

The simulator supports:

### Wave dynamics

* Schrodinger equation using FFT split operator  
* Dirac equation using a two spinor relativistic model  

### Experimental models

* double slit barriers  
* absorbing boundaries  
* detection screens  

### Analysis tools

* ridge tracking of probability flow  
* velocity field visualization  
* Bohmian trajectory integration  
* divergence diagnostics  
* retrodictive weighting (Emix)  

### Visualization

* animated density maps  
* ridge trajectory overlay  
* Bohmian paths  
* flow arrows  
* mp4 video export  

---

### Legacy experiments

The repository also contains an experimental/ directory.

This folder includes earlier standalone research scripts that were used during the development of the project.
Some of these scripts reproduce the originally demonstrated animations and visual experiments.

These legacy experiments often use a slightly different workflow compared to the newer modular theory classes:

* forward wave evolution is computed first

* a retrodictive weighting field (Emix) is constructed from detector-conditioned backward propagation

* the visible ridge structure is then derived from the overlap of forward density and the backward effect field

The newer theory modules in src/quantumtoy/theories attempt to internalize similar ideas directly into the dynamical models (for example Thick Front and worldline-style extensions).

The experimental scripts are kept in the repository because they reproduce the original research visualizations and may still be useful for comparison or further exploration.

---

## Example simulation

Typical experiment simulated by the code:
wave packet -> double slit -> interference -> detection screen

The simulation computes:

* forward wave evolution  
* backward click conditioned propagation  
* ridge trajectory through the probability landscape  
* optional Bohmian trajectories  

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

* split operator FFT methods  
* Dirac spinor propagation  
* Bohmian trajectory integration  
* retrodictive weighting methods