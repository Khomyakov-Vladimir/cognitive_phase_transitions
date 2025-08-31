# Cognitive Phase Transitions in Subjective Physics: Modeling Synchronization and Order Parameters with Reproducible Simulations

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXXXX.svg)](https://doi.org/10.5281/zenodo.XXXXXXXX)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

This repository is the official companion code for the paper:

> **Vladimir Khomyakov (2025).** _Cognitive Phase Transitions in Subjective Physics: Modeling Synchronization and Order Parameters with Reproducible Simulations._ Zenodo. https://doi.org/10.5281/zenodo.XXXXXXXX

It provides a complete implementation of the cognitive phase transition framework described in the paper, enabling reproducible simulations and analysis of synchronization dynamics in adaptive networks.

## Abstract

This work investigates cognitive phase transitions within the framework of Subjective Physics, providing direct computational evidence of a critical transition in an adaptive network of N=40 cognitive agents. We explicitly distinguish between the global order parameter |⟨ψ⟩| (Kuramoto mean-field amplitude) and the Mean Pairwise Coherence (MPC), a measure of local synchronization. 

A clear non-monotonic transition was detected at control parameter r_c = 1.534, characterized by a sharp change (|ΔMPC| = 0.133) indicating a structural reorganization of the system's synchronized clusters. Following a transient period of desynchronization (MPC minimum ∼0.42), the system stabilizes into a high-coherence phase (post-transition MPC = 0.992 ± 0.007). The results demonstrate a phenomenology analogous to second-order phase transitions in physical systems, confirming key hypotheses of Subjective Physics regarding the reorganization of observer states.

## Repository Structure

```
cognitive_phase_transitions/
│
├── scripts/
│ └── cognitive_phase_transitions.py   # Main simulation script
├── figures/                           # Directory for saving output figures
│ ├── simulation_results.zip
│ ├── simulation_results.z01
│ ├── simulation_results.z02
│ ├── final_weight_matrix.pdf
│ ├── mpc_change.pdf
│ ├── network_phase_coloring.pdf
│ ├── order_parameter_and_control.pdf
│ ├── phase_transition_curve.pdf
│ └── synchronization_measure.pdf
├── requirements.txt                   # Python dependencies
├── .zenodo.json
├── CITATION.cff
├── LICENSE                            # MIT License
└── README.md                          # This file
```

### The 'figures/' directory will be created automatically by the script to save output plots.

## File Descriptions

Below is a complete description of all scripts included in this version. These files form a **self-contained and reproducible package** supporting the simulations, figures, and data analyses presented in the article.

**cognitive_phase_transitions.py** — Core simulation framework:
- Implements adaptive network of phase oscillators
- Combines Kuramoto-like synchronization with Hebbian plasticity
- Includes online covariance estimation for free-energy minimization
- Features transition detection and consolidation mechanisms

## Installation

To run the cognitive phase transition simulations, you need Python ≥ 3.9. The required libraries can be installed via pip.

**Create a virtual environment (recommended):**

```bash
python -m venv cpt-env
# On macOS/Linux:
source cpt-env/bin/activate
# On Windows (PowerShell):
cpt-env\Scripts\activate
# On Windows (CMD):
cpt-env\Scripts\activate.bat
```

**Install dependencies:**

```bash
pip install -r requirements.txt
```

Alternatively, you can install them manually:

```bash
pip install numpy networkx matplotlib scipy
```

## Usage

**Running the simulation**:

```bash
# Run the comprehensive simulation from the repository root directory
python scripts/cognitive_phase_transitions.py
```

This will generate output showing:

- Phase transition detection at critical parameter values
- Evolution of order parameters and synchronization metrics
- Final network state visualization
- Statistical analysis of transition characteristics

## Output Figures

Running the script `cognitive_phase_transitions.py` generates the following figures, which are central to the analysis in the accompanying article:

- **`final_weight_matrix.pdf`**  
  Visualizes the final adaptive weight matrix \( W_{ij} \). The pattern reflects the history of the system's dynamics, with stronger connections consolidating the globally synchronized state achieved post-transition.  
  Corresponds to: *Figure 8. Final Weight Matrix* in the article.

- **`mpc_change.pdf`**  
  Plots the temporal change in Mean Pairwise Coherence (\( \Delta\text{MPC} \)). The phase transition is triggered when \( |\Delta\text{MPC}| > 0.1 \). The largest change (\( |\Delta\text{MPC}|=0.133 \)) occurs at \( t=413.70 \).  
  Corresponds to: *Figure 4. Change in Synchronization* in the article.

- **`network_phase_coloring.pdf`**  
  Displays the network structure with node coloring representing phase angles (final state, \( t=1000 \)). Demonstrates the high degree of global synchronization achieved after the transition and subsequent stabilization.  
  Corresponds to: *Figure 6. Network Structure and Final State* in the article.

- **`order_parameter_and_control.pdf`**  
  Shows the dynamics of the global order parameter \( |\langle \psi \rangle| \) and the control parameter \( r \). \( |\langle \psi \rangle| \) remains negligible until the system exits the multi-cluster state after the transition, then increases steadily as global coherence is established.  
  Corresponds to: *Figure 2. Dynamics of Global Order and Control* in the article.

- **`phase_transition_curve.pdf`**  
  Illustrates the phase transition curve: Mean Pairwise Coherence (MPC) as a function of the control parameter \( r \) (smoothed with a moving average). Highlights the critical region near \( r \approx 1.5 \), the subsequent valley, and the final stabilization at high MPC for \( r > 2.0 \).  
  Corresponds to: *Figure 3. Phase Transition Curve* in the article.

- **`synchronization_measure.pdf`**  
  Charts the evolution of Mean Pairwise Coherence (MPC). The phase transition is detected at \( t=413.70 \). Shows the non-monotonic trajectory featuring a desynchronization valley before final stabilization.  
  Corresponds to: *Figure 1. Evolution of Mean Pairwise Coherence* in the article.

## Data Archive

For complete scientific reproducibility, the raw numerical data used to generate all figures is preserved in a compressed split archive:

- **`simulation_results.zip`** (17.4 MB) - Main archive part
- **`simulation_results.z01`** (21.0 MB) - Archive part 1
- **`simulation_results.z02`** (21.0 MB) - Archive part 2

### Archive Contents

The archive contains the file `simulation_results.npz` (62.5 MB uncompressed) which includes:
- `psi_history` - Complete history of complex field values ψ for all N=40 agents across all timesteps
- `order_param_history` - Evolution of the global order parameter |⟨ψ⟩|
- `r_history` - Values of the control parameter r at each timestep
- `W` - Final weight matrix after simulation completion

### Extraction Instructions

To reconstruct the original `simulation_results.npz` file:
1. Ensure all three archive parts are in the same directory
2. Use any archive manager that supports split ZIP files (e.g., 7-Zip, WinRAR)
3. Extract using the main file: `simulation_results.zip`
4. The complete `simulation_results.npz` will be restored to its original 62.5 MB size

> Note: This split archive format was necessary due to GitHub's 25 MB file size limit. The NPZ file contains the complete simulation data needed to verify all results and regenerate any figure independently.

## Running the simulation

```bash
# Run the comprehensive simulation
python scripts/cognitive_phase_transitions.py
```

This will generate output showing:

- Phase transition detection at critical parameter values
- Evolution of order parameters and synchronization metrics
- Final network state visualization
- Statistical analysis of transition characteristics

## Expected Results & Outputs

Running the code will demonstrate:

**Cognitive Phase Transition**: The system undergoes a non-monotonic transition at r_c = 1.534, characterized by cluster reorganization.

**Order Parameter Dynamics**: Distinct behavior of global order parameter |⟨ψ⟩| and local synchronization measure (MPC).

**Desynchronization Valley**: Transient decrease in coherence (MPC minimum ∼0.42) during structural reconfiguration.

**Stabilization**: Final convergence to high-coherence phase (MPC = 0.992 ± 0.007).

**Network Visualization**: Final network state with phase coloring and weight matrix patterns.

## Citation

If you use this model or code in your research, please cite the original publication:

```bibtex
@misc{khomyakov_vladimir_2025_XXXXXXXX,
  author       = {Khomyakov, Vladimir},
  title        = {Cognitive Phase Transitions in Subjective Physics: Modeling Synchronization and Order Parameters with Reproducible Simulations},
  month        = aug,
  year         = 2025,
  publisher    = {Zenodo},
  version      = {1.0},
  doi          = {10.5281/zenodo.XXXXXXXX},
  url          = {https://doi.org/10.5281/zenodo.XXXXXXXX}
}
```

- **Version-specific DOI:** [10.5281/zenodo.XXXXXXXX](https://doi.org/10.5281/zenodo.XXXXXXXX)  
- **Concept DOI (latest version):** [10.5281/zenodo.XXXXXXXX](https://doi.org/10.5281/zenodo.XXXXXXXX)  
- **Download PDF:** [Direct link to paper on Zenodo](https://zenodo.org/records/XXXXXXXX/files/cognitive_phase_transitions.pdf?download=1)

## References

**Theoretical Foundation**: This work is based on the principles of Subjective Physics and builds upon the "Minimal Model of Cognitive Projection" (DOI: [10.5281/zenodo.16888675](https://doi.org/10.5281/zenodo.16888675)).

**Methodological Inspiration**: The framework combines elements from Ginzburg-Landau theory, Kuramoto synchronization models, and Hebbian/STDP-like synaptic plasticity.

## Keywords

subjective physics, cognitive phase transitions, synchronization, order parameters, kuramoto model, adaptive networks, reproducible simulations, observer entropy, collective intelligence, information theory

## License

This project is licensed under the MIT License. See the LICENSE file for details.

## Acknowledgements

This work builds upon the hypothesis of **Subjective Physics** formulated by Alexander Kaminsky. Special thanks to the researchers whose work on synchronization dynamics and phase transitions has inspired this computational framework.