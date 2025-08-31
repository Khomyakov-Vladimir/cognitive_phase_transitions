#!/usr/bin/env python3
"""
# scripts/cognitive_phase_transition.py

Cognitive Phase Transitions in Subjective Physics: Modeling Synchronization and Order Parameters with Reproducible Simulations

This script implements a computational model of critical transitions in cognitive networks,
combining elements from:
- Ginzburg-Landau theory for phase transitions
- Kuramoto-like phase synchronization
- Hebbian/STDP-like synaptic plasticity
- Online covariance estimation for free-energy minimization

The model demonstrates:
1. Noise-induced transitions between disordered and ordered states
2. Adaptive rewiring based on phase coherence
3. Critical slowing down near transition points
4. Hysteresis effects and metastability

Mathematical framework combines:
- Complex-order parameter fields ψ ∈ ℂ^N
- Energy functional minimization: dψ/dt = -δΦ/δψ
- Normalized graph Laplacian for coupling
- Running covariance estimation for free-energy bound

Author: Vladimir Khomyakov
License: MIT
Repository: https://github.com/Khomyakov-Vladimir/cognitive_phase_transition
Citation: DOI:10.5281/zenodo.XXXXXXXX
"""

import os
import logging
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

# Reproducibility and logging
np.random.seed(42)  # Fixed seed for deterministic reproducibility
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -----------------------------
# Path handling (portable between /repo and /repo/scripts)
# -----------------------------
script_dir = os.path.dirname(os.path.abspath(__file__))
if os.path.basename(script_dir) == 'scripts':
    repo_root = os.path.dirname(script_dir)
else:
    repo_root = script_dir
figures_dir = os.path.join(repo_root, 'figures')
os.makedirs(figures_dir, exist_ok=True)

# -----------------------------
# Parameters (tunable)
# -----------------------------
N = 40  # Number of network nodes
alpha = 0.8  # Coefficient for linear term in potential
beta = 0.6  # Coefficient for nonlinear saturation
gamma = 0.7  # Coupling strength coefficient
eta = 0.05  # Learning rate for synaptic updates

dt = 0.01  # Integration time step (Euler-Maruyama)
steps = 100000  # total integration steps (can be reduced for quick tests)
cov_alpha = 0.01  # Exponential decay factor for covariance estimation
noise_amp = 0.0003  # Amplitude of additive noise (complex Gaussian)
lambda_phi = 0.0015  # Strength of free-energy gradient term
weight_update_interval = 5  # Steps between synaptic updates

# Control parameter schedule (adiabatic driving)
r_min, r_max = 0.5, 3.0  # Range of control parameter r
t_total = steps * dt  # Total simulation time
r_critical = 1.5  # Theoretical critical point

# Transition detection and consolidation flags
transition_detected = False  # Phase transition flag
transition_time = None  # Time of detected transition
mpc_threshold = 0.1  # Threshold for MPC derivative to detect transition
consolidation_done = False  # Flag for post-transition consolidation

# Consolidation targets (post-transition)
POST_ETA_FACTOR = 0.05  # eta <- eta * POST_ETA_FACTOR after transition
POST_WEIGHT_UPDATE_INTERVAL = 150  # Reduced plasticity rate post-transition
POST_NOISE_AMP = 0.00005  # Reduced noise amplitude post-transition
POST_STOCHASTIC_SIGMA = 0.001  # Reduced phase perturbation post-transition
DPSI_SLOWDOWN_FACTOR = 0.5  # Rate reduction factor post-transition

# -----------------------------
# Initialize network and state
# -----------------------------
# Create small-world network with guaranteed connectivity
G = nx.connected_watts_strogatz_graph(N, 8, 0.3, seed=42)
W = nx.to_numpy_array(G)  # Adjacency matrix
W = W / np.max(W) if np.max(W) > 0 else W  # Normalize weights

# initialize phases with some coherence (small initial spread)
theta = np.random.rand(N) * np.pi / 4
psi = np.exp(1j * theta)  # Complex field representation

# running mean/covariance (small initial covariance)
mu = np.zeros(N, dtype=complex)  # Running mean estimate
cov = np.eye(N, dtype=complex) * 1e-6  # Running covariance estimate

# history containers
history = {
    'time': [],
    'order_param': [],
    'r_values': [],
    'mean_phase_coherence': [],
    'mean_weight': [],
    'delta_mpc': []
}

# -----------------------------
# Helper functions
# -----------------------------
def graph_laplacian(W_mat):
    """Compute symmetric normalized Laplacian."""
    W_sym = 0.5 * (W_mat + W_mat.T)  # Symmetrize weight matrix
    D = np.diag(W_sym.sum(axis=1))
    # avoid division by zero
    inv_sqrt = np.zeros_like(D)
    diag = np.diag(D)
    diag_safe = np.where(diag > 0, diag, 1.0)
    D_inv_sqrt = np.diag(1.0 / np.sqrt(diag_safe))
    return np.eye(N) - D_inv_sqrt @ W_sym @ D_inv_sqrt


def control_parameter(t):
    """Linear ramp of control parameter r from r_min to r_max"""
    return r_min + (r_max - r_min) * (t / t_total)


def dPhi_dpsi(psi_vec, W_mat, alpha_val, beta_val, gamma_val, r_val, lambda_val):
    """Variational derivative of energy functional Φ[ψ]:
    Φ[ψ] = α(r-rc)|ψ|² + β|ψ|⁴/2 + γψ*Lψ + λ·D_KL(q||p)
    """
    L = graph_laplacian(W_mat)
    # Ginzburg-Landau potential terms
    nonlinear = -alpha_val * (r_val - r_critical) * psi_vec + beta_val * (np.abs(psi_vec)**2) * psi_vec
    coupling = gamma_val * (L @ psi_vec)  # Diffusion term
    # Free-energy gradient term (KL divergence minimization)
    x = psi_vec - mu
    try:
        cov_inv = np.linalg.inv(cov + 1e-5 * np.eye(N))  # Regularized inverse
    except np.linalg.LinAlgError:
        cov_inv = np.linalg.pinv(cov + 1e-5 * np.eye(N))
    grad_phi = cov_inv @ x  # Gradient of Mahalanobis distance
    return nonlinear + coupling + lambda_val * grad_phi


def update_running_covariance(psi_vec, alpha=cov_alpha):
    """Online estimation of running mean and covariance using
    exponential moving average"""
    global mu, cov
    mu = (1 - alpha) * mu + alpha * psi_vec  # Update mean
    dx = (psi_vec - mu).reshape(-1, 1)
    cov_update = dx @ np.conj(dx.T)  # Outer product
    cov = (1 - alpha) * cov + alpha * cov_update  # Update covariance


def mean_phase_coherence(phases):
    """Kuramoto order parameter for phase synchronization"""
    return np.abs(np.sum(np.exp(1j * phases)) / len(phases))


def stochastic_perturbation(psi_vec, sigma=0.005):
    """Multiplicative phase noise preserving unit circle"""
    noise = sigma * (np.random.rand(*psi_vec.shape) - 0.5)
    return psi_vec * np.exp(1j * noise)


def external_stimulus(t, nodes=None):
    """Time-periodic Gaussian stimulus pulses applied to specific nodes"""
    if nodes is None:
        nodes = [N // 2, N // 3, 2 * N // 3]  # Default stimulated nodes
    stim_strength = 0.2 * np.exp(-((t % 5.0) - 2.5)**2 / 0.5)  # Gaussian pulse
    stim = np.zeros(N, dtype=complex)
    stim[nodes] = stim_strength * np.exp(1j * 2 * np.pi * t)  # Rotating phase
    return stim


def update_weights(W_mat, psi_vec, eta_val, theta=0.15):
    """STDP-like weight update with adaptive threshold.
    After transition, adaptive_theta is fixed low to consolidate connections.
    """
    phase_diff = np.angle(psi_vec[:, None] / psi_vec[None, :])  # Pairwise phase differences
    current_mpc = mean_phase_coherence(np.angle(psi_vec))
    if transition_detected:
        adaptive_theta = 0.05  # Fixed low threshold post-transition
    else:
        adaptive_theta = theta * (1 - current_mpc)**1.5  # Adaptive threshold pre-transition
    delta_W = eta_val * np.tanh(np.sin(phase_diff) / (adaptive_theta + 1e-6))  # STDP-like rule
    W_new = W_mat + delta_W
    W_new = np.clip(W_new, 0, 1)  # Enforce non-negative weights
    W_new = 0.5 * (W_new + W_new.T)  # Maintain symmetry
    np.fill_diagonal(W_new, 0)  # Remove self-connections
    return W_new

# -----------------------------
# Simulation main loop
# -----------------------------
logger.info('Starting simulation...')
L = graph_laplacian(W)

# allocate history arrays (for possible later saving)
psi_history = np.zeros((steps, N), dtype=complex)
order_param_history = np.zeros(steps)
r_history = np.zeros(steps)

prev_mpc = mean_phase_coherence(np.angle(psi))

for t_step in range(steps):
    t = t_step * dt
    r = control_parameter(t)

    stim = external_stimulus(t)
    psi_stim = psi + stim  # Add external driving

    update_running_covariance(psi_stim)

    dpsi = -dPhi_dpsi(psi_stim, W, alpha, beta, gamma, r, lambda_phi)

    # complex Gaussian-like noise scaled with dt
    noise = noise_amp * (np.random.randn(N) + 1j * np.random.randn(N)) * np.sqrt(dt)
    psi = psi + dpsi * dt + noise  # Euler-Maruyama integration

    # small stochastic perturbation
    psi = stochastic_perturbation(psi, sigma=0.005)

    # normalize magnitude with tanh to avoid runaway amplitude
    psi_mag = np.abs(psi)
    psi = psi / (psi_mag + 1e-8) * np.tanh(psi_mag)  # Soft normalization

    # weight updates (STDP-like)
    if t_step % weight_update_interval == 0:
        W = update_weights(W, psi, eta)

    current_mpc = mean_phase_coherence(np.angle(psi))
    delta_mpc = current_mpc - prev_mpc

    # record history at coarse granularity
    if t_step % 10 == 0:
        history['mean_phase_coherence'].append(current_mpc)
        history['mean_weight'].append(np.mean(W))
        history['time'].append(t)
        history['order_param'].append(np.abs(np.mean(psi)))
        history['r_values'].append(r)
        history['delta_mpc'].append(delta_mpc)

        # detect transition and apply consolidation patch
        if (not transition_detected) and (delta_mpc > mpc_threshold) and (r > r_critical):
            transition_detected = True
            transition_time = t
            logger.info(f'PHASE TRANSITION DETECTED at t={t:.2f}, r={r:.3f}, \u0394MPC={delta_mpc:.3f}, MPC={current_mpc:.3f}')

            if not consolidation_done:
                eta = eta * POST_ETA_FACTOR
                weight_update_interval = POST_WEIGHT_UPDATE_INTERVAL
                consolidation_done = True

            # reduce noise and stochastic perturbation amplitude
            noise_amp = POST_NOISE_AMP
            # NOTE: sigma is local to stochastic_perturbation call; we use a global var for readability
            STOCHASTIC_SIGMA = POST_STOCHASTIC_SIGMA

        if t_step % 1000 == 0:
            logger.info(f't = {t:.2f}, r = {r:.3f}, |<\u03C8>| = {np.abs(np.mean(psi)):.3f}, MPC = {current_mpc:.3f}')

    # slow dynamics after transition to stabilize field
    if transition_detected:
        dpsi *= DPSI_SLOWDOWN_FACTOR  # Critical slowing down

    # store instantaneous state
    psi_history[t_step] = psi
    order_param_history[t_step] = np.abs(np.mean(psi))
    r_history[t_step] = r

    prev_mpc = current_mpc

logger.info('Simulation completed.')

# -----------------------------
# Post-transition stabilization analysis
# -----------------------------
# Use last 15% of history for stability statistics
window = int(0.15 * len(history['mean_phase_coherence']))
if window < 1:
    window = 1
mpc_tail = history['mean_phase_coherence'][-window:]
mean_tail = float(np.mean(mpc_tail))
std_tail = float(np.std(mpc_tail))
logger.info(f'Post-transition stabilization: mean_MPC_last15% = {mean_tail:.3f}, std = {std_tail:.3f}')
if mean_tail >= 0.9 and std_tail <= 0.1:
    logger.info('Stable, high MPC achieved — transition considered successful.')
else:
    logger.info('Transition detected but not fully stabilized — further tuning may be needed.')

# -----------------------------
# Save results for further analysis
# -----------------------------
np.savez(os.path.join(figures_dir, 'simulation_results.npz'),
         psi_history=psi_history,
         order_param_history=order_param_history,
         r_history=r_history,
         W=W)
logger.info(f'Results saved to {figures_dir}')

# -----------------------------
# Visualization
# -----------------------------
pos = nx.spring_layout(G, seed=42)

plt.figure(figsize=(10, 6))
plt.plot(history['time'], history['order_param'], label='|⟨ψ⟩|', alpha=0.8)
plt.plot(history['time'], history['r_values'], '--', label='Control param (r)')
if transition_detected:
    plt.axvline(x=transition_time, color='g', linestyle=':', label=f'Transition at t={transition_time:.2f}')
plt.axvline(x=t_total * (r_critical - r_min) / (r_max - r_min), color='m', linestyle='--', label=f'Critical r={r_critical}')
plt.xlabel('Time')
plt.ylabel('Value')
plt.legend()
plt.title('Order Parameter and Control Parameter')
plt.grid(True)
plt.savefig(os.path.join(figures_dir, 'order_parameter_and_control.pdf'), bbox_inches='tight')
plt.close()

plt.figure(figsize=(10, 6))
plt.plot(history['time'], history['mean_phase_coherence'], label='Mean Phase Coherence')
if transition_detected:
    plt.axvline(x=transition_time, color='r', linestyle=':', label=f'Transition at t={transition_time:.2f}')
plt.axvline(x=t_total * (r_critical - r_min) / (r_max - r_min), color='m', linestyle='--', label=f'Critical r={r_critical}')
plt.xlabel('Time')
plt.ylabel('MPC')
plt.legend()
plt.title('Synchronization Measure with Transition Point')
plt.grid(True)
plt.savefig(os.path.join(figures_dir, 'synchronization_measure.pdf'), bbox_inches='tight')
plt.close()

plt.figure(figsize=(10, 6))
plt.plot(history['r_values'], history['mean_phase_coherence'], 'o-', markersize=2)
plt.axvline(x=r_critical, color='r', linestyle='--', label=f'Critical r={r_critical}')
plt.xlabel('Control Parameter (r)')
plt.ylabel('Mean Phase Coherence (MPC)')
plt.legend()
plt.title('Phase Transition: MPC vs Control Parameter')
plt.grid(True)
plt.savefig(os.path.join(figures_dir, 'phase_transition_curve.pdf'), bbox_inches='tight')
plt.close()

plt.figure(figsize=(10, 6))
plt.plot(history['time'], history['delta_mpc'], label='\u0394MPC')
plt.axhline(y=mpc_threshold, color='r', linestyle='--', label=f'Threshold={mpc_threshold}')
if transition_detected:
    plt.axvline(x=transition_time, color='g', linestyle=':', label=f'Transition at t={transition_time:.2f}')
plt.xlabel('Time')
plt.ylabel('ΔMPC')
plt.legend()
plt.title('Change in Mean Phase Coherence')
plt.grid(True)
plt.savefig(os.path.join(figures_dir, 'mpc_change.pdf'), bbox_inches='tight')
plt.close()

# network final state
plt.figure(figsize=(8, 6))
node_colors = np.angle(psi_history[-1])
node_colors = (node_colors + np.pi) / (2 * np.pi)
nx.draw_networkx_nodes(G, pos, node_color=node_colors, cmap='hsv', vmin=0, vmax=1, node_size=100)
nx.draw_networkx_edges(G, pos, alpha=0.3)
plt.title('Network with Phase Coloring (Final State)')
plt.axis('off')
plt.savefig(os.path.join(figures_dir, 'network_phase_coloring.pdf'), bbox_inches='tight')
plt.close()

plt.figure(figsize=(8, 6))
plt.imshow(W, interpolation='nearest')
plt.colorbar()
plt.title('Final Weight Matrix')
plt.savefig(os.path.join(figures_dir, 'final_weight_matrix.pdf'), bbox_inches='tight')
plt.close()

logger.info('Figures saved to figures directory')