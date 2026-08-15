# Real-Time FDIA Detection in Nuclear Power Plants using Time-Series Transformers

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![MATLAB](https://img.shields.io/badge/MATLAB-R2023a-e16723.svg)](https://www.mathworks.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository contains the simulation, dataset generation, and machine learning defense pipeline for detecting **False Data Injection Attacks (FDIA)** in nuclear reactor cyber-physical systems. By coupling a deterministic MATLAB digital twin with a stochastic PyTorch Time-Series Transformer, this project demonstrates a highly accurate methodology for securing critical infrastructure against zero-residual eigenvector attacks.

---

## 1. Motivation
The assumption that critical infrastructure is inherently secure due to "air-gapping" is obsolete. In 2019, the Kudankulam Nuclear Power Plant (KKNPP) in India suffered a significant cyber intrusion involving the Dtrack malware. While the breach was contained to the administrative network and did not compromise the reactor's control systems, it exposed a chilling reality: state-sponsored threat actors possess the capability to penetrate the IT networks of active nuclear facilities. 

If an attacker bridges the gap between the IT network and the Operational Technology (OT) network, they can manipulate Supervisory Control and Data Acquisition (SCADA) systems. A successful attack on the physical telemetry of a nuclear core could lead to catastrophic thermal limit violations or core meltdowns. Defending against these advanced cyber-physical threats requires intrusion detection systems that understand the fundamental laws of thermodynamics.

## 2. Problem Statement
Modern attackers do not simply shut down sensors; they employ **Stealth False Data Injection Attacks (FDIA)**. By understanding the system's underlying mathematical topology, an attacker can inject malicious data vectors that spoof critical sensor readings (e.g., artificially lowering the reported core temperature) while simultaneously manipulating secondary variables (e.g., neutron flux) to keep the state-estimation residuals perfectly balanced at zero. 

Because the mathematical equations appear balanced, traditional SCADA threshold alarms and standard anomaly detectors fail to flag the malicious injection, allowing the physical reactor to drift toward a critical failure state.

## 3. Existing Solutions
Historically, power plants rely on:
* **State Estimation & Kalman Filters:** Highly effective against random sensor noise, but mathematically blind to zero-residual eigenvector injections.
* **Recurrent Neural Networks (RNNs/LSTMs):** While capable of time-series analysis, recurrent architectures struggle with the vanishing gradient problem over long sequences and fail to efficiently capture the slow-moving, long-term thermodynamic dependencies inherent in nuclear kinetics (such as Xenon-135 poisoning buildup).

## 4. Methods and Methodologies
This project adopts a dual-stack cyber-physical methodology:
1. **Physical Simulation (MATLAB):** The `reactor_simulation.m` and `simulation_graph.m` scripts model the non-linear reactor point kinetics and thermodynamic coupling in a strict state-space matrix. This serves as a digital twin, generating high-fidelity baseline data and simulating sophisticated, mathematically sound stealth attacks.
2. **Deep Learning Optimization (Python/PyTorch):** The `main_final.ipynb` notebook engineers a deep learning pipeline to ingest the telemetry. The data is processed into multidimensional sliding windows and fed into a custom classification engine. 

## 5. Proposed Solution: The Time-Series Transformer
To overcome the limitations of LSTMs, this repository proposes a **Time-Series Transformer** architecture. 
* **Multi-Head Self-Attention:** Instead of processing time sequentially, the Transformer analyzes all variables across a 40-step operational time window simultaneously. 
* **Thermodynamic Cross-Correlation:** The attention mechanism inherently learns the physical coupling between variables. Even if an attacker perfectly balances the mathematical residual of the temperature sensor, the Transformer detects the micro-divergences in delayed variables (like Xenon residuals), exposing the spoofed telemetry.

## 6. Dataset and Results
The model was trained and evaluated on a custom-engineered Parquet dataset comprising **1.38 million sliding windows** spanning normal operations, naive attacks, and zero-residual stealth attacks. 

### Model Performance Metrics
* **Total Parameters Optimized:** ~150,000+
* **Aggregate Accuracy:** 99.92%
* **F1-Score (Stealth Attack):** 0.9986
* **False Positive Rate (Normal State):** ~0.039% (Critically low, ensuring the reactor is not subjected to costly false scrams).

### Visualizing the Physical Divergence
*(Note: Upload your PDF/PNG graphics to the `Results` folder and link them here)*

The Transformer effectively maps the divergence between the true physical state and the spoofed SCADA data. As demonstrated in the results, the AI detects the thermodynamic decoupling almost instantly after the eigenvector injection, well before the true core temperature reaches critical meltdown thresholds.

![Confusion Matrix](Results/confusion_matrix_results.png)
*(Figure 1: The model achieves a near-perfect classification matrix, strictly limiting false positive alarms.)*

![Timeline Trajectory](Results/stealth_attack_timeline_crisp.png)
*(Figure 2: Real-time detection timeline showcasing the AI flagging the exact moment the spoofed sensor reading deviates from the true physical meltdown trajectory.)*

## 7. Outcome
The deployment of the Time-Series Transformer successfully closes the vulnerability gap exploited by stealth FDIAs. By shifting the detection paradigm from purely mathematical state-estimation to physics-aware, self-attention neural networks, the system can autonomously identify and flag cyber-physical intrusions with sub-second latency.

## 8. Conclusion
As nuclear facilities modernize and digitize, the attack surface for advanced persistent threats expands. This project demonstrates that combining rigorous deterministic operations research (via MATLAB digital twins) with stochastic deep learning (PyTorch Transformers) yields state-of-the-art security for critical infrastructure. The resulting model not only achieves unparalleled accuracy but does so while strictly respecting the complex kinetic reality of a nuclear reactor.
