# Quantum Computing & QML — Python Reference

A consolidated reference distilled from a set of overview infographics (library landscapes,
QML frameworks, QML architecture, quantum states), corrected and annotated. Conceptual
material (states, the QML workflow) is stable; **library versions, package names, vendor
offerings, and activity levels move fast — verify volatile specifics against current docs.**

## How to reuse this
- This file is the portable piece. Drop it in your references folder; for any **future project**,
  upload it or add it to that Claude Project's knowledge base and it's instantly back in context.
- Claude's *memory* is scoped per project, so it won't carry this across projects — the file does.

---

## 1. The library landscape (grouped by what you'd reach for)

### Circuit construction & full-stack SDKs
| Library | By | What it's for |
|---|---|---|
| **Qiskit** | IBM | Build, simulate (Aer), and run circuits on IBM Quantum hardware. Largest ecosystem. |
| **Cirq** | Google | NISQ circuit construction with fine-grained, low-level control. |
| **Qibo** | Community | High-performance circuit framework + simulation backends. |
| **ProjectQ** | ETH Zurich | Hardware-agnostic compilation and simulation. |

### Quantum machine learning
| Library | By | What it's for |
|---|---|---|
| **PennyLane** | Xanadu | The leading hybrid/differentiable QML framework; device-agnostic; autodiff with PyTorch/TF/JAX. |
| **Qiskit Machine Learning** | IBM | QML on the Qiskit stack — QNNs (Estimator/Sampler), quantum kernels, VQC; scikit-learn-style API. |
| **TensorFlow Quantum** | Google | Hybrid QML integrated with TensorFlow + Cirq. |
| **Tequila** | Community | High-level abstraction for writing variational algorithms across backends. |

### Simulators (state-vector / dynamics / high-performance)
| Library | By | What it's for |
|---|---|---|
| **QuTiP** | Community | Quantum *dynamics* — open quantum systems, master equations (not a circuit SDK). |
| **Qulacs** | Community | Very fast state-vector circuit simulator for large qubit counts. |
| **Yao** | Community | High-performance simulator/framework — **Julia, not Python** (the odd one out in these lists). |

### Hardware access / cloud
| Library | By | What it's for |
|---|---|---|
| **Amazon Braket SDK** | AWS | Build/run on multiple hardware backends + managed simulators. |
| **Cirq-on-Braket** | — | Run Cirq circuits on Braket-hosted hardware. |
| **D-Wave Ocean SDK** | D-Wave | Quantum *annealing* — map problems to QUBO/Ising for optimization. |
| **pyQuil / Forest** | Rigetti | Program Rigetti QPUs (Quil language). |
| **Pulser** | Pasqal | Pulse-level control of neutral-atom hardware. |
| **MindQuantum** | Huawei | QC framework on the MindSpore stack. |
| **Quantum Inspire SDK** | QuTech | Access to QuTech's hardware/simulators. |

### Domain-specific
| Library | By | What it's for |
|---|---|---|
| **OpenFermion** | Google | Quantum chemistry — fermionic simulations, molecular Hamiltonians. |
| **Strawberry Fields** | Xanadu | Photonic / continuous-variable quantum computing. |

### Compilation, IR & tooling
| Library | By | What it's for |
|---|---|---|
| **TKET / pytket** | Quantinuum | Retargetable circuit compiler/optimizer; hardware-aware compilation. |
| **OpenQASM 3 / pyqasm** | OpenQASM | Standard quantum assembly language / its Python interface. |
| **Mitiq** | Unitary Fund | Backend-agnostic *error mitigation* for noisy (NISQ) results. |

---

## 2. Picking a QML framework — PennyLane vs Qiskit ML vs Cirq

| Aspect | PennyLane | Qiskit Machine Learning | Cirq |
|---|---|---|---|
| By | Xanadu | IBM | Google |
| Core focus | Hybrid QML, differentiable programming | QML algorithms on the Qiskit stack | Low-level circuit construction |
| Differentiable programming | Native (Autograd, JAX, TF, PyTorch) | Via connectors | Not native (manual/parameter-shift) |
| ML integration | Excellent (PyTorch/TF/JAX) | Good (scikit-learn, PyTorch connector) | Limited (no built-in ML) |
| Hardware | Many, via plugins (IBM, Rigetti, IonQ…) | IBM Quantum | Google + via Braket |
| Best for | Hybrid/variational QML, research | QML in the IBM ecosystem, ready-made algorithms | Custom circuit design & control |

**Rule of thumb:** PennyLane if differentiable hybrid QML is the point · Qiskit ML if you want the
IBM ecosystem and ready algorithms · Cirq if you need explicit low-level circuit control. All three
prototype on simulators — no hardware required.

---

## 3. QML architecture & workflow (the standard hybrid loop)

**Pipeline:** input data → quantum **encoding** → **variational/parameterized quantum circuit (VQC)**
→ **measurement** (expectation values of observables, e.g. ⟨Z⟩, ⟨X⟩) → classical post-processing
→ **loss** → **classical optimizer** updates the circuit parameters → repeat.

- **Encoding techniques:** amplitude · angle · basis · feature maps (e.g. ZZFeatureMap).
- **Core algorithms & uses:**
  - **VQC** — variational classifier (classification)
  - **QSVC / QSVR** — quantum-kernel support vector classification/regression
  - **VQE** — variational eigensolver (ground-state energy; chemistry/physics)
  - **QAOA** — combinatorial optimization
  - **QGAN** — generative adversarial modeling
  - **QNN** — general quantum neural network for ML tasks
- **Optimizers:** COBYLA, SPSA, Adam, gradient descent.
- **Gradients:** the **parameter-shift rule** — ∂L/∂θ = ½ [ L(θ⁺) − L(θ⁻) ] — gives exact circuit
  gradients, so a classical optimizer can train the quantum parameters (θ ← θ − η ∇L).
- **Loss functions:** cross-entropy, MSE, quantum-fidelity loss.

---

## 4. Quantum states — quick reference

| State | What it is | Key facts | Form |
|---|---|---|---|
| **Pure** | Full information about the system | Single wavefunction; no classical uncertainty | \|ψ⟩ = α\|0⟩ + β\|1⟩, \|α\|²+\|β\|²=1 |
| **Mixed** | State unknown, probabilities known | Incomplete info; needs a density matrix | ρ = Σ pᵢ \|ψᵢ⟩⟨ψᵢ\| |
| **Single-qubit** | One qubit | Lives on the Bloch sphere | \|ψ⟩ = α\|0⟩ + β\|1⟩ |
| **Multi-qubit** | Two or more qubits | State space 2ⁿ; can entangle | \|ψ⟩ = Σ cᵢ\|i⟩ over 2ⁿ basis states |
| **Superposition** | Multiple states at once | Enables parallelism; collapses on measurement | (\|0⟩ + \|1⟩)/√2 |
| **Entangled** | Correlated beyond classical | Measuring one affects the other; key resource | (\|00⟩ + \|11⟩)/√2 (Bell) |
| **Product / separable** | Multi-qubit, not entangled | Factorizes; qubits independent | \|ψ⟩ = \|a⟩ ⊗ \|b⟩ |
| **Basis** | Building blocks | Any state = combination of basis states | \|0⟩, \|1⟩ (computational) |
| **Coherent** | Resembles a classical wave | Minimum-uncertainty; common in quantum optics | \|α⟩ = e^(−\|α\|²/2) Σ (αⁿ/√n!) \|n⟩ |
| **Ground** | Lowest-energy state | Most stable; system relaxes toward it | lowest-energy eigenstate of H |
| **Excited** | Above the ground state | Temporary; relaxes back down | any higher-energy eigenstate |
| **Thermal** | Equilibrium at temperature T | Mixture of energy states; T-dependent | ρ = e^(−H/kT) / Z, Z = Tr(e^(−H/kT)) |

**Bloch sphere (single qubit):** any pure single-qubit state is
\|ψ⟩ = cos(θ/2)\|0⟩ + e^(iφ) sin(θ/2)\|1⟩, with 0 ≤ θ ≤ π, 0 ≤ φ < 2π.

---

## 5. Honest notes on applying QML to scientific-ML problems

- **No demonstrated advantage** for dense 3D segmentation or large-scale field data (e.g. a U-Net
  on density cubes). Quantum models classify *low-dimensional* inputs into a *few* outputs; loading
  high-dimensional data into a quantum state is itself a bottleneck (state preparation can cost back
  whatever you'd hope to save), and per-voxel labeling isn't a shape these models take.
- **Where it does fit:** small variational classifiers or quantum kernels on a *reduced* feature set,
  framed as research / benchmarking a hybrid artifact — not as a speedup.
- **Entry points:** PennyLane (hybrid, autodiff) or Qiskit ML (ecosystem). Simulators are enough to
  prototype; no quantum hardware needed.
- **Era caveat:** this is NISQ-era tooling. Expect parity, not magic, on real problems, and re-check
  library status/versions when you actually build.
