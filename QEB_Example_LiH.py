import numpy as np
import matplotlib.pyplot as plt
from time import time

# Qiskit imports
from qiskit_nature.units import DistanceUnit
from qiskit_nature.second_q.drivers import PySCFDriver
from qiskit_nature.second_q.mappers import JordanWignerMapper
from qiskit_nature.second_q.transformers import FreezeCoreTransformer
from qiskit_nature.second_q.circuit.library import HartreeFock
from qiskit_algorithms import VQE, NumPyMinimumEigensolver
from qiskit_algorithms.optimizers import SPSA
from qiskit.primitives import StatevectorEstimator

# Import our QEB ansatz
from qeb_ansatz import QEBansatz


def main():
    print("=" * 80)
    print("QEB ANSATZ ON LiH: COMPLETE EXAMPLE")
    print("=" * 80)

    # ========================================================================
    # STEP 1: DEFINE LiH MOLECULE
    # ========================================================================
    print("\n[1] Setting up LiH molecule...")

    driver = PySCFDriver(
        atom="Li 0 0 0; H 0 0 1.595",  # LiH equilibrium bond length
        basis="sto3g",
        charge=0,
        spin=0,
        unit=DistanceUnit.ANGSTROM,
    )

    problem = driver.run()
    print(f"    Spatial orbitals: {problem.num_spatial_orbitals}")
    print(f"    Particles: {problem.num_particles}")
    print(f"    Nuclear repulsion: {problem.nuclear_repulsion_energy:.6f} Ha")

    # ========================================================================
    # STEP 1B: APPLY FROZEN CORE APPROXIMATION
    # ========================================================================
    print("\n[1B] Applying frozen core approximation...")

    # Freeze the core electrons (1s orbital in Li, innermost)
    transformer = FreezeCoreTransformer(freeze_core=True, remove_orbitals=[5])
    problem = transformer.transform(problem)

    print(f"    After freezing core:")
    print(f"    Spatial orbitals: {problem.num_spatial_orbitals}")
    print(f"    Particles: {problem.num_particles}")

    # Get the frozen core energy offset
    frozen_core_energy = 0.0
    if hasattr(transformer, 'shifted_value'):
        frozen_core_energy = transformer.shifted_value
    print(f"    Core energy offset: {frozen_core_energy:.6f} Ha")

    # ========================================================================
    # STEP 2: GET HAMILTONIAN AND MAP TO QUBITS
    # ========================================================================
    print("\n[2] Mapping to qubit Hamiltonian...")

    hamiltonian = problem.hamiltonian
    second_q_op = hamiltonian.second_q_op()

    mapper = JordanWignerMapper()
    qubit_op = mapper.map(second_q_op)

    print(f"    Qubits required: {qubit_op.num_qubits}")
    print(f"    Hamiltonian terms: {len(qubit_op)}")

    # ========================================================================
    # STEP 3: COMPUTE EXACT SOLUTION (NumPy)
    # ========================================================================
    print("\n[3] Computing exact ground state energy...")

    numpy_solver = NumPyMinimumEigensolver()
    numpy_result = numpy_solver.compute_minimum_eigenvalue(qubit_op)

    exact_electronic = numpy_result.eigenvalue.real
    exact_total = exact_electronic + problem.nuclear_repulsion_energy

    print(f"    Exact electronic energy: {exact_electronic:.8f} Ha")
    print(f"    Exact total energy:      {exact_total:.8f} Ha")

    # ========================================================================
    # STEP 4: CREATE QEB ANSATZ
    # ========================================================================
    print("\n[4] Creating QEB ansatz...")

    num_qubits = qubit_op.num_qubits
    num_particles = problem.num_particles  # (n_alpha, n_beta)
    num_spatial_orbitals = problem.num_spatial_orbitals

    qeb_ansatz = QEBansatz(
        num_qubits=num_qubits,
        num_particles=num_particles,
        num_spatial_orbitals=num_spatial_orbitals,
        depth=6,  # Increase to 6 for better expressiveness with Givens rotations
        include_double=True,  # Include double excitations
        use_hartree_fock_init=True,  # Use Qiskit's HartreeFock initialization
        mapper=mapper,  # Pass mapper for proper HartreeFock construction
    )

    info = qeb_ansatz.get_info()
    print(f"    QEB configuration:")
    for key, val in info.items():
        if key != "ansatz_type":
            print(f"      {key}: {val}")

    # ========================================================================
    # STEP 5: INITIALIZE PARAMETERS
    # ========================================================================
    print("\n[5] Initializing parameters...")

    # Start from zero (Hartree-Fock reference) with small random perturbation
    initial_params = np.zeros(qeb_ansatz.num_parameters)
    np.random.seed(42)
    initial_params += np.random.normal(0, 0.01, qeb_ansatz.num_parameters)
    print(f"    Parameters initialized: {len(initial_params)}")
    print(f"    Param range: [{initial_params.min():.4f}, {initial_params.max():.4f}]")

    # ========================================================================
    # STEP 6: RUN VQE WITH QEB ANSATZ
    # ========================================================================
    print("\n[6] Running VQE with QEB ansatz...")
    print("    (This may take 1-2 minutes)")

    convergence_history = []
    start_time = time()

    def callback(eval_count, parameters, mean, std):
        """Track convergence"""
        convergence_history.append(mean)
        elapsed = time() - start_time
        if len(convergence_history) % 20 == 0:
            print(
                f"      Eval {len(convergence_history):3d}: "
                f"E = {mean:.6f} Ha, "
                f"Error = {abs(mean - exact_electronic)*1000:.2f} mHa, "
                f"Time = {elapsed:.1f}s"
            )

    # Create VQE solver
    estimator = StatevectorEstimator()
    # SPSA does 2 evaluations per iteration, so maxiter=1000 → ~2000 evaluations on graph
    # Tuned for QEB: lower perturbation to reduce noise, allow more iterations for convergence
    optimizer = SPSA(
        maxiter=2000,  # Increase iterations to allow better convergence
        learning_rate=0.01,
        perturbation=0.01,  # Reduce from 0.05 to 0.01 to avoid divergence
    )

    vqe_solver = VQE(estimator, qeb_ansatz, optimizer, callback=callback)
    vqe_solver.initial_point = initial_params

    # Run optimization
    vqe_result = vqe_solver.compute_minimum_eigenvalue(qubit_op)
    elapsed_total = time() - start_time

    vqe_electronic = vqe_result.eigenvalue.real
    vqe_total = vqe_electronic + problem.nuclear_repulsion_energy

    # ========================================================================
    # STEP 7: RESULTS AND COMPARISON
    # ========================================================================
    print("\n" + "=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)

    print(f"\nElectronic Energy:")
    print(f"  Exact (NumPy):      {exact_electronic:.8f} Ha")
    print(f"  VQE (QEB):          {vqe_electronic:.8f} Ha")
    print(f"  Error:              {abs(vqe_electronic - exact_electronic):.2e} Ha")
    print(f"  Error (mHa):        {abs(vqe_electronic - exact_electronic)*1000:.2f} mHa")

    print(f"\nTotal Energy (with nuclear repulsion):")
    print(f"  Exact (NumPy):      {exact_total:.8f} Ha")
    print(f"  VQE (QEB):          {vqe_total:.8f} Ha")
    print(f"  Error:              {abs(vqe_total - exact_total):.2e} Ha")

    print(f"\nOptimization Statistics:")
    print(f"  Total evaluations:  {len(convergence_history)}")
    print(f"  Total time:         {elapsed_total:.1f} seconds")
    print(f"  Time per eval:      {elapsed_total/len(convergence_history):.2f} s")
    print(f"  Initial energy:     {convergence_history[0]:.8f} Ha")
    print(f"  Final energy:       {convergence_history[-1]:.8f} Ha")
    print(f"  Energy decrease:    {convergence_history[0] - convergence_history[-1]:.8f} Ha")

    # ========================================================================
    # STEP 8: VISUALIZATION
    # ========================================================================
    print("\n[7] Creating convergence plots...")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    iterations = np.arange(1, len(convergence_history) + 1)
    errors = np.array(convergence_history) - exact_electronic

    # Plot 1: Energy vs Iteration (with tight scaling to show convergence clearly)
    axes[0].plot(iterations, convergence_history, "b.-", linewidth=2, markersize=3, label="VQE Energy")
    axes[0].axhline(exact_electronic, color="r", linestyle="--", linewidth=2, label="Exact Ground State")
    axes[0].set_xlabel("Iteration", fontsize=12)
    axes[0].set_ylabel("Electronic Energy (Ha)", fontsize=12)
    axes[0].set_title("LiH VQE Convergence (Frozen Core, 8 Qubits)", fontsize=13, fontweight="bold")
    # Tight y-axis scaling to show convergence clearly
    energy_min = min(exact_electronic, min(convergence_history)) - 0.003
    energy_max = max(exact_electronic, max(convergence_history)) + 0.003
    axes[0].set_ylim([energy_min, energy_max])
    axes[0].legend(fontsize=11, loc="upper right")
    axes[0].grid(True, alpha=0.3)

    # Plot 2: Absolute Error vs Iteration (log scale)
    axes[1].semilogy(iterations, np.abs(errors), "g.-", linewidth=2, markersize=3)
    axes[1].set_xlabel("Iteration", fontsize=12)
    axes[1].set_ylabel("Absolute Error (Ha, log scale)", fontsize=12)
    axes[1].set_title("LiH VQE: Absolute Energy Error", fontsize=13, fontweight="bold")
    axes[1].grid(True, alpha=0.3, which="both")

    plt.tight_layout()
    plt.savefig("qeb_convergence_lih.png", dpi=150, bbox_inches="tight")
    print(f"    Plot saved as 'qeb_convergence_lih.png'")

    plt.show()

    # ========================================================================
    # STEP 9: SAVE RESULTS FOR COMPARISON
    # ========================================================================
    print("\n[8] Saving results for analysis...")

    results_dict = {
        "ansatz": "QEB",
        "depth": qeb_ansatz.depth,
        "num_parameters": qeb_ansatz.num_parameters,
        "exact_energy": exact_electronic,
        "vqe_energy": vqe_electronic,
        "final_error_mha": abs(vqe_electronic - exact_electronic) * 1000,
        "convergence_history": convergence_history,
        "total_time": elapsed_total,
        "total_evals": len(convergence_history),
    }

    # Save to file for later comparison
    import json

    with open("qeb_results_lih.json", "w") as f:
        json.dump(
            {k: v for k, v in results_dict.items() if k != "convergence_history"},
            f,
            indent=2,
        )
    print(f"    Results saved as 'qeb_results_lih.json'")

    print("\n" + "=" * 80)
    print("✓ QEB Ansatz Example Complete!")
    print("=" * 80)

    return results_dict


if __name__ == "__main__":
    results = main()
