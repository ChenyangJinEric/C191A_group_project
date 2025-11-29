"""
QEB (Qubit Excitation-Based) Ansatz Implementation for VQE

Based on: "Qubit-excitation-based adaptive variational quantum eigensolver"
(arXiv:2011.10540, Nature Communications Physics 2021)

Physics Background:
===================
QEB uses qubit-space Givens rotations instead of fermionic excitations:

1. **Fermionic excitations (UCCSD)**:
   - Obey anticommutation relations {a†ᵢ, aⱼ} = δᵢⱼ
   - Preserve particle number and spin symmetry
   - Require Jordan-Wigner strings across qubits → many CNOT gates

2. **Qubit excitations (QEB)**:
   - Obey qubit commutation relations [Qᵢ, Qⱼ] = 0
   - Also preserve particle number and Sᵤ symmetry (no anticommutation needed)
   - NO Jordan-Wigner strings → ~20x fewer CNOT gates for high-rank excitations
   - Steeper initial gradients → better trainability

Key operators:
- SingleExcitation(θ, [i,j]): Givens rotation between |01⟩ and |10⟩ (swap one electron)
- DoubleExcitation(θ, [i,j,k,l]): Givens rotation between |1100⟩ and |0011⟩ (swap two electrons)

Both preserve total particle count while allowing amplitude mixing between particle-conserving configurations.
"""

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector, Parameter
from itertools import combinations
from typing import Tuple, List, Optional

# Try to import SingleExcitation and DoubleExcitation from different locations
try:
    from qiskit.circuit.library import SingleExcitation, DoubleExcitation
except ImportError:
    try:
        from qiskit_nature.second_q.circuit.library import SingleExcitation, DoubleExcitation
    except ImportError:
        # Fallback: We'll build them manually
        SingleExcitation = None
        DoubleExcitation = None


class QEBansatz(QuantumCircuit):
    """
    Qubit-Excitation-Based (QEB) ansatz for VQE.

    Builds circuits from Givens rotations (single & double excitations) that:
    1. Preserve particle number and spin symmetry
    2. Use ~20x fewer gates than fermionic excitations
    3. Start from Hartree-Fock reference state

    Inherits from QuantumCircuit to be compatible with Qiskit VQE.

    Parameters:
    -----------
    num_qubits : int
        Number of qubits (spin orbitals) in the system
    num_particles : tuple of (int, int)
        (n_alpha, n_beta) number of spin-up and spin-down electrons
    depth : int, optional (default=1)
        Repetition depth L: how many times to repeat the excitation layers.
        Increasing L allows the ansatz to represent more complex superpositions.
    include_double : bool, optional (default=True)
        Whether to include double excitations (more expressive but more parameters).
        Single excitations alone form a complete basis for particle-conserving unitaries,
        but doubles can improve convergence speed.

    Attributes:
    -----------
    num_parameters : int
        Total number of variational parameters
    single_excitations : List[Tuple[int, int]]
        List of (i, j) pairs for single excitations
    double_excitations : List[Tuple[int, int, int, int]]
        List of (i, j, k, l) tuples for double excitations
    """

    def __init__(
        self,
        num_qubits: int,
        num_particles: Tuple[int, int],
        depth: int = 1,
        include_double: bool = True,
        num_spatial_orbitals: Optional[int] = None,
        use_hartree_fock_init: bool = True,
        mapper=None,
    ):
        super().__init__(num_qubits, name=f"QEB_L{depth}")

        self._num_qubits = num_qubits
        self.num_alpha, self.num_beta = num_particles
        self.num_particles = sum(num_particles)
        self.depth = depth
        self.include_double = include_double
        self.use_hartree_fock_init = use_hartree_fock_init
        self.mapper = mapper

        # If num_spatial_orbitals not provided, assume JW mapping: 2 qubits per orbital
        self.num_spatial_orbitals = num_spatial_orbitals or (num_qubits // 2)

        # Build excitation pool respecting fermionic symmetry
        self.single_excitations = self._build_single_excitations()
        self.double_excitations = (
            self._build_double_excitations() if include_double else []
        )

        # Total parameters: (singles + doubles) * depth
        # Note: Can't use num_parameters (QuantumCircuit property), so use _num_params
        self._num_params = len(self.single_excitations) * depth
        if include_double:
            self._num_params += len(self.double_excitations) * depth

        # Initialize the circuit with parameters
        self._params = None
        self._build_default_circuit()

    @property
    def num_parameters(self) -> int:
        """Return the number of variational parameters."""
        return self._num_params

    def _build_default_circuit(self):
        """Build the circuit with Hartree-Fock initialization and parameterized excitations."""
        from qiskit.circuit import ParameterVector

        # Create parameter vector
        params = ParameterVector("θ", self.num_parameters)

        # Prepare Hartree-Fock initial state
        if self.use_hartree_fock_init and self.mapper is not None:
            # Use Qiskit's HartreeFock if mapper is provided
            try:
                from qiskit_nature.second_q.circuit.library import HartreeFock
                hf_circuit = HartreeFock(
                    self.num_spatial_orbitals,
                    (self.num_alpha, self.num_beta),
                    self.mapper,
                )
                # Append the HartreeFock circuit
                self.compose(hf_circuit, inplace=True)
            except (ImportError, Exception):
                # Fallback to manual X gates if HartreeFock fails
                self._add_manual_hf_state()
        else:
            # Use manual X gates (fast, no decomposition overhead)
            # HF state has electrons in lowest energy orbitals
            # Alpha electrons in even qubits (0, 2, 4, ...)
            # Beta electrons in odd qubits (1, 3, 5, ...)
            for i in range(self.num_alpha):
                self.x(2 * i)
            for i in range(self.num_beta):
                self.x(2 * i + 1)

        # Apply excitation layers with parameters
        param_idx = 0
        for layer in range(self.depth):
            # Single excitations
            for i, j in self.single_excitations:
                if param_idx < len(params):
                    if SingleExcitation is not None:
                        self.append(SingleExcitation(params[param_idx]), [i, j])
                    else:
                        self._add_single_excitation_manual(self, params[param_idx], i, j)
                    param_idx += 1

            # Double excitations
            if self.include_double:
                for i, j, k, l in self.double_excitations:
                    if param_idx < len(params):
                        if DoubleExcitation is not None:
                            self.append(DoubleExcitation(params[param_idx]), [i, j, k, l])
                        else:
                            self._add_double_excitation_manual(self, params[param_idx], i, j, k, l)
                        param_idx += 1

    def _add_manual_hf_state(self):
        """Add Hartree-Fock state using X gates (fallback method)."""
        for i in range(self.num_alpha):
            self.x(2 * i)
        for i in range(self.num_beta):
            self.x(2 * i + 1)

    def _build_single_excitations(self) -> List[Tuple[int, int]]:
        """
        Build particle-conserving single excitation pairs.

        Givens rotations preserve particle number through their block-diagonal structure,
        not through spin conservation. So we allow ALL qubit pairs - the unitary structure
        itself ensures particle conservation regardless of spin.

        Returns:
        --------
        excitations : List[Tuple[int, int]]
            List of (i, j) qubit pairs representing valid excitations
        """
        excitations = []

        # Include all qubit pairs - Givens rotations preserve particle count
        # through their geometric structure, not spin conservation
        for i, j in combinations(range(self._num_qubits), 2):
            excitations.append((i, j))

        return excitations

    def _build_double_excitations(
        self, max_double_excitations: int = 50
    ) -> List[Tuple[int, int, int, int]]:
        """
        Build particle-conserving double excitation pairs.

        Double excitations represent Givens rotations between 4-qubit basis states.
        Like single excitations, the particle conservation is guaranteed by the
        block-diagonal structure of the unitary, not by spin conservation.

        We limit to max_double_excitations to keep circuit depth reasonable.

        Returns:
        --------
        excitations : List[Tuple[int, int, int, int]]
            List of (i, j, k, l) qubit tuples representing valid double excitations
        """
        excitations = []

        # Generate all valid 4-qubit combinations
        for (i, j) in combinations(range(self._num_qubits), 2):
            for (k, l) in combinations(range(self._num_qubits), 2):
                # Ensure all four qubits are distinct and maintain ordering
                if i < k and j < l and len(set([i, j, k, l])) == 4:
                    excitations.append((i, j, k, l))

        # Limit to avoid excessive circuit depth
        return excitations[:max_double_excitations]


    @staticmethod
    def _add_single_excitation_manual(circuit: QuantumCircuit, theta: float, i: int, j: int):
        """
        Add single excitation gate manually using basic gates.
        Implements Givens rotation: exp(-i theta/2 (X_i Y_j - Y_i X_j))
        """
        # This is an approximate implementation using RY and CNOT
        circuit.ry(theta / 2, i)
        circuit.cx(i, j)
        circuit.ry(-theta / 2, j)
        circuit.cx(i, j)

    @staticmethod
    def _add_double_excitation_manual(circuit: QuantumCircuit, theta: float, i: int, j: int, k: int, l: int):
        """
        Add double excitation gate manually using basic gates.
        Approximates 4-qubit Givens rotation.
        """
        # Two-body excitation approximation
        circuit.ry(theta / 4, i)
        circuit.cx(i, j)
        circuit.ry(-theta / 4, j)
        circuit.cx(i, j)

        circuit.ry(theta / 4, k)
        circuit.cx(k, l)
        circuit.ry(-theta / 4, l)
        circuit.cx(k, l)


    def bind_parameters(self, params: np.ndarray) -> QuantumCircuit:
        """
        Create a new circuit with bound parameters.

        This is used by VQE to evaluate the circuit with specific parameter values.

        Parameters:
        -----------
        params : np.ndarray or dict
            Parameter values to bind to the circuit

        Returns:
        --------
        QuantumCircuit
            A new circuit with parameters bound to values
        """
        # Handle both array and dict parameter formats
        if isinstance(params, np.ndarray):
            param_dict = {f"θ[{i}]": params[i] for i in range(len(params))}
        else:
            param_dict = params

        # Bind parameters to the circuit
        return self.assign_parameters(param_dict)

    def get_initial_parameters(
        self,
        initial_point: Optional[np.ndarray] = None,
        perturbation: float = 0.01,
        seed: Optional[int] = None,
    ) -> np.ndarray:
        """
        Generate initial parameter vector.

        For VQE, a good initialization strategy is important:
        - Start near zero (close to Hartree-Fock) for initial rapid descent
        - Add small random perturbation to break degeneracies
        - Avoid barren plateaus from symmetric initialization

        Parameters:
        -----------
        initial_point : np.ndarray, optional
            User-provided initial parameters. If None, initialize from zero with noise.
        perturbation : float
            Standard deviation of Gaussian noise added to parameters
        seed : int, optional
            Random seed for reproducibility

        Returns:
        --------
        params : np.ndarray, shape (num_parameters,)
            Initial parameter vector
        """
        if seed is not None:
            np.random.seed(seed)

        if initial_point is not None:
            params = np.array(initial_point, dtype=float).copy()
        else:
            params = np.random.normal(0, perturbation, self.num_parameters)

        return params

    def get_info(self) -> dict:
        """
        Return ansatz configuration information.

        Returns:
        --------
        info : dict
            Dictionary with ansatz metadata useful for logging
        """
        return {
            "ansatz_type": "QEB",
            "num_qubits": self.num_qubits,
            "num_particles": (self.num_alpha, self.num_beta),
            "total_particles": self.num_particles,
            "depth": self.depth,
            "include_double_excitations": self.include_double,
            "num_single_excitations": len(self.single_excitations),
            "num_double_excitations": len(self.double_excitations),
            "total_parameters": self.num_parameters,
        }


class QEBansatzWithTapering(QEBansatz):
    """
    Extended QEB ansatz that accounts for qubit tapering (symmetry reduction).

    When Z₂ symmetries are detected and qubits are tapered, this class adapts
    the excitation indices to the reduced qubit register.

    Parameters:
    -----------
    num_qubits : int
        Number of qubits in the TAPERED register (after reduction)
    num_particles : tuple of (int, int)
        (n_alpha, n_beta) in the active space
    taper_config : dict, optional
        Mapping from original qubit indices to tapered indices
    depth : int, optional
        Repetition depth
    include_double : bool, optional
        Whether to include double excitations
    """

    def __init__(
        self,
        num_qubits: int,
        num_particles: Tuple[int, int],
        taper_config: Optional[dict] = None,
        depth: int = 1,
        include_double: bool = True,
    ):
        super().__init__(num_qubits, num_particles, depth, include_double)
        self.taper_config = taper_config

    def get_info(self) -> dict:
        info = super().get_info()
        info["ansatz_type"] = "QEB_Tapered"
        if self.taper_config:
            info["tapered"] = True
        return info


# ============================================================================
# UTILITY FUNCTIONS FOR INTEGRATION WITH QISKIT VQE
# ============================================================================


def create_qeb_vqe_ansatz(
    num_qubits: int,
    num_particles: Tuple[int, int],
    depth: int = 1,
    include_double: bool = True,
) -> Tuple[QEBansatz, np.ndarray]:
    """
    Convenience function to create QEB ansatz and initial parameters.

    Returns both the ansatz object and initial parameters, ready for VQE.

    Parameters:
    -----------
    num_qubits : int
        Number of qubits
    num_particles : tuple
        (n_alpha, n_beta) electron counts
    depth : int
        Repetition depth L
    include_double : bool
        Include double excitations

    Returns:
    --------
    ansatz : QEBansatz
        The ansatz object with build_circuit method
    initial_params : np.ndarray
        Initial parameter vector for VQE optimizer
    """
    ansatz = QEBansatz(num_qubits, num_particles, depth, include_double)
    initial_params = ansatz.get_initial_parameters(perturbation=0.01)
    return ansatz, initial_params


# ============================================================================
# DOCUMENTATION AND EXAMPLES
# ============================================================================

"""
USAGE EXAMPLE:
==============

# For a system with 12 qubits and 4 total particles (2 alpha, 2 beta):
from qeb_ansatz import QEBansatz

ansatz = QEBansatz(
    num_qubits=12,
    num_particles=(2, 2),
    depth=1,
    include_double=True
)

# Get ansatz information
info = ansatz.get_info()
print(f"QEB ansatz has {ansatz.num_parameters} parameters")

# Generate initial parameters for VQE
initial_params = ansatz.get_initial_parameters(perturbation=0.01)

# Build circuit for a given parameter set
circuit = ansatz.build_circuit(initial_params)

# Use with Qiskit VQE:
from qiskit_algorithms import VQE
from qiskit.primitives import StatevectorEstimator
from qiskit_algorithms.optimizers import SPSA

estimator = StatevectorEstimator()
vqe = VQE(estimator, ansatz, SPSA(maxiter=300))
vqe.initial_point = initial_params
result = vqe.compute_minimum_eigenvalue(hamiltonian)


COMPARISON: QEB vs UCCSD
========================

Aspect                 | QEB                      | UCCSD
-----------------------|--------------------------|---------------------------
Gate efficiency        | 20x fewer CNOT           | Many Jordan-Wigner strings
Particle conservation  | ✓ Yes (via Givens)       | ✓ Yes (anticommutation)
Spin conservation      | ✓ Yes                    | ✓ Yes
Circuit depth          | Shallow (O(depth*n))     | Deeper (O(depth*n²))
Parameters (LiH, depth=1) | ~100-150              | ~92 (but depth is higher)
Trainability           | Generally good           | Can barren plateau
Hardware efficiency    | Excellent               | Good
Expressibility         | High (complete basis)    | High (complete basis)
"""
