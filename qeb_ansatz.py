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

"""
QEB (Qubit Excitation-Based) Ansatz Implementation for VQE - FIXED VERSION

CRITICAL FIX: This version properly respects spin conservation when working
with fermionic Hamiltonians mapped via Jordan-Wigner or Parity mappings.

The original QEB paper uses pure qubit operations, but when working with
fermionically-mapped Hamiltonians, we must respect the fermionic structure:
- Even qubits = alpha spin orbitals
- Odd qubits = beta spin orbitals
- Excitations must preserve spin (alpha -> alpha, beta -> beta). <-- ok i did not do this before
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
        SingleExcitation = None
        DoubleExcitation = None


class QEBansatz(QuantumCircuit):
    """
    Qubit-Excitation-Based (QEB) ansatz for VQE - CORRECTED VERSION
    
    This version properly handles fermionic mappings by respecting spin conservation.
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

        self.num_spatial_orbitals = num_spatial_orbitals or (num_qubits // 2)

        # Build excitation pool respecting fermionic symmetry
        self.single_excitations = self._build_single_excitations()
        self.double_excitations = (
            self._build_double_excitations() if include_double else []
        )

        # Total parameters
        self._num_params = len(self.single_excitations) * depth
        if include_double:
            self._num_params += len(self.double_excitations) * depth

        self._params = None
        self._build_default_circuit()

    @property
    def num_parameters(self) -> int:
        """Return the number of variational parameters."""
        return self._num_params

    def _build_default_circuit(self):
        """Build the circuit with Hartree-Fock initialization and parameterized excitations."""
        from qiskit.circuit import ParameterVector

        params = ParameterVector("θ", self.num_parameters)

        # Hartree-Fock initial state
        if self.use_hartree_fock_init and self.mapper is not None:
            try:
                from qiskit_nature.second_q.circuit.library import HartreeFock
                hf_circuit = HartreeFock(
                    self.num_spatial_orbitals,
                    (self.num_alpha, self.num_beta),
                    self.mapper,
                )
                self.compose(hf_circuit, inplace=True)
            except (ImportError, Exception):
                self._add_manual_hf_state()
        else:
            # Manual HF state
            for i in range(self.num_alpha):
                self.x(2 * i)
            for i in range(self.num_beta):
                self.x(2 * i + 1)

        # Apply excitation layers
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
        """Add Hartree-Fock state using X gates."""
        for i in range(self.num_alpha):
            self.x(2 * i)
        for i in range(self.num_beta):
            self.x(2 * i + 1)

    def _build_single_excitations(self) -> List[Tuple[int, int]]:
        """
        Build spin-conserving single excitation pairs.
        
        CRITICAL FIX: When working with fermionic Hamiltonians mapped via
        Jordan-Wigner or Parity mapping, we MUST respect spin conservation:
        - Even qubits (0, 2, 4, ...) = alpha spin orbitals
        - Odd qubits (1, 3, 5, ...) = beta spin orbitals
        - Excitations preserve spin: alpha->alpha, beta->beta
        
        This is essential for the ansatz to work with fermionically-mapped operators.
        """
        excitations = []
        
        # Alpha spin excitations (even qubits only)
        alpha_qubits = [i for i in range(self._num_qubits) if i % 2 == 0]
        for i, j in combinations(alpha_qubits, 2):
            excitations.append((i, j))
        
        # Beta spin excitations (odd qubits only)
        beta_qubits = [i for i in range(self._num_qubits) if i % 2 == 1]
        for i, j in combinations(beta_qubits, 2):
            excitations.append((i, j))
        
        return excitations

    def _build_double_excitations(
        self, max_double_excitations: int = 70
    ) -> List[Tuple[int, int, int, int]]:
        """
        Build spin-conserving double excitation pairs.
        
        CRITICAL FIX: Double excitations must also respect spin conservation.
        Valid patterns:
        - (alpha, alpha) -> (alpha, alpha): (i_α, j_α, k_α, l_α)
        - (beta, beta) -> (beta, beta): (i_β, j_β, k_β, l_β)
        - (alpha, beta) -> (alpha, beta): (i_α, j_β, k_α, l_β)
        """
        excitations = []
        
        alpha_qubits = [i for i in range(self._num_qubits) if i % 2 == 0]
        beta_qubits = [i for i in range(self._num_qubits) if i % 2 == 1]
        
        # Alpha-alpha double excitations
        for (i, j, k, l) in combinations(alpha_qubits, 4):
            excitations.append((i, j, k, l))
        
        # Beta-beta double excitations
        for (i, j, k, l) in combinations(beta_qubits, 4):
            excitations.append((i, j, k, l))
        
        # Alpha-beta mixed double excitations
        for i, k in combinations(alpha_qubits, 2):
            for j, l in combinations(beta_qubits, 2):
                excitations.append((i, j, k, l))

        return excitations[:max_double_excitations]

    @staticmethod
    def _add_single_excitation_manual(circuit: QuantumCircuit, theta: float, i: int, j: int):
        """Add single excitation gate manually using basic gates."""
        circuit.ry(theta / 2, i)
        circuit.cx(i, j)
        circuit.ry(-theta / 2, j)
        circuit.cx(i, j)

    @staticmethod
    def _add_double_excitation_manual(circuit: QuantumCircuit, theta: float, i: int, j: int, k: int, l: int):
        """Add double excitation gate manually using basic gates."""
        circuit.ry(theta / 4, i)
        circuit.cx(i, j)
        circuit.ry(-theta / 4, j)
        circuit.cx(i, j)

        circuit.ry(theta / 4, k)
        circuit.cx(k, l)
        circuit.ry(-theta / 4, l)
        circuit.cx(k, l)

    def bind_parameters(self, params: np.ndarray) -> QuantumCircuit:
        """Bind parameters to the circuit."""
        if isinstance(params, np.ndarray):
            param_dict = {f"θ[{i}]": params[i] for i in range(len(params))}
        else:
            param_dict = params
        return self.assign_parameters(param_dict)

    def get_initial_parameters(
        self,
        initial_point: Optional[np.ndarray] = None,
        perturbation: float = 0.01,
        seed: Optional[int] = None,
    ) -> np.ndarray:
        """Generate initial parameter vector."""
        if seed is not None:
            np.random.seed(seed)

        if initial_point is not None:
            params = np.array(initial_point, dtype=float).copy()
        else:
            params = np.random.normal(0, perturbation, self.num_parameters)

        return params

    def get_info(self) -> dict:
        """Return ansatz configuration information."""
        return {
            "ansatz_type": "QEB_Fixed",
            "num_qubits": self.num_qubits,
            "num_particles": (self.num_alpha, self.num_beta),
            "total_particles": self.num_particles,
            "depth": self.depth,
            "include_double_excitations": self.include_double,
            "num_single_excitations": len(self.single_excitations),
            "num_double_excitations": len(self.double_excitations),
            "total_parameters": self.num_parameters,
        }
