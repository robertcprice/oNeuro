"""Gene expression pipeline: DNA → RNA → Protein → channels/receptors/enzymes.

Long-term plasticity through receptor trafficking:
- Upregulate GRIN1 → more NMDA receptors → stronger synapse (LTP)
- Downregulate GRIA1 → fewer AMPA receptors → weaker synapse (LTD)

Enhanced with:
- Transcription factors (CREB, c-Fos, Arc, Zif268, NF-κB)
- Immediate early genes with realistic temporal dynamics
- Epigenetic layer (histone acetylation/methylation, DNA methylation)
- CREB phosphorylation bridge from CaMKII/PKA (second messengers)

Uses nQPU's DNA.transcribe() → RNA.translate() → QuantumProteinFolder
when available; otherwise uses simplified rate models.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

from oneuro.molecular.backend import HAS_NQPU, get_nqpu_chem
from oneuro.molecular.receptors import ReceptorType


class GeneID(Enum):
    """Genes that control receptor/channel/enzyme expression."""

    # Glutamate receptor subunits
    GRIN1 = "GRIN1"  # NMDA NR1 subunit
    GRIN2A = "GRIN2A"  # NMDA NR2A subunit
    GRIA1 = "GRIA1"  # AMPA GluA1 subunit
    GRIA2 = "GRIA2"  # AMPA GluA2 subunit

    # GABA receptor subunits
    GABRA1 = "GABRA1"  # GABA-A alpha1

    # Ion channels
    SCN1A = "SCN1A"  # Na_v 1.1
    KCNA1 = "KCNA1"  # K_v 1.1
    CACNA1A = "CACNA1A"  # Ca_v 2.1

    # Cholinergic
    CHRNA4 = "CHRNA4"  # nAChR alpha4

    # Enzymes
    ACHE = "ACHE"  # Acetylcholinesterase
    GAD1 = "GAD1"  # Glutamic acid decarboxylase (GABA synthesis)

    # Neurotrophic
    BDNF = "BDNF"  # Brain-derived neurotrophic factor

    # Immediate early genes (IEGs)
    FOS = "FOS"  # c-Fos proto-oncogene, peaks 30-60 min after stimulus
    ARC = "ARC"  # Activity-regulated cytoskeleton-associated, mRNA → dendrites
    ZIF268 = "ZIF268"  # Zif268/Egr1, rapid transcription factor
    NFKB = "NFKB"  # NF-κB, inflammation and plasticity


class TranscriptionFactorType(Enum):
    """Major transcription factors in neural plasticity."""

    CREB = "CREB"  # cAMP response element binding protein
    CFOS = "c-Fos"  # Immediate early gene product
    ARC_TF = "Arc"  # Activity-regulated cytoskeleton protein
    ZIF268_TF = "Zif268"  # Early growth response 1
    NFKB_TF = "NF-kB"  # Nuclear factor kappa-light-chain-enhancer


# Gene → what it produces
_GENE_PRODUCTS = {
    GeneID.GRIN1: {"product": "NMDA_receptor", "receptor_type": ReceptorType.NMDA},
    GeneID.GRIN2A: {"product": "NMDA_receptor", "receptor_type": ReceptorType.NMDA},
    GeneID.GRIA1: {"product": "AMPA_receptor", "receptor_type": ReceptorType.AMPA},
    GeneID.GRIA2: {"product": "AMPA_receptor", "receptor_type": ReceptorType.AMPA},
    GeneID.GABRA1: {"product": "GABA_A_receptor", "receptor_type": ReceptorType.GABA_A},
    GeneID.CHRNA4: {"product": "nAChR_receptor", "receptor_type": ReceptorType.nAChR},
    GeneID.SCN1A: {"product": "Na_channel"},
    GeneID.KCNA1: {"product": "K_channel"},
    GeneID.CACNA1A: {"product": "Ca_channel"},
    GeneID.ACHE: {"product": "AChE_enzyme"},
    GeneID.GAD1: {"product": "GAD_enzyme"},
    GeneID.BDNF: {"product": "neurotrophin"},
    GeneID.FOS: {"product": "transcription_factor", "tf_type": TranscriptionFactorType.CFOS},
    GeneID.ARC: {"product": "cytoskeleton_regulator", "tf_type": TranscriptionFactorType.ARC_TF},
    GeneID.ZIF268: {"product": "transcription_factor", "tf_type": TranscriptionFactorType.ZIF268_TF},
    GeneID.NFKB: {"product": "transcription_factor", "tf_type": TranscriptionFactorType.NFKB_TF},
}


@dataclass
class TranscriptionFactor:
    """A transcription factor that regulates downstream gene expression."""

    tf_type: TranscriptionFactorType
    active_level: float = 0.0  # [0, 1] activation/phosphorylation state
    nuclear_level: float = 0.0  # [0, 1] nuclear translocation

    # Temporal dynamics (ms)
    activation_tau: float = 5000.0  # Rise time to peak
    deactivation_tau: float = 30000.0  # Decay time
    nuclear_translocation_tau: float = 10000.0  # Time to enter nucleus

    def activate(self, signal: float) -> None:
        """Activate (phosphorylate) the transcription factor."""
        self.active_level = min(1.0, self.active_level + signal)

    def step(self, dt: float) -> None:
        """Update TF dynamics."""
        # Nuclear translocation follows activation
        target_nuclear = self.active_level
        self.nuclear_level += dt * (target_nuclear - self.nuclear_level) / self.nuclear_translocation_tau
        self.nuclear_level = max(0.0, min(1.0, self.nuclear_level))

        # Deactivation (dephosphorylation)
        self.active_level -= dt * self.active_level / self.deactivation_tau
        self.active_level = max(0.0, self.active_level)

    @property
    def transcription_drive(self) -> float:
        """How strongly this TF drives gene expression [0, 1]."""
        return self.nuclear_level


@dataclass
class EpigeneticState:
    """Epigenetic modifications that gate gene expression.

    Histone acetylation opens chromatin → increases transcription.
    DNA methylation at CpG sites → silences genes.
    """

    # Histone modifications
    acetylation: float = 0.5  # [0, 1], higher = more open chromatin, more expression
    methylation_h3k4: float = 0.3  # Activating mark
    methylation_h3k27: float = 0.2  # Repressive mark (Polycomb)

    # DNA methylation
    cpg_methylation: float = 0.3  # [0, 1], higher = more silenced

    # Enzyme activities
    hat_activity: float = 0.5  # Histone acetyltransferase (opens chromatin)
    hdac_activity: float = 0.5  # Histone deacetylase (closes chromatin)
    dnmt_activity: float = 0.3  # DNA methyltransferase (adds CpG methylation)
    tet_activity: float = 0.3  # TET enzyme (removes CpG methylation)

    def step(self, dt: float, neural_activity: float = 0.0) -> None:
        """Update epigenetic state.

        High neural activity → HAT↑, HDAC↓ → acetylation↑ → open chromatin.
        Low activity → HDAC↑, DNMT↑ → closed/methylated → silenced.
        """
        # Activity modulates enzyme balance
        if neural_activity > 0.5:
            activity_signal = min(1.0, (neural_activity - 0.5) * 2.0)
            self.hat_activity = min(1.0, self.hat_activity + 0.0001 * activity_signal * dt)
            self.hdac_activity = max(0.1, self.hdac_activity - 0.00005 * activity_signal * dt)
        else:
            self.hat_activity = max(0.2, self.hat_activity - 0.00002 * dt)
            self.hdac_activity = min(0.8, self.hdac_activity + 0.00001 * dt)

        # Acetylation dynamics
        d_acetyl = (self.hat_activity - self.hdac_activity) * 0.0001 * dt
        self.acetylation = max(0.0, min(1.0, self.acetylation + d_acetyl))

        # DNA methylation dynamics (very slow)
        d_cpg = (self.dnmt_activity - self.tet_activity) * 0.00001 * dt
        self.cpg_methylation = max(0.0, min(1.0, self.cpg_methylation + d_cpg))

    @property
    def expression_multiplier(self) -> float:
        """How much epigenetic state modulates transcription [0.1, 2.0]."""
        # Open chromatin (high acetylation, low CpG methylation) → high expression
        openness = self.acetylation * (1.0 - self.cpg_methylation * 0.8)
        return max(0.1, min(2.0, 0.5 + openness * 1.5))


@dataclass
class GeneState:
    """Expression state of a single gene."""

    gene_id: GeneID
    expression_level: float = 1.0  # Baseline = 1.0
    mrna_level: float = 0.0
    protein_level: float = 0.0

    # Transcription dynamics
    transcription_rate: float = 0.01  # mRNA produced per ms
    mrna_decay_rate: float = 0.001  # mRNA degraded per ms
    translation_rate: float = 0.005  # Protein produced per ms per mRNA
    protein_decay_rate: float = 0.0001  # Protein degraded per ms

    # IEG-specific: faster kinetics for immediate early genes
    is_ieg: bool = field(init=False, default=False)

    # Regulation
    _activity_integral: float = field(init=False, default=0.0)
    _regulation_signal: float = field(init=False, default=0.0)

    def __post_init__(self):
        # IEGs have faster kinetics
        if self.gene_id in (GeneID.FOS, GeneID.ARC, GeneID.ZIF268):
            self.is_ieg = True
            self.transcription_rate = 0.05  # 5x faster than late genes
            self.mrna_decay_rate = 0.005  # Faster turnover
            self.translation_rate = 0.02
            self.protein_decay_rate = 0.001  # Short-lived proteins

    def update(self, dt: float, epigenetic_mult: float = 1.0,
               tf_drive: float = 0.0) -> None:
        """Update gene expression for one timestep.

        mRNA and protein levels follow first-order kinetics:
          d[mRNA]/dt = transcription_rate * expression_level * epigenetic * tf_drive - decay
          d[protein]/dt = translation_rate * [mRNA] - protein_decay_rate * [protein]

        Args:
            dt: Timestep in ms.
            epigenetic_mult: Epigenetic state multiplier [0.1, 2.0].
            tf_drive: Additional transcription factor drive [0, 1].
        """
        # Effective transcription combines basal + TF-driven + epigenetic
        effective_transcription = (
            self.transcription_rate * self.expression_level * epigenetic_mult
            + self.transcription_rate * tf_drive * 0.5  # TF adds up to 50% boost
        )

        # mRNA dynamics
        d_mrna = effective_transcription - self.mrna_decay_rate * self.mrna_level
        self.mrna_level += d_mrna * dt
        self.mrna_level = max(0.0, self.mrna_level)

        # Protein dynamics
        d_protein = (
            self.translation_rate * self.mrna_level
            - self.protein_decay_rate * self.protein_level
        )
        self.protein_level += d_protein * dt
        self.protein_level = max(0.0, self.protein_level)

    def upregulate(self, amount: float) -> None:
        """Increase expression level (e.g., from LTP signal)."""
        self.expression_level = min(5.0, self.expression_level + amount)

    def downregulate(self, amount: float) -> None:
        """Decrease expression level (e.g., from LTD signal)."""
        self.expression_level = max(0.1, self.expression_level - amount)


@dataclass
class GeneExpressionPipeline:
    """Manages gene expression for a neuron.

    Translates activity signals into receptor/channel/enzyme changes.
    This is the molecular basis of long-term plasticity.

    Enhanced with transcription factors, epigenetics, and IEG cascades.
    """

    genes: Dict[GeneID, GeneState] = field(default_factory=dict)

    # Transcription factors
    transcription_factors: Dict[TranscriptionFactorType, TranscriptionFactor] = field(
        default_factory=dict
    )

    # Epigenetic state (per-gene would be more accurate but too expensive)
    epigenetics: EpigeneticState = field(default_factory=EpigeneticState)

    # CREB phosphorylation — the master plasticity switch
    creb_phosphorylation: float = 0.0  # [0, 1], set by CaMKII/PKA from second messengers

    # Pending receptor changes (accumulated until threshold)
    _pending_receptor_insertions: Dict[ReceptorType, float] = field(
        default_factory=lambda: {}
    )
    _pending_receptor_removals: Dict[ReceptorType, float] = field(
        default_factory=lambda: {}
    )

    # Thresholds for receptor trafficking
    _insertion_threshold: float = 10.0  # Protein level to insert one receptor
    _removal_threshold: float = 0.5  # Protein level below which removal occurs

    # Activity tracking for epigenetics
    _recent_activity: float = field(init=False, default=0.0)

    def __post_init__(self):
        if not self.genes:
            for gene_id in GeneID:
                self.genes[gene_id] = GeneState(gene_id=gene_id)

        if not self.transcription_factors:
            # Default TF set
            self.transcription_factors = {
                TranscriptionFactorType.CREB: TranscriptionFactor(
                    tf_type=TranscriptionFactorType.CREB,
                    activation_tau=5000.0,
                    deactivation_tau=60000.0,
                    nuclear_translocation_tau=15000.0,
                ),
                TranscriptionFactorType.CFOS: TranscriptionFactor(
                    tf_type=TranscriptionFactorType.CFOS,
                    activation_tau=10000.0,  # Peaks 30-60 min
                    deactivation_tau=120000.0,  # ~2h half-life
                    nuclear_translocation_tau=5000.0,
                ),
                TranscriptionFactorType.ARC_TF: TranscriptionFactor(
                    tf_type=TranscriptionFactorType.ARC_TF,
                    activation_tau=15000.0,
                    deactivation_tau=60000.0,
                    nuclear_translocation_tau=20000.0,  # Arc mRNA travels to dendrites
                ),
                TranscriptionFactorType.ZIF268_TF: TranscriptionFactor(
                    tf_type=TranscriptionFactorType.ZIF268_TF,
                    activation_tau=3000.0,  # Very rapid
                    deactivation_tau=30000.0,
                    nuclear_translocation_tau=5000.0,
                ),
                TranscriptionFactorType.NFKB_TF: TranscriptionFactor(
                    tf_type=TranscriptionFactorType.NFKB_TF,
                    activation_tau=10000.0,
                    deactivation_tau=60000.0,
                    nuclear_translocation_tau=10000.0,
                ),
            }

    def update(self, dt: float, neural_activity: float = 0.0) -> None:
        """Advance all gene expression by dt ms.

        Args:
            dt: Timestep in ms.
            neural_activity: Normalized activity [0, 1] for epigenetic modulation.
        """
        # Update CREB from external phosphorylation signal
        creb = self.transcription_factors.get(TranscriptionFactorType.CREB)
        if creb is not None:
            creb.active_level = self.creb_phosphorylation

        # Update all transcription factors
        for tf in self.transcription_factors.values():
            tf.step(dt)

        # Update epigenetics
        self._recent_activity = 0.99 * self._recent_activity + 0.01 * neural_activity
        self.epigenetics.step(dt, self._recent_activity)
        epi_mult = self.epigenetics.expression_multiplier

        # Compute TF drive for each gene
        for gene_id, gene_state in self.genes.items():
            tf_drive = self._compute_tf_drive(gene_id)
            gene_state.update(dt, epigenetic_mult=epi_mult, tf_drive=tf_drive)

    def _compute_tf_drive(self, gene_id: GeneID) -> float:
        """Compute transcription factor drive for a specific gene."""
        drive = 0.0

        # CREB drives BDNF, GRIA1 (LTP genes), c-Fos, Arc
        creb = self.transcription_factors.get(TranscriptionFactorType.CREB)
        if creb is not None:
            if gene_id in (GeneID.BDNF, GeneID.GRIA1, GeneID.FOS, GeneID.ARC):
                drive += creb.transcription_drive * 0.8
            elif gene_id in (GeneID.GRIN1, GeneID.GRIN2A):
                drive += creb.transcription_drive * 0.4

        # c-Fos drives late response genes (BDNF, receptor subunits)
        cfos = self.transcription_factors.get(TranscriptionFactorType.CFOS)
        if cfos is not None:
            if gene_id in (GeneID.BDNF, GeneID.GRIA1, GeneID.GRIN1):
                drive += cfos.transcription_drive * 0.5

        # Arc regulates AMPA trafficking genes
        arc = self.transcription_factors.get(TranscriptionFactorType.ARC_TF)
        if arc is not None:
            if gene_id in (GeneID.GRIA1, GeneID.GRIA2):
                drive += arc.transcription_drive * 0.6

        # NF-κB drives inflammatory/neuroprotective genes
        nfkb = self.transcription_factors.get(TranscriptionFactorType.NFKB_TF)
        if nfkb is not None:
            if gene_id == GeneID.BDNF:
                drive += nfkb.transcription_drive * 0.3

        return min(1.0, drive)

    def signal_ltp(self, ca_level_nM: float) -> None:
        """High calcium triggers LTP gene expression cascade.

        Full cascade: Ca²⁺ → CaMKII → CREB-P → IEGs (c-Fos 30min, Arc) → late genes (BDNF 2h)
        """
        if ca_level_nM > 500.0:
            strength = min(1.0, (ca_level_nM - 500.0) / 2000.0)

            # Direct gene upregulation (fast, existing mechanism)
            if GeneID.GRIA1 in self.genes:
                self.genes[GeneID.GRIA1].upregulate(0.1 * strength)
            if GeneID.GRIN1 in self.genes:
                self.genes[GeneID.GRIN1].upregulate(0.05 * strength)
            if GeneID.BDNF in self.genes:
                self.genes[GeneID.BDNF].upregulate(0.1 * strength)

            # Activate IEG transcription factors
            for tf_type in (TranscriptionFactorType.CFOS,
                           TranscriptionFactorType.ARC_TF,
                           TranscriptionFactorType.ZIF268_TF):
                tf = self.transcription_factors.get(tf_type)
                if tf is not None:
                    tf.activate(0.2 * strength)

            # Upregulate IEG genes directly
            if GeneID.FOS in self.genes:
                self.genes[GeneID.FOS].upregulate(0.3 * strength)
            if GeneID.ARC in self.genes:
                self.genes[GeneID.ARC].upregulate(0.25 * strength)
            if GeneID.ZIF268 in self.genes:
                self.genes[GeneID.ZIF268].upregulate(0.35 * strength)

    def signal_ltd(self, ca_level_nM: float) -> None:
        """Moderate calcium triggers LTD gene expression.

        LTD: downregulate AMPA (GRIA1).
        Range: 200 nM < Ca2+ < 500 nM.
        """
        if 200.0 < ca_level_nM < 500.0:
            strength = (ca_level_nM - 200.0) / 300.0
            if GeneID.GRIA1 in self.genes:
                self.genes[GeneID.GRIA1].downregulate(0.05 * strength)

    def signal_creb_phosphorylation(self, p_creb: float) -> None:
        """Set CREB phosphorylation level from CaMKII/PKA (second messengers).

        This is the bridge between second messenger cascades and gene expression.
        """
        self.creb_phosphorylation = max(0.0, min(1.0, p_creb))

    def signal_inflammation(self, damage_signal: float) -> None:
        """Inflammatory signals activate NF-κB pathway."""
        nfkb = self.transcription_factors.get(TranscriptionFactorType.NFKB_TF)
        if nfkb is not None:
            nfkb.activate(min(1.0, damage_signal))

    def get_receptor_changes(self) -> Dict[ReceptorType, int]:
        """Check if protein levels warrant receptor insertion/removal.

        Returns dict of ReceptorType → count change (positive = insert, negative = remove).
        """
        changes: Dict[ReceptorType, int] = {}

        for gene_id, gene_state in self.genes.items():
            product_info = _GENE_PRODUCTS.get(gene_id, {})
            receptor_type = product_info.get("receptor_type")
            if receptor_type is None:
                continue

            # Track accumulated protein
            if receptor_type not in self._pending_receptor_insertions:
                self._pending_receptor_insertions[receptor_type] = 0.0

            self._pending_receptor_insertions[receptor_type] += gene_state.protein_level * 0.001

            # Check if enough protein accumulated to insert a receptor
            if self._pending_receptor_insertions[receptor_type] >= self._insertion_threshold:
                count = int(self._pending_receptor_insertions[receptor_type] / self._insertion_threshold)
                changes[receptor_type] = changes.get(receptor_type, 0) + count
                self._pending_receptor_insertions[receptor_type] -= count * self._insertion_threshold

            # Check for removal (low expression)
            if gene_state.expression_level < self._removal_threshold:
                changes[receptor_type] = changes.get(receptor_type, 0) - 1

        return changes

    def get_protein_level(self, gene_id: GeneID) -> float:
        """Get current protein level for a gene."""
        if gene_id in self.genes:
            return self.genes[gene_id].protein_level
        return 0.0

    def get_tf_activity(self, tf_type: TranscriptionFactorType) -> float:
        """Get transcription factor activity level."""
        tf = self.transcription_factors.get(tf_type)
        return tf.active_level if tf is not None else 0.0

    def get_ieg_levels(self) -> Dict[str, float]:
        """Get immediate early gene protein levels for monitoring."""
        return {
            "c-Fos": self.get_protein_level(GeneID.FOS),
            "Arc": self.get_protein_level(GeneID.ARC),
            "Zif268": self.get_protein_level(GeneID.ZIF268),
            "BDNF": self.get_protein_level(GeneID.BDNF),
        }

    @classmethod
    def excitatory_neuron(cls) -> "GeneExpressionPipeline":
        """Gene expression profile for an excitatory (glutamatergic) neuron."""
        genes = {}
        for gene_id in [
            GeneID.GRIN1, GeneID.GRIN2A, GeneID.GRIA1, GeneID.GRIA2,
            GeneID.SCN1A, GeneID.KCNA1, GeneID.CACNA1A, GeneID.BDNF,
            GeneID.FOS, GeneID.ARC, GeneID.ZIF268,
        ]:
            genes[gene_id] = GeneState(gene_id=gene_id)
        return cls(genes=genes)

    @classmethod
    def inhibitory_neuron(cls) -> "GeneExpressionPipeline":
        """Gene expression profile for an inhibitory (GABAergic) neuron."""
        genes = {}
        for gene_id in [
            GeneID.GABRA1, GeneID.GAD1,
            GeneID.SCN1A, GeneID.KCNA1, GeneID.BDNF,
            GeneID.FOS, GeneID.ARC,
        ]:
            genes[gene_id] = GeneState(gene_id=gene_id)
        return cls(genes=genes)
