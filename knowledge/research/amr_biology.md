# Antimicrobial Resistance Biology

## Competition Species Overview

| Species | Sample Count | Key Characteristics |
|---------|--------------|---------------------|
| E. coli (0) | 559 (17%) | Gram-negative, gut commensal, common MDR |
| K. pneumoniae (1) | 939 (28%) | Gram-negative, hospital-acquired, carbapenemase producer |
| P. mirabilis (2) | 415 (12%) | Gram-negative, urinary tract, swarming motility |
| P. aeruginosa (3) | 1447 (43%) | Gram-negative, opportunistic, high intrinsic resistance |

## Resistance Mechanisms by Species

### E. coli
- **ESBLs** (CTX-M, TEM, SHV): Extended-spectrum beta-lactamases
- **Porin loss** (OmpC, OmpF): Reduced antibiotic entry
- **AmpC overexpression**: Cephalosporin resistance
- **MALDI detectability**: Good (AUC 0.74-0.88)

### K. pneumoniae
- **ESBLs**: Common CTX-M-15
- **Carbapenemases**: KPC (Class A), NDM/VIM/IMP (Class B), OXA-48 (Class D)
- **Porin loss**: OmpK35/OmpK36
- **MALDI detectability**: Moderate (AUC 0.74-0.83)

### P. mirabilis
- **Intrinsic resistance**: To tetracyclines, polymyxins
- **ESBLs**: Acquired resistance
- **Limited research** on MALDI-TOF detection

### P. aeruginosa
- **High intrinsic resistance**: Efflux pumps, low permeability
- **AmpC**: Inducible cephalosporinase
- **Carbapenemases**: VIM, IMP (metallo-β-lactamases)
- **MALDI detectability**: Good (AUC up to 0.87)

## Antibiotic Classes in Competition

| Antibiotic | Class | Primary Target | Notes |
|------------|-------|----------------|-------|
| Ampicillin | Aminopenicillin | Cell wall | P. aeruginosa intrinsically resistant |
| Amox/Clav | Penicillin + inhibitor | Cell wall | Beta-lactamase inhibitor combination |
| Cefotaxime | 3rd gen cephalosporin | Cell wall | ESBL target |
| Cefuroxime | 2nd gen cephalosporin | Cell wall | Less active vs Gram-neg |
| Ertapenem | Carbapenem | Cell wall | NOT active vs P. aeruginosa |
| Imipenem | Carbapenem | Cell wall | Broad spectrum |
| Levofloxacin | Fluoroquinolone | DNA gyrase | Cross-resistance with ciprofloxacin |
| Ciprofloxacin | Fluoroquinolone | DNA gyrase | Cross-resistance with levofloxacin |

## Intrinsic Resistance Patterns

### P. aeruginosa (Species 3) - CRITICAL INSIGHT
From our data analysis, P. aeruginosa shows **100% resistance** to:
- Ampicillin
- Amoxicillin/Clavulanic acid (limited data: n=54)
- Ertapenem
- Cefotaxime
- Cefuroxime

**Implication**: For these antibiotics, species_id=3 → predict resistance=1 with ~100% confidence.

### P. mirabilis (Species 2)
- **97.2% resistant** to Imipenem (intrinsic, unusual carbapenem resistance)
- Only 0.2% resistant to Ertapenem

## Correlation Patterns (from EDA)

Strong correlations indicate shared resistance mechanisms:
- **Levofloxacin ↔ Ciprofloxacin** (r=0.92): Same target (DNA gyrase)
- **Imipenem ↔ Ertapenem** (r=0.77): Both carbapenems
- **Ertapenem ↔ Cefotaxime** (r=0.81): Related resistance mechanisms
- **Cefotaxime ↔ Cefuroxime** (r=0.66): Both cephalosporins

## Modeling Implications

1. **Species-specific models** may outperform global models
2. **P. aeruginosa** can be near-deterministic for 5/8 antibiotics
3. **Fluoroquinolones** (Levo/Cipro) should be modeled together
4. **Carbapenems** (Imipenem/Ertapenem) share mechanisms but differ by species
5. **Cephalosporins** (Cefotaxime/Cefuroxime) correlate moderately
