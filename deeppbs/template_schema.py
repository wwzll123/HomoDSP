"""Named layout of the archived 92-channel interface representation (no NN imports)."""

AA_NAMES = 'ALA ARG ASN ASP CYS GLN GLU GLY HIS ILE LEU LYS MET PHE PRO SER THR TRP TYR VAL'.split()
AA_PROPERTIES = 'hydrophobic polar positive negative aromatic aliphatic small sulfur glycine proline'.split()
DNA_BASES = ['A', 'C', 'G', 'T', 'OTHER']
FEATURE_NAMES = (
    [f'aa_{x}' for x in AA_NAMES]
    + [f'aa_prop_{x}' for x in AA_PROPERTIES]
    + [f'closest_atom_{x}' for x in ['c', 'n', 'o', 's', 'p', 'other']]
    + [f'closest_atom_role_{x}' for x in ['backbone', 'sidechain', 'terminal', 'other']]
    + [f'dna_group_{x}' for x in ['phosphate', 'sugar', 'base']]
    + [f'dna_base_{x.lower()}' for x in DNA_BASES]
    + ['min_dist_all_over_10', 'min_dist_phosphate_over_10', 'min_dist_sugar_over_10',
       'min_dist_base_over_10', 'log_contacts_all', 'log_contacts_phosphate',
       'log_contacts_sugar', 'log_contacts_base', 'residue_atom_count_over_30', 'rank_norm']
    + ['local_dx_over_10', 'local_dy_over_10', 'local_dz_over_10']
    + ['local_unit_dx', 'local_unit_dy', 'local_unit_dz']
    + [f'dist_rbf_{i:02d}' for i in range(16)]
    + ['sidechain_lx', 'sidechain_ly', 'sidechain_lz', 'ca_to_dna_lx', 'ca_to_dna_ly',
       'ca_to_dna_lz', 'sidechain_length_over_10', 'has_sidechain_vector']
    + ['is_foldseek_aligned', 'aligned_query_pos_norm', 'aligned_template_pos_norm', 'aligned_same_aa']
)
assert len(FEATURE_NAMES) == 92
FEATURE_INDEX = {name: i for i, name in enumerate(FEATURE_NAMES)}
DNA_INDICES = [FEATURE_INDEX[f'dna_base_{b.lower()}'] for b in DNA_BASES]
POSITION_INDICES = [FEATURE_INDEX[f'local_d{axis}_over_10'] for axis in 'xyz']
SIDECHAIN_INDICES = [FEATURE_INDEX[f'sidechain_l{axis}'] for axis in 'xyz']
