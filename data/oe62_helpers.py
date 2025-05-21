'''
Helper functions to work with the 62k-dataset of molecular orbital energies
'''

# These packages are needed for xyz2ase
try:
    import StringIO as io
except ImportError:
    import io

try: 
    import ase.io
except:
        pass


subsets = ['PBE+vdW_vacuum',
           'PBE0_vacuum',
           'PBE0_vacuum_tzvp',
           'PBE0_vacuum_qzvp',
           'PBE0_water',
           'GOWO_at_PBE0_cbs',
           'GOWO_at_PBE0_tzvp',
           'GOWO_at_PBE0_qzvp']


fields_occupied = ['energies_occ_pbe',
                   'energies_occ_pbe0_vac_tier2',
                   'energies_occ_pbe0_vac_tzvp',
                   'energies_occ_pbe0_vac_qzvp',
                   'energies_occ_pbe0_water',
                   'cbs_occ_gw',
                   'energies_occ_gw_tzvp',
                   'energies_occ_gw_qzvp']
                   
fields_unoccupied = ['energies_unocc_pbe',
                    'energies_unocc_pbe0_vac_tier2',
                    'energies_unocc_pbe0_vac_tzvp',
                    'energies_unocc_pbe0_vac_qzvp',
                    'energies_unocc_pbe0_water',
                    'cbs_unocc_gw',
                    'energies_unocc_gw_tzvp',
                    'energies_unocc_gw_qzvp'
                   ]


def get_level(row, level_type='HOMO', subset='PBE+vdW_vacuum'):
    
    '''
    Extract the needed level for the given row from the list of molecular orbitals
    
    Parameters:
    ------------
    
    row: Row from the pandas dataframe. 
    
    level_type: string
    
        To access the occupied levels, the following notation is used:
        'HOMO' for highest occupied molecular orbital
        'HOMO-1' for one level below the HOMO
        'HOMO-2' for level two levels below the HOMO
           .... and so forth
           
        To access the unoccupied (virtual) levels, the following notation is used:
        'LUMO' for lowest unoccupied (virtual) molecular orbital
        'LUMO+1' one level above the LUMO
        'LUMO+2' two levels above the LUMO
            .... and so forth
    
        Please note: The number of occupied levels is limited by the 
                     number of electrons, so the level HOMO-50 might
                     not be present for every molecule. If not found
                     0.0 is returned.
                     The number of virtual levels is also limited,
                     and we are only including virtual levels below
                     the vacuum level, so if e.g. the level LUMO+3 
                     is not present 0.0 is returned.
                     Note also, that if the LUMO energy is positive,
                     only the LUMO energy is listed. 
                     
    subset: string
        Access to levels from all the subsets in the paper. 
        Following the notation introduced in the publication, this string
        can take the following values given in the subsets list above.
        
    '''
    
    if 'HOMO' in level_type:
        
        level = 0
        if '-' in level_type:
            level = int(level_type.split('-')[1])
            
        if subset in subsets: field = row[fields_occupied[subsets.index(subset)]]
        else: field = row[fiels_occupied[0]]
        
        if isinstance(field, list):
            if len(field) > level: energy_mo = float(field[-1-level])
            else: energy_mo = None
        else:
            energy_mo = None
            print("err", row.refcode_csd)


        
    if 'LUMO' in level_type:
        
        level = 0
        if '+' in level_type:
            level = int(level_type.split('+')[1])
        
        if subset in subsets: field = row[fields_unoccupied[subsets.index(subset)]]
        else: field = row[fiels_unoccupied[0]]
            
        if isinstance(field, list):
            if len(field) > level: energy_mo = float(field[level])
            else: energy_mo = None

        else: 
            energy_mo = None
            print("err", row.refcode_csd)
        
        
    return energy_mo



def xyz2ase(xyz_str):
    """
    Convert a xyz file to an ASE atoms object via in-memory file (StringIO).
    """
    
    xyzfile = io.StringIO()
    xyzfile.write(xyz_str)
    mol = ase.io.read(xyzfile, format="xyz")
    
    return mol

