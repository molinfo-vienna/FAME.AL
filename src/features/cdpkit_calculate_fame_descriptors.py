import sys
import os
import argparse
import ast
import numpy
import pandas as pd
import warnings
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)

import CDPL.Chem as Chem
import CDPL.MolProp as MolProp
import CDPL.ForceField as ForceField

from datetime import datetime

# calculate cpdkit descriptor.
    # input using .sdf or text file with smiles
    # when using .sdf, the columns should have mol_id, site of metabolism labeled as a list of atom index in "soms"
    # when using smiles, the columns should be mol_id, preprocessed_smi


def perceive_mol(mol):
    Chem.calcImplicitHydrogenCounts(mol, False)     # calculate implicit hydrogen counts and set corresponding property for all atoms
    Chem.perceiveHybridizationStates(mol, False)    # perceive atom hybridization states and set corresponding property for all atoms
    Chem.perceiveSSSR(mol, False)                   # perceive smallest set of smallest rings and store as Chem.MolecularGraph property
    Chem.setRingFlags(mol, False)                   # perceive cycles and set corresponding atom and bond properties
    Chem.setAromaticityFlags(mol, False)            # perceive aromaticity and set corresponding atom and bond properties
    Chem.perceiveSybylAtomTypes(mol, False)         # perceive Sybyl atom types and set corresponding property for all atoms


def perceive_mol_further(mol):
    Chem.calcTopologicalDistanceMatrix(mol, False)  # calculate topological distance matrix and store as Chem.MolecularGraph property
                                                          # (required for effective polarizability calculations)
    
    Chem.perceivePiElectronSystems(mol, False)   # perceive pi electron systems and store info as Chem.MolecularGraph property
                                                 # (required for MHMO calculations)

    # calculate sigma charges and electronegativities using the PEOE method and store values as atom properties
    # (prerequisite for MHMO calculations)
    MolProp.calcPEOEProperties(mol, False)  

    # calculate pi charges, electronegativities and other properties by a modified Hueckel MO method and store values as properties
    MolProp.calcMHMOProperties(mol, False)

    ForceField.perceiveMMFF94AromaticRings(mol, False)        # perceive aromatic rings according to the MMFF94 aroamticity model and store data as Chem.MolecularGraph property
    ForceField.assignMMFF94AtomTypes(mol, False, False)       # perceive MMFF94 atom types (tolerant mode) set corresponding property for all atoms
    ForceField.assignMMFF94BondTypeIndices(mol, False, False) # perceive MMFF94 bond types (tolerant mode) set corresponding property for all bonds
    ForceField.calcMMFF94AtomCharges(mol, False, False)       # calculate MMFF94 atom charges (tolerant mode) set corresponding property for all atoms


# function called for each read molecule
def procMolecule(mol: Chem.Molecule, radius: int, isSoMLabel:bool = True, isSmiles:bool = False) -> None: 
    if not isSmiles:    
        if not Chem.hasStructureData(mol):        # is a structure data property available?
            print('Error: structure data not available for molecule \'%s\'!' % Chem.getName(mol))
            return
        
        struct_data = Chem.getStructureData(mol)  # retrieve structure data        

        for entry in struct_data:                 # iterate of structure data entries consisting of a header line and the actual data
            if 'mol_id' in entry.header:
                mol_id = entry.data
            if 'soms' in entry.header:
                soms = ast.literal_eval(entry.data)
                
    perceive_mol(mol)
    perceive_mol_further(mol)

    id_som_dic = {}
    for atom in mol.atoms:
        if Chem.getSybylType(atom) == 24: continue    # remove hydrogen this way to not mess up the atom index with SoMs
        
        atom_idx = mol.getAtomIndex(atom)
        descr_names, descr = genFAMEDescriptor(atom, mol, radius)
        
        if isSoMLabel:
            if atom_idx in soms:
                som_label = 1
            else:
                som_label = 0
        
        else:
            som_label = ''
        
        if isSmiles:
            mol_id = ''
            som_label = ''
        
        id_som_dic[(mol_id,atom_idx)] = (som_label,descr)
    
    return descr_names, id_som_dic


# descriptor calculation function called for each atom of the read molecule
def genFAMEDescriptor(ctr_atom: Chem.Atom, molgraph: Chem.MolecularGraph, radius: int) -> numpy.array:
    # functions need only (atom) are commented, others need (atom,mol)
    properties_dic = {
        Chem.getSybylType: 'AtomType',    # (atom)
        MolProp.getHeavyAtomCount: 'AtomDegree',
        MolProp.getHybridPolarizability: 'HybridPolarizability',
        MolProp.getVSEPRCoordinationGeometry: 'VSEPRgeometry',
        MolProp.calcExplicitValence: 'AtomValence',
        MolProp.calcEffectivePolarizability: 'EffectivePolarizability',
        MolProp.getPEOESigmaCharge: 'SigmaCharge',    # (atom)
        ForceField.getMMFF94Charge: 'MMFF94Charge',    # (atom)
        MolProp.calcPiElectronegativity: 'PiElectronegativity',
        MolProp.getPEOESigmaElectronegativity: 'SigmaElectronegativity',    # (atom)
        MolProp.calcInductiveEffect: 'InductiveEffect',
}

    # Sybyl atom types to keep
    sybyl_atom_type_idx_cpdkit = [1,2,3,4,6,7,8,9,10,11,12,13,14,15,18,19,21,22,23,38,47,48,49,54]

    (fs,names) = zip(*properties_dic.items())  
    descr_names = [prefix + '_' + Chem.getSybylAtomTypeString(i) + '_' + str(j) for prefix in names for j in range(radius+1) for i in sybyl_atom_type_idx_cpdkit]
    descr = numpy.zeros((len(properties_dic), len(sybyl_atom_type_idx_cpdkit) * (radius + 1)),dtype=float)

    env = Chem.Fragment()                                                      # for storing of extracted environment atoms

    Chem.getEnvironment(ctr_atom, molgraph, radius, env)                       # extract environment of center atom reaching
                                                                               # out up to 'radius' bonds
    for atom in env.atoms:                                                     # iterate over extracted environment atoms
        sybyl_type = Chem.getSybylType(atom)                                   # retrieve Sybyl type of environment atom
        if sybyl_type not in sybyl_atom_type_idx_cpdkit: continue
        
        for c, t in zip(range(len(sybyl_atom_type_idx_cpdkit)),sybyl_atom_type_idx_cpdkit):
            if sybyl_type == t:
                position = c

        top_dist = Chem.getTopologicalDistance(ctr_atom, atom, molgraph)       # get top. distance between center atom and environment atom
        descr[0, (top_dist * len(sybyl_atom_type_idx_cpdkit) + position)] += 1  # instead of 1 (= Sybyl type presence) also any other numeric atom

        # for properties
        for i, f_, name in zip(range(len(fs)), fs, names):
            if name == 'AtomType':
                continue
            if name in ['SigmaCharge', 'MMFF94Charge', 'SigmaElectronegativity']:
                prop = f_(atom)   
            else:
                prop = f_(atom, molgraph)
            descr[(i,(top_dist * len(sybyl_atom_type_idx_cpdkit) + position))] += prop    # sum up property
    
    for i in range(len(fs)-1):
        descr[i+1,:] = numpy.divide(descr[i+1,:], descr[0,:], out = numpy.zeros_like(descr[i+1,:]), where = descr[0,:]!=0)    # averaging property and when divide by 0 give 0

    # calculate max_top_dist, the longest distance in a molecules, independent from atoms
    max_top_dist =0
    for atom1 in molgraph.atoms:
        for atom2 in molgraph.atoms:
            distance = Chem.getTopologicalDistance(atom1, atom2, molgraph)
            if distance > max_top_dist:
                max_top_dist = distance
    
    # calculate max_distance_center, the longest distance between this center atom and other
    max_distance_center = 0
    for atom in molgraph.atoms:
        distance = Chem.getTopologicalDistance(ctr_atom, atom, molgraph)
        if distance > max_distance_center:
            max_distance_center = distance

    # add the 4 descriptors related to topological distance        
    descr_names = numpy.append(descr_names, ['longestMaxTopDistinMolecule','highestMaxTopDistinMatrixRow','diffSPAN','refSPAN'])
    descr = descr.flatten() 
    if max_top_dist == 0: #had compound with one heavy atom, lazy solution now
        descr = numpy.append(descr, [max_top_dist, max_distance_center, max_top_dist-max_distance_center, 0/1])
    else:
        descr = numpy.append(descr, [max_top_dist, max_distance_center, max_top_dist-max_distance_center, max_distance_center/max_top_dist])

    descr = descr.round(4)
    return descr_names, descr



def parseArgs() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Calculates cicular 2D atom descriptors for given input molecules.')

    parser.add_argument('-i',
                        dest='in_file',
                        required=True,
                        metavar='<input file>',
                        help='Input molecule file')
    parser.add_argument('-o',
                        dest='out_folder',
                        required=True,
                        metavar='<output folder>',
                        help='Descriptors output location')
    parser.add_argument('-r',
                        dest='radius',
                        required=False,
                        metavar='<radius>',
                        default=5,
                        help='Max. atom environment radius in number of bonds',
                        type=int)
    parser.add_argument('-s',
                        dest='isSmiles',
                        required=False,
                        metavar='<is input smiles>',
                        help='the input is file with smiles or not, default is not Smiles',
                        action=argparse.BooleanOptionalAction)
    parser.add_argument('-m',
                        dest='isSoMLabel',
                        required=False,
                        metavar='<The input has SoM annotation>',
                        help='the input has SoMs notations; combine with sdf input (no -s)',
                        action=argparse.BooleanOptionalAction)

    parse_args = parser.parse_args()

    return parse_args


def main() -> None:
    args = parseArgs()
    # print(args.radius)
    # print(args.isSmiles)
    # print(args.isSoMLabel)

    sybyl_atom_type_idx_cpdkit = [1,2,3,4,6,7,8,9,10,11,12,13,14,15,18,19,21,22,23,38,47,48,49,54] + [24]

    # create output folder if it does not exist
    if not os.path.exists(args.out_folder):
        os.makedirs(args.out_folder)
        print('The new output folder is created.')

    if args.isSmiles:
        file_df = pd.read_csv(args.in_file)
        
        # # TODO: to be removed, test with first rows
        # file_df = file_df.head()
        
        out_not_calculated_cpds = '%s/%s_%s%s.csv' % (args.out_folder,args.in_file.split('/')[-1].split('.')[0],str(args.radius),'_not_calculated_cpds')
        if not os.path.exists(out_not_calculated_cpds):
            with open(out_not_calculated_cpds,'w') as f:
                f.write('Sybyl_AtomTypeIdx\tSybyl_AtomType\tCompound_id\n')

                out_descriptors = '%s/%s_%s%s.csv' % (args.out_folder,args.in_file.split('/')[-1].split('.')[0],str(args.radius),'_descriptors')
                if not os.path.exists(out_descriptors):
                    with open(out_descriptors,'w') as f2:
                        i = 0
                        for cpd_id, smi in zip(list(file_df.mol_id),list(file_df.preprocessed_smi)):
                            mol = Chem.parseSMILES(smi)
                            
                            # if have element other than in the list, don't process with it
                            perceive_mol(mol)
                            uncommon_element = 0
                            for atom in mol.atoms:
                                atom_type = Chem.getSybylType(atom)
                                if atom_type not in sybyl_atom_type_idx_cpdkit:
                                    f.write(str(atom_type)+'\t'+Chem.getSybylAtomTypeString(atom_type)+'\t'+str(cpd_id)+'\n')
                                    uncommon_element = 1
                            
                            if uncommon_element == 1:
                                continue
                                
                            else:
                                descr_names, property_dic = procMolecule(mol, args.radius, isSoMLabel=False, isSmiles=args.isSmiles)
                                if i == 0:
                                    f2.write('som_label,mol_id,atom_idx,'+','.join(descr_names)+'\n')
                                i += 1
                                for key, value in property_dic.items():
                                    som_label, descriptors = value
                                    mol_id, atom_idx = key
                                    f2.write('%s,%s,%s,%s\n' % (str(som_label),str(cpd_id),str(atom_idx),','.join([str(des) for des in descriptors])))
                                        
                else:
                    print('File '+out_descriptors+' already exists, cannot overwrite.')
                        
        else:
            print('File '+out_not_calculated_cpds+' already exists, cannot overwrite.')   
                                           
        
    else:
        # create reader for MDL SD-files
        reader = Chem.FileSDFMoleculeReader(args.in_file)
        
        # create an instance of the default implementation of the Chem.Molecule interface
        mol = Chem.BasicMolecule()

        out_not_calculated_cpds = '%s/%s_%s%s.csv' % (args.out_folder,args.in_file.split('/')[-1].split('.')[0],str(args.radius),'_not_calculated_cpds')
        if not os.path.exists(out_not_calculated_cpds):
            with open(out_not_calculated_cpds,'w') as f:
                f.write('Sybyl_AtomTypeIdx\tSybyl_AtomType\tCompound_id\n')

                out_descriptors = '%s/%s_%s%s.csv' % (args.out_folder,args.in_file.split('/')[-1].split('.')[0],str(args.radius),'_descriptors')
                if not os.path.exists(out_descriptors):
                    with open(out_descriptors,'w') as f2:
                        i = 0
                        
                        # read and process molecules one after the other until the end of input has been reached
                        try:
                            while reader.read(mol): 
                                try: 
                                    # if have element other than in the list, don't process with it
                                    perceive_mol(mol)
                                    uncommon_element = 0
                                    for atom in mol.atoms:
                                        atom_type = Chem.getSybylType(atom)
                                        if atom_type not in sybyl_atom_type_idx_cpdkit:
                                            f.write(str(atom_type)+'\t'+Chem.getSybylAtomTypeString(atom_type)+'\t'+str(cpd_id)+'\n')
                                            uncommon_element = 1
                                    
                                    if uncommon_element == 1:
                                        continue
                                        
                                    else:
                                        descr_names, property_dic = procMolecule(mol, args.radius, isSoMLabel = args.isSoMLabel)
                                        if i == 0:
                                            f2.write('som_label,mol_id,atom_idx,'+','.join(descr_names)+'\n')
                                        i += 1
                                        for key, value in property_dic.items():
                                            som_label, descriptors = value
                                            mol_id, atom_idx = key
                                            f2.write('%s,%s,%s,%s\n' % (str(som_label),str(mol_id),str(atom_idx),','.join([str(des) for des in descriptors])))

                                except Exception as e:
                                    sys.exit('Error: processing of molecule failed:\n' + str(e))
                                    
                        except Exception as e: # handle exception raised in case of severe read errors
                            sys.exit('Error: reading molecule failed:\n' + str(e))
                else:
                    print('File '+out_descriptors+' already exists, cannot overwrite.')
                        
        else:
            print('File '+out_not_calculated_cpds+' already exists, cannot overwrite.')   
            

        
if __name__ == '__main__':
    start_time = datetime.now()

    main()

    print('Finished in:')
    print(datetime.now() - start_time)
    
    sys.exit(0)
