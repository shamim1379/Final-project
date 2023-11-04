import pandas as pd
import pathlib as pth
import numpy as np
import sklearn as sk
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, silhouette_score,mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.model_selection import GridSearchCV
from sklearn.kernel_ridge import KernelRidge
from rdkit import Chem



my_csv = pth.Path("C:/Users/shamim/Desktop/FP.csv")
df = pd.read_csv(my_csv.resolve(), sep=',')
smiles=df.iloc[:, 18].values
anhydride=[]
esther=[]
COON=[]
COHalid=[]
COOLiNa=[]
alcohol=[]
acid=[]
ether=[]
carbonate=[]
ketone=[]
nitrile=[]
imine=[]
amide=[]
NCON=[]
NCOON=[]
CON=[]
CONCO=[]
NCONCO=[]
NCONCON=[]
CONCONCO=[]
NCONCONCO=[]
Aldehyde=[]
OCON=[]
tertiaryAmine=[]
secondaryAmine=[]
primaryAmine=[]
benzene=[]
OCONCO=[]
OCONCON=[]
OCONCOO=[]
NOCONCO=[]
Azo=[]
SCOC=[]
ThioKetone=[]
OCOON=[]
CON2=[]
NO2=[]
def count_anhydride(smiles):
    mol = Chem.MolFromSmiles(smiles)
    pattern = Chem.MolFromSmarts('[#6][#6](=[#8])[#8][#6](=[#8])[#6]')
    matches = mol.GetSubstructMatches(pattern)
    return len(matches)
def count_nitro(smiles):
    mol = Chem.MolFromSmiles(smiles)
    pattern = Chem.MolFromSmarts('[#7]([#8])=[#8]')
    matches = mol.GetSubstructMatches(pattern)
    return len(matches)
def count_esther(smiles):
    mol = Chem.MolFromSmiles(smiles)
    pattern = Chem.MolFromSmarts('[#6](=[#8])[#8][#6]')
    matches = mol.GetSubstructMatches(pattern)
    return len(matches)-2*count_anhydride(smiles)-2*count_carbonate(smiles)-2*count_OCONCOO(smiles)-count_OCONCON(smiles)-count_OCONCO(smiles)-count_OCON(smiles)-count_OCOON(smiles)
#-2*count_carbonate(smiles)-2*count_OCONCOO(smiles)-count_OCONCON(smiles)-count_OCONCO(smiles)-count_OCON(smiles)-count_OCOON(smiles)
def count_Azo(smiles):
    mol = Chem.MolFromSmiles(smiles)
    pattern = Chem.MolFromSmarts('[#6][#7]=[#7][#6]')
    matches = mol.GetSubstructMatches(pattern)
    return len(matches)
def count_SCOC(smiles):
    mol = Chem.MolFromSmiles(smiles)
    pattern1 = Chem.MolFromSmarts('[#16][#6](=[#8])[#6]')
    matches1 = mol.GetSubstructMatches(pattern1)
    return len(matches1)
def count_COON(smiles):
    mol = Chem.MolFromSmiles(smiles)
    pattern = Chem.MolFromSmarts('[#6][#6](=[#8])[#8][#7]')
    matches = mol.GetSubstructMatches(pattern)
    return len(matches)
#-count_NOCONCO(smiles)-count_NCOON(smiles)-count_OCOON(smiles)
def count_COHalide(smiles):
    mol = Chem.MolFromSmiles(smiles)
    pattern1 = Chem.MolFromSmarts('[#6](=[#8])[#9]')
    matches1 = mol.GetSubstructMatches(pattern1)
    pattern2 = Chem.MolFromSmarts('[#6](=[#8])[#17]')
    matches2 = mol.GetSubstructMatches(pattern2)
    pattern3 = Chem.MolFromSmarts('[#6](=[#8])[#35]')
    matches3 = mol.GetSubstructMatches(pattern3)
    return len(matches1)+len(matches2)+len(matches3)
def count_COOLiNa(smiles):
    mol = Chem.MolFromSmiles(smiles)
    pattern1 = Chem.MolFromSmarts('[#6][#6](=[#8])[#8][#3]')
    matches1 = mol.GetSubstructMatches(pattern1)
    pattern2 = Chem.MolFromSmarts('[#6][#6](=[#8])[#8][#11]')
    matches2 = mol.GetSubstructMatches(pattern2)
    return len(matches1)+len(matches2)
def count_acid(smiles):
    mol = Chem.MolFromSmiles(smiles)
    pattern = Chem.MolFromSmarts('[#6][#6](=[#8])[#8]')
    matches = mol.GetSubstructMatches(pattern)
    return len(matches)-count_esther(smiles)-2*count_anhydride(smiles)-count_COOLiNa(smiles)-count_COON(smiles)

def count_ether(smiles):
    mol = Chem.MolFromSmiles(smiles)
    pattern = Chem.MolFromSmarts('[#6][#8][#6]')
    matches = mol.GetSubstructMatches(pattern)
    return len(matches)-2*count_carbonate(smiles)-count_anhydride(smiles)-count_esther(smiles)-count_OCON(smiles)-count_OCONCO(smiles)-count_OCONCON(smiles)-2*count_OCONCOO(smiles)-count_OCOON(smiles)
def count_carbonate(smiles):
    mol = Chem.MolFromSmiles(smiles)
    pattern = Chem.MolFromSmarts('[#6][#8][#6](=[#8])[#8][#6]')
    matches = mol.GetSubstructMatches(pattern)
    return len(matches)
def count_Ketone(smiles):
    mol = Chem.MolFromSmiles(smiles)
    ketone_pattern = Chem.MolFromSmarts('[#6][#6](=[#8])[#6]')
    matches = mol.GetSubstructMatches(ketone_pattern)
    return len(matches)
def count_ThioKetone(smiles):
    mol = Chem.MolFromSmiles(smiles)
    pattern = Chem.MolFromSmarts('[#6](=[#16])')
    matches = mol.GetSubstructMatches(pattern)
    pattern2 = Chem.MolFromSmarts('[#6]=[#16][#6](=[#8])[#6]')
    matches2 = mol.GetSubstructMatches(pattern2)
    return len(matches)-len(matches2)
def count_nitrile(smiles):
    mol = Chem.MolFromSmiles(smiles)
    nitrile = Chem.MolFromSmarts('[#6]#[#7]')
    matches = mol.GetSubstructMatches(nitrile)
    return len(matches)

def count_imine(smiles):
    mol = Chem.MolFromSmiles(smiles)
    imine = Chem.MolFromSmarts('[#6]=[#7]')
    matches = mol.GetSubstructMatches(imine)
    imine2 = Chem.MolFromSmarts('[#6]=[#7][#7]')
    matches2 = mol.GetSubstructMatches(imine2)
    imine3 = Chem.MolFromSmarts('[#6]=[#7][#8][#6]')
    matches3 = mol.GetSubstructMatches(imine3)
    return len(matches)-len(matches2)-len(matches3)-count_Amide2(smiles)
def count_CON(smiles):
    mol = Chem.MolFromSmiles(smiles)
    pattern = Chem.MolFromSmarts('[#6][#8][#7]')
    matches = mol.GetSubstructMatches(pattern)
    return len(matches)-count_COON(smiles)-count_NCOON(smiles)-count_NOCONCO(smiles)-count_OCOON(smiles)
def count_alcohol(smiles):
    mol = Chem.MolFromSmiles(smiles)
    pattern = Chem.MolFromSmarts('[#6][#8]')
    matches = mol.GetSubstructMatches(pattern)
    return len(matches)-4*count_carbonate(smiles)-2*count_anhydride(smiles)-2*count_esther(smiles)-2*count_ether(smiles)-count_acid(smiles)-count_COON(smiles)-count_CON(smiles)-2*count_OCON(smiles)-2*count_OCONCO(smiles)-2*count_OCONCON(smiles)-4*count_OCONCOO(smiles)-count_NOCONCO(smiles)-2*count_OCOON(smiles)
def count_amide(smiles):
    mol = Chem.MolFromSmiles(smiles)
    pattern = Chem.MolFromSmarts('[#6](=[#8])[#7][#6]')
    matches = mol.GetSubstructMatches(pattern)
    return len(matches) - count_NCOON(smiles)- 2 * count_NCON(smiles) - 2 * count_CONCO(smiles) - 3 * count_NCONCO(smiles) - 4 * count_NCONCON(smiles) - 4 * count_CONCONCO(smiles) - 5 * count_NCONCONCO(smiles)-count_OCON(smiles)-2*count_OCONCO(smiles)-3*count_OCONCON(smiles)-2*count_OCONCOO(smiles)-2*count_NOCONCO(smiles)
def count_Amide2(smiles):
    mol = Chem.MolFromSmiles(smiles)
    pattern = Chem.MolFromSmarts('[#6](=[#8])[#7]=[#6]')
    matches = mol.GetSubstructMatches(pattern)
    return len(matches)
def count_NCON(smiles):
    mol = Chem.MolFromSmiles(smiles)
    pattern = Chem.MolFromSmarts('[#6][#7][#6](=[#8])[#7][#6]')
    matches = mol.GetSubstructMatches(pattern)
    return len(matches)-count_NCONCO(smiles)-2*count_NCONCON(smiles)-2*count_NCONCONCO(smiles)-count_CONCONCO(smiles)-count_OCONCON(smiles)
def count_NCOON(smiles):
    mol = Chem.MolFromSmiles(smiles)
    pattern = Chem.MolFromSmarts('[#6][#7][#6](=[#8])[#8][#7]')
    matches = mol.GetSubstructMatches(pattern)
    return len(matches)-count_NOCONCO(smiles)
def count_OCOON(smiles):
    mol = Chem.MolFromSmiles(smiles)
    pattern = Chem.MolFromSmarts('[#6][#8][#6](=[#8])[#8][#7]')
    matches = mol.GetSubstructMatches(pattern)
    return len(matches)
def count_CONCO(smiles):
    mol = Chem.MolFromSmiles(smiles)
    pattern = Chem.MolFromSmarts('[#6](=[#8])[#7][#6](=[#8])')
    matches = mol.GetSubstructMatches(pattern)
    return len(matches)-count_NCONCO(smiles)-count_NCONCON(smiles)-2*count_NCONCONCO(smiles)-2*count_CONCONCO(smiles)-count_OCONCO(smiles)-count_OCONCOO(smiles)-count_OCONCON(smiles)-count_NOCONCO(smiles)
def count_NCONCO(smiles):
    mol = Chem.MolFromSmiles(smiles)
    pattern = Chem.MolFromSmarts('[#6][#7][#6](=[#8])[#7][#6](=[#8])')
    matches = mol.GetSubstructMatches(pattern)
    return len(matches)-2*count_NCONCON(smiles)-3*count_NCONCONCO(smiles)-2*count_CONCONCO(smiles)-count_OCONCON(smiles)
def count_NCONCON(smiles):
    mol = Chem.MolFromSmiles(smiles)
    pattern = Chem.MolFromSmarts('[#6][#7][#6](=[#8])[#7][#6](=[#8])[#7][#6]')
    matches = mol.GetSubstructMatches(pattern)
    return len(matches)-count_NCONCONCO(smiles)
def count_CONCONCO(smiles):
    mol = Chem.MolFromSmiles(smiles)
    pattern = Chem.MolFromSmarts('[#6](=[#8])[#7][#6](=[#8])[#7][#6](=[#8])')
    matches = mol.GetSubstructMatches(pattern)
    return len(matches)-count_NCONCONCO(smiles)
def count_NCONCONCO(smiles):
    mol = Chem.MolFromSmiles(smiles)
    pattern = Chem.MolFromSmarts('[#6][#7][#6](=[#8])[#7][#6](=[#8])[#7][#6](=[#8])')
    matches = mol.GetSubstructMatches(pattern)
    return len(matches)
def count_OCON(smiles):
    mol = Chem.MolFromSmiles(smiles)
    pattern = Chem.MolFromSmarts('[#8][#6](=[#8])[#7][#6]')
    matches = mol.GetSubstructMatches(pattern)
    return len(matches)-count_OCONCON(smiles)-count_OCONCO(smiles)-2*count_OCONCOO(smiles)-count_NOCONCO(smiles)-count_NCOON(smiles)
#-2*count_OCONCOO(smiles)-count_OCONCON(smiles)-count_OCONCO(smiles)-count_NCOON(smiles)-count_NOCONCO(smiles)
def count_OCONCO(smiles):
    mol = Chem.MolFromSmiles(smiles)
    pattern = Chem.MolFromSmarts('[#6][#8][#6](=[#8])[#7][#6](=[#8])')
    matches = mol.GetSubstructMatches(pattern)
    return len(matches)-count_OCONCON(smiles)-2*count_OCONCOO(smiles)-count_NOCONCO(smiles)
#-count_NOCONCO(smiles)
def count_NOCONCO(smiles):
    mol = Chem.MolFromSmiles(smiles)
    pattern = Chem.MolFromSmarts('[#7][#8][#6](=[#8])[#7][#6](=[#8])')
    matches = mol.GetSubstructMatches(pattern)
    return len(matches)
def count_OCONCON(smiles):
    mol = Chem.MolFromSmiles(smiles)
    pattern = Chem.MolFromSmarts('[#8][#6](=[#8])[#7][#6](=[#8])[#7][#6]')
    matches = mol.GetSubstructMatches(pattern)
    return len(matches)
def count_OCONCOO(smiles):
    mol = Chem.MolFromSmiles(smiles)
    pattern = Chem.MolFromSmarts('[#8][#6](=[#8])[#7][#6](=[#8])[#8]')
    matches = mol.GetSubstructMatches(pattern)
    return len(matches)
def count_tertiaryAmine(smiles):
    mol = Chem.MolFromSmiles(smiles)
    pattern = Chem.MolFromSmarts('[#6][#7]([#6])[#6]')
    matches = mol.GetSubstructMatches(pattern)
    return len(matches)
def count_secondaryAmine(smiles):
    mol = Chem.MolFromSmiles(smiles)

    n_atoms = mol.GetNumAtoms()
    amines = 0
    for atom in mol.GetAtoms():
        # Check if atom is nitrogen
        if atom.GetAtomicNum() == 7:
            # Get neighbors and number of hydrogens for nitrogen
            neighbors = atom.GetNeighbors()
            num_Hs = atom.GetTotalNumHs()

            # Check if nitrogen has 1 hydrogen and 2 carbon neighbors
            if num_Hs == 1 and sum([n.GetAtomicNum() == 6 for n in neighbors]) == 2:
                amines += 1

    return amines-count_OCONCOO(smiles)-2*count_OCONCON(smiles)-count_OCONCO(smiles)-count_OCON(smiles)-count_NOCONCO(smiles)-3*count_NCONCONCO(smiles)-3*count_NCONCON(smiles)-2*count_CONCONCO(smiles)-2*count_NCONCO(smiles)-count_CONCO(smiles)-count_NCOON(smiles)-count_amide(smiles)-2*count_NCON(smiles)
def count_primaryAmine(smiles):
    mol = Chem.MolFromSmiles(smiles)
    pattern = Chem.MolFromSmarts('[#6][#7]')
    matches = mol.GetSubstructMatches(pattern)
    return len(matches)-3*count_tertiaryAmine(smiles)-2*count_secondaryAmine(smiles)
def count_benzene(smiles):
    mol = Chem.MolFromSmiles(smiles)
    pattern = Chem.MolFromSmarts('c1ccccc1')
    matches = mol.GetSubstructMatches(pattern)
    return len(matches)


for molecules in smiles:
    #anhydried_count=count_anhydride(molecules)
    #anhydride.append(anhydried_count)
    #ester_numbers=count_esther(molecules)
    #esther.append(ester_numbers)
    #COON_number=count_COON(molecules)
    #COON.append(COON_number)
    #COHalid_number=count_COHalide(molecules)
    #COHalid.append(COHalid_number)
    #alcohol_number=count_alcohol(molecules)
    #alcohol.append(alcohol_number)
    #acid_numbers=count_acid(molecules)
    #acid.append(acid_numbers)
    #COOLiNa_number=count_COOLiNa(molecules)
    #COOLiNa.append(COOLiNa_number)
    #ether_numbers=count_ether(molecules)
    #ether.append(ether_numbers)
    #carbonate_number=count_carbonate(molecules)
    #carbonate.append(carbonate_number)
    #ketone_number=count_Ketone(molecules)
    #ketone.append(ketone_number)
    #nitrile_number=count_nitrile(molecules)
    #nitrile.append(nitrile_number)
    #amide_number=count_amide(molecules)
    #amide.append(amide_number)
    #imine_number=count_imine(molecules)
    #imine.append(imine_number)
    #CON_numbers=count_CON(molecules)
    #CON.append(CON_numbers)
    #NCON_number=count_NCON(molecules)
    #NCON.append(NCON_number)
    #NCOON_number=count_NCOON(molecules)
    #NCOON.append(NCOON_number)
    #CONCO_number=count_CONCO(molecules)
    #CONCO.append(CONCO_number)
    #NCONCO_number=count_NCONCO(molecules)
    #NCONCO.append(NCONCO_number)
    #NCONCON_number=count_NCONCON(molecules)
    #NCONCON.append(NCONCON_number)
    #CONCONCO_number=count_CONCONCO(molecules)
    #CONCONCO.append(CONCONCO_number)
    #NCONCONCO_number=count_NCONCONCO(molecules)
    #NCONCONCO.append(NCONCONCO_number)
    #aldehyde_number=count_aldehyde(molecules)
    #Aldehyde.append(aldehyde_number)
    #OCON_number=count_OCON(molecules)
    #OCON.append(OCON_number)
    #tertiary_number=count_tertiaryAmine(molecules)
    #tertiaryAmine.append(tertiary_number)
    #secondary_number=count_secondaryAmine(molecules)
    #secondaryAmine.append(secondary_number)
    #primary_number=count_primaryAmine(molecules)
    #primaryAmine.append(primary_number)
    #benzene_counter=count_benzene(molecules)
    #benzene.append(benzene_counter)
    #OCONCO_number=count_OCONCO(molecules)
    #OCONCO.append(OCONCO_number)
    #OCONCON_number=count_OCONCON(molecules)
    #OCONCON.append(OCONCON_number)
    #OCONCOO_number=count_OCONCOO(molecules)
    #OCONCOO.append(OCONCOO_number)
    #NOCONCO_number = count_NOCONCO(molecules)
    #NOCONCO.append(NOCONCO_number)
    #Azo_number=count_Azo(molecules)
    #Azo.append(Azo_number)
    #SCOC_number=count_SCOC(molecules)
    #SCOC.append(SCOC_number)
    #ThioKetone_number=count_ThioKetone(molecules)
    #ThioKetone.append(ThioKetone_number)
    #OCOON_number=count_OCOON(molecules)
    #OCOON.append(OCOON_number)
    #carbonylSulfide_number=count_carbonylSulfide(molecules)
    #CarbonylSulfide.append(carbonylSulfide_number)
    #CON2_number=count_Amide2(molecules)
    #CON2.append(CON2_number)
    NO2_number=count_nitro(molecules)
    NO2.append(NO2_number)

groups=pd.DataFrame(NO2)
pd.DataFrame(groups).to_csv("C:/Users/shamim/Desktop/Final.csv")