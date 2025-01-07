import streamlit as st
from rdkit import Chem
import deepchem as dc
from rdkit.Chem.Draw import MolToImage
import joblib
import pandas as pd

# Load models with error handling
try:
    model_path_fluorescence = "best_classifier_compatible.joblib"
    model_fluorescence = joblib.load(model_path_fluorescence)
except Exception as e:
    st.error(f"Error loading fluorescence model: {e}")
    model_fluorescence = None

try:
    model_path_regression = "new_best_regressor_compatible.joblib"
    model_regression = joblib.load(model_path_regression)
except Exception as e:
    st.error(f"Error loading regression model: {e}")
    model_regression = None

try:
    model_path_emission = "best_regressor_emission_compatible.joblib"
    model_emission = joblib.load(model_path_emission)
except Exception as e:
    st.error(f"Error loading emission model: {e}")
    model_emission = None

# Calculate Morgan fingerprints from SMILES string
def smiles_to_morgan(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        st.error("Invalid SMILES string.")
        return None
    featurizer = dc.feat.CircularFingerprint(radius=3, size=1024)
    fp = featurizer.featurize([mol])
    return fp[0]

# Calculate descriptors for SMILES string
def smiles_to_descriptors(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        st.error("Invalid SMILES string.")
        return None
    featurizer = dc.feat.MACCSKeysFingerprint()
    descriptors = featurizer.featurize([mol])
    return pd.DataFrame(data=descriptors)

# Predict absorption max
def predict_absorption_max(model, smiles, solvent):
    smiles_desc = smiles_to_descriptors(smiles)
    solvent_desc = smiles_to_descriptors(solvent)
    if smiles_desc is None or solvent_desc is None:
        return None
    X = pd.concat([smiles_desc, solvent_desc], axis=1)
    y_pred = model.predict(X)
    return y_pred[0]

# Predict emission max
def predict_emission_max(model, smiles, solvent):
    try:
        smiles_desc = smiles_to_descriptors(smiles)
        solvent_desc = smiles_to_descriptors(solvent)
        if smiles_desc is None or solvent_desc is None:
            return None
        X = pd.concat([smiles_desc, solvent_desc], axis=1)
        emission_max_pred = model.predict(X)
        return emission_max_pred[0]
    except Exception as e:
        st.error(f"Error in predicting emission max: {e}")
        return None

# Calculate FRET efficiency
def calculate_fret_efficiency(donor_emission, acceptor_absorption):
    try:
        efficiency = (acceptor_absorption / (donor_emission + acceptor_absorption)) * 100
        return efficiency
    except Exception as e:
        st.error(f"Error in calculating FRET efficiency: {e}")
        return None

# Draw molecule structure
def draw_molecule(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return MolToImage(mol)

# Streamlit app
st.title("Molecular Predictor App")
st.sidebar.title("Options")
model_selector = st.sidebar.selectbox("Select Analysis", ["Classification", "Absorption Max", "Emission Max", "FRET Pair Analysis"])

if model_selector == "Classification":
    st.header("Fluorescence Classification")
    smiles = st.text_input("Enter a SMILES string:")
    if smiles and model_fluorescence:
        fp = smiles_to_morgan(smiles)
        if fp is not None:
            prediction = predict(model_fluorescence, fp)
            st.image(draw_molecule(smiles), caption="Molecule Structure")
            st.write("Fluorescent" if prediction == 1 else "Non-Fluorescent")

elif model_selector == "Absorption Max":
    st.header("Absorption Max Prediction")
    smiles = st.text_input("Enter a SMILES string:")
    solvent = st.text_input("Enter a solvent SMILES string:")
    if smiles and solvent and model_regression:
        result = predict_absorption_max(model_regression, smiles, solvent)
        if result is not None:
            st.image(draw_molecule(smiles), caption="Molecule Structure")
            st.write(f"Predicted Absorption Max: {result:.2f}")

elif model_selector == "Emission Max":
    st.header("Emission Max Prediction")
    smiles = st.text_input("Enter a SMILES string:")
    solvent = st.text_input("Enter a solvent SMILES string:")
    if smiles and solvent and model_emission:
        result = predict_emission_max(model_emission, smiles, solvent)
        if result is not None:
            st.image(draw_molecule(smiles), caption="Molecule Structure")
            st.write(f"Predicted Emission Max: {result:.2f}")

elif model_selector == "FRET Pair Analysis":
    st.header("FRET Pair Analysis")
    donor_smiles = st.text_input("Enter Donor SMILES string:")
    acceptor_smiles = st.text_input("Enter Acceptor SMILES string:")
    if donor_smiles and acceptor_smiles and model_emission:
        donor_emission = predict_emission_max(model_emission, donor_smiles, "")
        acceptor_absorption = predict_absorption_max(model_regression, acceptor_smiles, "")
        if donor_emission is not None and acceptor_absorption is not None:
            efficiency = calculate_fret_efficiency(donor_emission, acceptor_absorption)
            st.image(draw_molecule(donor_smiles), caption="Donor Molecule Structure")
            st.image(draw_molecule(acceptor_smiles), caption="Acceptor Molecule Structure")
            st.write(f"FRET Efficiency: {efficiency:.2f}%")
