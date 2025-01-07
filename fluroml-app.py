import streamlit as st
from rdkit import Chem
from rdkit.Chem import Draw, rdMolDraw2D
import deepchem as dc
import joblib
import pandas as pd
import py3Dmol
import matplotlib.pyplot as plt

# App Configuration
st.set_page_config(
    page_title="FluroML - Molecular Prediction",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Load Models
@st.cache_resource
def load_model(path):
    try:
        return joblib.load(path)
    except Exception as e:
        st.error(f"Error loading model from {path}: {e}")
        return None

model_fluorescence = load_model("best_classifier_compatible.joblib")
model_regression = load_model("new_best_regressor_compatible.joblib")
model_emission = load_model("best_regressor_emission_compatible.joblib")

# Helper Functions
def smiles_to_morgan(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if not mol:
        st.error("Invalid SMILES string.")
        return None
    featurizer = dc.feat.CircularFingerprint(radius=3, size=1024)
    return featurizer.featurize([mol])[0]

def smiles_to_descriptors(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if not mol:
        st.error("Invalid SMILES string.")
        return None
    featurizer = dc.feat.MACCSKeysFingerprint()
    return pd.DataFrame(featurizer.featurize([mol]))

def draw_molecule(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        drawer = rdMolDraw2D.MolDraw2DCairo(400, 400)
        rdMolDraw2D.PrepareAndDrawMolecule(drawer, mol)
        drawer.FinishDrawing()
        return drawer.GetDrawingText()
    return None

def visualize_molecule_3d(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if not mol:
        st.error("Invalid SMILES for 3D view.")
        return
    mol_block = Chem.MolToMolBlock(mol)
    view = py3Dmol.view(width=600, height=400)
    view.addModel(mol_block, "mol")
    view.setStyle({"stick": {}})
    view.zoomTo()
    view.show()

def plot_prediction(values, title):
    fig, ax = plt.subplots()
    ax.bar(range(len(values)), values)
    ax.set_title(title)
    ax.set_xlabel("Features")
    ax.set_ylabel("Prediction")
    st.pyplot(fig)

def predict(model, features):
    return model.predict([features])[0] if model else None

# Streamlit Tabs
st.title("FluroML: Advanced Molecular Predictor")
tab1, tab2, tab3, tab4 = st.tabs([
    "Fluorescence Classification",
    "Absorption Max Prediction",
    "Emission Max Prediction",
    "FRET Analysis",
])

# Tab 1: Classification
with tab1:
    st.markdown("## 🧪 Fluorescence Classification")
    st.markdown("Enter SMILES strings to predict fluorescence properties.")
    smiles = st.text_input("Enter a SMILES string:", help="Provide a molecular structure in SMILES format.")
    if smiles and model_fluorescence:
        features = smiles_to_morgan(smiles)
        if features is not None:
            prediction = predict(model_fluorescence, features)
            st.image(draw_molecule(smiles), caption="Molecule Structure")
            st.success("Fluorescent" if prediction == 1 else "Non-Fluorescent")

# Tab 2: Absorption Max
with tab2:
    st.markdown("## 🌈 Absorption Max Prediction")
    st.markdown("Predict the absorption maximum for a molecule.")
    smiles = st.text_input("Enter Molecule SMILES:")
    solvent = st.text_input("Enter Solvent SMILES:")
    if smiles and solvent and model_regression:
        desc_smiles = smiles_to_descriptors(smiles)
        desc_solvent = smiles_to_descriptors(solvent)
        if desc_smiles is not None and desc_solvent is not None:
            features = pd.concat([desc_smiles, desc_solvent], axis=1)
            prediction = predict(model_regression, features)
            st.image(draw_molecule(smiles), caption="Molecule Structure")
            st.write(f"**Predicted Absorption Max:** {prediction:.2f} nm")

# Tab 3: Emission Max
with tab3:
    st.markdown("## 🔦 Emission Max Prediction")
    st.markdown("Predict the emission maximum for a molecule.")
    smiles = st.text_input("Enter Molecule SMILES:")
    solvent = st.text_input("Enter Solvent SMILES:")
    if smiles and solvent and model_emission:
        desc_smiles = smiles_to_descriptors(smiles)
        desc_solvent = smiles_to_descriptors(solvent)
        if desc_smiles is not None and desc_solvent is not None:
            features = pd.concat([desc_smiles, desc_solvent], axis=1)
            prediction = predict(model_emission, features)
            st.image(draw_molecule(smiles), caption="Molecule Structure")
            st.write(f"**Predicted Emission Max:** {prediction:.2f} nm")

# Tab 4: FRET Analysis
with tab4:
    st.markdown("## 🔬 FRET Pair Analysis")
    st.markdown("Analyze FRET pairs and calculate efficiency.")
    donor_smiles = st.text_input("Enter Donor Molecule SMILES:")
    acceptor_smiles = st.text_input("Enter Acceptor Molecule SMILES:")
    if donor_smiles and acceptor_smiles:
        donor_emission = predict(model_emission, smiles_to_descriptors(donor_smiles))
        acceptor_absorption = predict(model_regression, smiles_to_descriptors(acceptor_smiles))
        if donor_emission and acceptor_absorption:
            fret_efficiency = (
                acceptor_absorption / (donor_emission + acceptor_absorption)
            ) * 100
            st.image(draw_molecule(donor_smiles), caption="Donor Molecule")
            st.image(draw_molecule(acceptor_smiles), caption="Acceptor Molecule")
            st.write(f"**FRET Efficiency:** {fret_efficiency:.2f}%")
