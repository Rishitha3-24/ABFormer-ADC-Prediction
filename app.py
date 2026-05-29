import streamlit as st
import json
import pandas as pd
import glob


try:
    from rdkit import Chem
    from rdkit.Chem import Draw
    from rdkit.Chem import Descriptors
    from rdkit.Chem import Lipinski

    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False


st.set_page_config(
    page_title="ABFormer ADC Predictor",
    layout="wide"
)




def apply_theme(dark_mode=True):

    if dark_mode:
        st.markdown("""
        <style>

        /* ===== Main Background ===== */
        .stApp {
            background-color: #0E1117;
            color: #FAFAFA;
        }

        /* ===== Remove White Header ===== */
        header {
            background-color: #0E1117 !important;
        }

        [data-testid="stHeader"] {
            background-color: #0E1117 !important;
        }

        /* ===== Sidebar ===== */
        [data-testid="stSidebar"] {
            background-color: #161B22;
            border-right: 1px solid #30363D;
        }

        /* ===== General Text ===== */
        h1, h2, h3, h4, h5, h6, p, label, div, span {
            color: #FAFAFA !important;
        }

        /* ===== Text Areas ===== */
        .stTextArea textarea {
            background-color: #1E1E1E !important;
            color: white !important;
            border-radius: 10px !important;
            border: 1px solid #444 !important;
        }

        /* ===== Placeholder Text ===== */
        .stTextArea textarea::placeholder {
            color: #AAAAAA !important;
            opacity: 1 !important;
        }

        /* ===== Number Inputs ===== */
        .stNumberInput input {
            background-color: #1E1E1E !important;
            color: white !important;
            border: 1px solid #444 !important;
            border-radius: 10px !important;
        }

        /* ===== Input Text ===== */
        input {
            color: white !important;
        }

        /* ===== Cursor Visibility ===== */
        textarea, input {
            caret-color: white !important;
        }

        /* ===== Secondary Buttons ===== */
        button[kind="secondary"] {
            background-color: #1E1E1E !important;
            color: white !important;
            border: 1px solid #444 !important;
        }

        button[kind="secondary"]:hover {
            background-color: #2A2A2A !important;
            color: white !important;
        }

        /* ===== Number Input +/- Buttons ===== */
        .stNumberInput button {
            background-color: #1E1E1E !important;
            color: white !important;
            border: 1px solid #444 !important;
        }

        .stNumberInput button:hover {
            background-color: #2A2A2A !important;
            color: white !important;
        }
                    
        /* ===== Selectbox Fix ===== */

        .stSelectbox div[data-baseweb="select"] > div {

            background-color: #1E1E1E !important;

            color: white !important;

            border: 1px solid #444 !important;

            border-radius: 14px !important;
        }

        /* Selected value text */

        .stSelectbox div[data-baseweb="select"] span {

            color: white !important;
        }

        /* Dropdown menu */

        div[role="listbox"] {

            background-color: #1E1E1E !important;

            color: white !important;

            border: 1px solid #444 !important;
        }

        /* Dropdown items */

        div[role="option"] {

            background-color: #1E1E1E !important;

            color: white !important;
        }

        div[role="option"]:hover {

            background-color: #2A2A2A !important;
        }
                    
        /* ===== Dropdown Popup Fix ===== */

        div[data-baseweb="popover"] {

            background-color: #1E1E1E !important;
        }

        ul {

            background-color: #1E1E1E !important;
        }

        li {

            background-color: #1E1E1E !important;

            color: white !important;
        }

        /* Selected option */

        li[aria-selected="true"] {

            background-color: #2A2A2A !important;

            color: white !important;
        }

        /* Hover effect */

        li:hover {

            background-color: #333333 !important;

            color: white !important;
        }           

        /* ===== Expander ===== */
        [data-testid="stExpander"] {
            background-color: #161B22 !important;
            border: 1px solid #30363D !important;
            border-radius: 12px !important;
        }

        [data-testid="stExpander"] details {
            background-color: #161B22 !important;
            color: white !important;
        }

        [data-testid="stExpander"] summary {
            background-color: #161B22 !important;
            color: white !important;
            border-radius: 12px !important;
        }

        .streamlit-expanderHeader {
            background-color: #161B22 !important;
            color: white !important;
        }

        /* ===== File Uploaders ===== */
        [data-testid="stFileUploader"] {
            background-color: #161B22 !important;
            border: 1px solid #30363D !important;
            border-radius: 12px !important;
            padding: 10px !important;
        }

        [data-testid="stFileUploader"] section {
            background-color: #161B22 !important;
            color: white !important;
        }

        [data-testid="stFileUploaderDropzone"] {
            background-color: #1E1E1E !important;
            color: white !important;
            border: 2px dashed #444 !important;
        }

        [data-testid="stFileUploaderDropzone"] * {
            color: white !important;
        }

        /* ===== JSON Viewer ===== */
        [data-testid="stJson"] {
            background-color: #161B22 !important;
            border: 1px solid #30363D !important;
            border-radius: 12px !important;
            padding: 10px !important;
        }

        [data-testid="stJson"] * {
            background-color: #161B22 !important;
            color: white !important;
        }

        /* ===== Code Blocks ===== */
        pre {
            background-color: #161B22 !important;
            color: white !important;
            border-radius: 10px !important;
        }

        /* ===== DataFrame/Table ===== */
        [data-testid="stDataFrame"] {
            background-color: #161B22 !important;
            color: white !important;
            border-radius: 12px !important;
        }

        [data-testid="stDataFrame"] * {
            color: white !important;
        }

        /* ===== DataFrame Toolbar ===== */
        [data-testid="stDataFrameToolbar"] {
            background-color: #161B22 !important;
        }

        [data-testid="stDataFrameToolbar"] * {
            background-color: #161B22 !important;
            color: white !important;
        }

        /* ===== Spinner ===== */
        .stSpinner > div {
            color: white !important;
        }

        .stSpinner svg {
            stroke: #FF4B4B !important;
        }

        /* ===== Download Button ===== */
        .stDownloadButton > button {
            background-color: #1E1E1E !important;
            color: white !important;
            border: 1px solid #444 !important;
            border-radius: 10px !important;
            font-weight: 600 !important;
        }

        .stDownloadButton > button:hover {
            background-color: #2A2A2A !important;
            color: white !important;
        }

        /* ===== Primary Buttons ===== */
        .stButton > button {
            background-color: #FF4B4B !important;
            color: white !important;
            border-radius: 10px !important;
            border: none !important;
            font-weight: 600 !important;
            padding: 0.5rem 1rem !important;
        }

        .stButton > button:hover {
            background-color: #ff2e2e !important;
            color: white !important;
        }

        /* ===== Tabs ===== */
        .stTabs [data-baseweb="tab"] {
            color: #CCCCCC !important;
        }

        .stTabs [aria-selected="true"] {
            color: #FF4B4B !important;
            border-bottom: 2px solid #FF4B4B !important;
        }

        /* ===== Metric Cards ===== */
        /* ===== Premium Glass Metric Cards ===== */

        [data-testid="metric-container"] {

            background: linear-gradient(
                145deg,
                rgba(255,255,255,0.06),
                rgba(255,255,255,0.025)
            ) !important;

            border: 1px solid rgba(255,255,255,0.10) !important;

            border-radius: 22px !important;

            padding: 22px !important;

            backdrop-filter: blur(14px);
            -webkit-backdrop-filter: blur(14px);

            box-shadow:
                0 10px 30px rgba(0,0,0,0.18);

            transition: all 0.25s ease !important;
        }

        [data-testid="metric-container"]:hover {

            transform: translateY(-3px);

            box-shadow:
                0 16px 42px rgba(0,0,0,0.28);
        }

/* ===== Premium Progress Bar ===== */

        .stProgress {

            background: rgba(255,255,255,0.04) !important;

            padding: 14px !important;

            border-radius: 18px !important;

            border: 1px solid rgba(255,255,255,0.08) !important;

            box-shadow:
                0 6px 24px rgba(0,0,0,0.12);

            margin-top: 15px !important;
        }

        .stProgress > div > div > div > div {

            background: linear-gradient(
                90deg,
                #FF4B4B,
                #ff7b7b
            ) !important;

            border-radius: 20px !important;
        }

        /* ===== Success/Error Boxes ===== */
        .stSuccess {
            background-color: rgba(0, 200, 83, 0.15) !important;
        }

        .stError {
            background-color: rgba(255, 82, 82, 0.15) !important;
        }

        /* ===== Molecule Images ===== */
        img {
            border-radius: 12px !important;
            padding: 10px !important;
            background-color: white !important;
        }

        /* ===== Divider Lines ===== */
        hr {
            border-color: #30363D !important;
        }

        </style>
        """, unsafe_allow_html=True)

    else:
        st.markdown("""
        <style>

        .stApp {
            background-color: white;
            color: black;
        }

        [data-testid="stSidebar"] {
            background-color: #F5F5F5;
        }

        .stTextArea textarea {
            border-radius: 10px;
        }

        .stButton > button {
            border-radius: 10px;
            font-weight: 600;
        }

        [data-testid="metric-container"] {
            border-radius: 12px;
            padding: 15px;
            border: 1px solid #E0E0E0;
        }

        img {
            border-radius: 12px;
            padding: 10px;
            background-color: white;
        }
                    
        

        </style>
        """, unsafe_allow_html=True)



def draw_molecule(smiles):
    if not RDKIT_AVAILABLE:
        return None

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    return Draw.MolToImage(mol, size=(360, 260))

def get_molecular_properties(smiles):

    if not RDKIT_AVAILABLE:
        return None

    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        return None

    return {
        "Molecular Weight": round(Descriptors.MolWt(mol), 2),
        "LogP": round(Descriptors.MolLogP(mol), 2),
        "TPSA": round(Descriptors.TPSA(mol), 2),
        "H-Bond Donors": Lipinski.NumHDonors(mol),
        "H-Bond Acceptors": Lipinski.NumHAcceptors(mol),
        "Rotatable Bonds": Lipinski.NumRotatableBonds(mol)
    }


def normalize_sample_keys(sample):
    key_map = {
        "heavy_chain_sequence": "Heavy_Chain_Sequence",
        "light_chain_sequence": "Light_Chain_Sequence",
        "antigen_sequence": "Antigen_Sequence",
        "payload_smiles": "Payload_SMILES",
        "linker_smiles": "Linker_SMILES",
        "dar": "DAR"
    }

    normalized = {}

    for key, value in sample.items():
        normalized_key = key_map.get(key, key)
        normalized[normalized_key] = value

    return normalized



def validate_sample(sample):
    required_keys = [
        "Heavy_Chain_Sequence",
        "Light_Chain_Sequence",
        "Antigen_Sequence",
        "Payload_SMILES",
        "Linker_SMILES",
        "DAR"
    ]

    missing = []

    for key in required_keys:
        if key not in sample:
            missing.append(key)
        elif isinstance(sample[key], str) and not sample[key].strip():
            missing.append(key)

    return missing



def show_prediction(sample, seed, threshold,prediction_mode):

    with st.spinner("Loading models and running ABFormer prediction..."):

        from inference import predict_sample

        try:
            probability, label = predict_sample(
                sample=sample,
                seed=seed,
                threshold=threshold,
                ensemble=(prediction_mode == "Ensemble")
            )

        except FileNotFoundError as e:
            st.error(str(e))
            return

        except Exception as e:
            st.exception(e)
            return

    label_text = "Active" if label == 1 else "Inactive"

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Predicted Probability",
        f"{probability:.4f}"
    )

    col2.metric(
        "Prediction",
        label_text
    )

    col3.metric(
        "Threshold",
        f"{threshold:.2f}"
    )

    st.progress(float(probability))

    if label == 1:
        st.success("The model predicts this ADC candidate as Active.")
    else:
        st.error("The model predicts this ADC candidate as Inactive.")

    result = {
        "predicted_probability": probability,
        "predicted_label": label,
        "prediction_text": label_text,
        "threshold": threshold,
        "input": sample
    }

    st.download_button(
        "Download Prediction JSON",
        data=json.dumps(result, indent=2),
        file_name="abformer_prediction.json",
        mime="application/json"
    )



with st.sidebar:
    st.header("ABFormer")
    st.caption("ADC activity prediction")
    dark_mode = st.toggle("🌙 Dark Mode", value=False)

    apply_theme(dark_mode)

    seed = 1
    threshold = 0.40



    with st.expander("Advanced model settings"):

        available_ckpts = [
            int(f.split("_")[1])
            for f in glob.glob("ckpts/ADC_*_best_model.pth")
        ]

        if len(available_ckpts) == 0:
            st.error("No model checkpoints found.")
            st.stop()

        prediction_mode = st.radio(
            "Prediction Mode",
            ["Single Seed", "Ensemble"]
        )

        if prediction_mode == "Single Seed":

            seed = st.selectbox(
                "Checkpoint seed",
                sorted(available_ckpts)
            )

        else:
            seed = None

        threshold = st.slider(
            "Prediction threshold",
            0.0,
            1.0,
            0.40,
            0.01
        )

    


st.markdown("""
<div class="hero-container">
""", unsafe_allow_html=True)

st.title("ABFormer ADC Predictor")
st.caption("Predict ADC activity using antibody, antigen, payload, linker, and DAR inputs.")

tab_manual, tab_json, tab_csv = st.tabs([
    "Manual Input",
    "Upload JSON",
    "Batch CSV"
])


with tab_manual:
    st.subheader("Single ADC Candidate")

    left, right = st.columns([1.2, 1])

    with left:
        heavy = st.text_area(
            "Heavy Chain Sequence",
            placeholder="Paste heavy chain amino acid sequence...",
            height=140
        )
        st.caption(f"Length: {len(heavy.strip())} amino acids")
        if heavy and len(heavy.strip()) > 149:
            st.warning("Heavy chain sequence is longer than 149 amino acids and may be truncated.")

        light = st.text_area(
            "Light Chain Sequence",
            placeholder="Paste light chain amino acid sequence...",
            height=140
        )
        st.caption(f"Length: {len(light.strip())} amino acids")

        antigen = st.text_area(
            "Antigen Sequence",
            placeholder="Paste antigen amino acid sequence...",
            height=140
        )
        st.caption(f"Length: {len(antigen.strip())} amino acids")
        if antigen and len(antigen.strip()) > 1024:
            st.warning("Antigen sequence is longer than 1024 amino acids and may be truncated.")


    with right:
        payload = st.text_area(
            "Payload SMILES",
            placeholder="Paste payload SMILES...",
            height=90
        )
        st.caption(f"Length: {len(payload.strip())} characters")
        if payload and len(payload.strip()) > 195:
            st.warning("Payload SMILES is longer than 195 tokens/characters and may be truncated.")

        linker = st.text_area(
            "Linker SMILES",
            placeholder="Paste linker SMILES...",
            height=90
        )
        st.caption(f"Length: {len(linker.strip())} characters")
        if linker and len(linker.strip()) > 184:
            st.warning("Linker SMILES is longer than 184 tokens/characters and may be truncated.")

        dar = st.number_input(
            "DAR",
            min_value=0.0,
            max_value=20.0,
            value=4.0,
            step=0.1
        )

    st.divider()

    if st.button("Predict", type="primary"):
        sample = {
            "Heavy_Chain_Sequence": heavy,
            "Light_Chain_Sequence": light,
            "Antigen_Sequence": antigen,
            "Payload_SMILES": payload,
            "Linker_SMILES": linker,
            "DAR": dar
        }

        missing = validate_sample(sample)

        if missing:
            st.error("Please fill these required fields: " + ", ".join(missing))
        else:
            st.success("Input accepted.")
            show_prediction(sample, seed, threshold,prediction_mode)


    st.divider()

    preview_col1, preview_col2 = st.columns(2)

    with preview_col1:
        st.subheader("Payload Structure")
        if payload.strip():
            img = draw_molecule(payload)
            if img is not None:
                st.image(img)

                props = get_molecular_properties(payload)

                if props:
                    st.markdown("### Molecular Properties")

                    col1, col2 = st.columns(2)

                    items = list(props.items())

                    for i, (key, value) in enumerate(items):

                        if i % 2 == 0:
                            with col1:
                                st.metric(key, value)
                        else:
                            with col2:
                                st.metric(key, value)
            elif RDKIT_AVAILABLE:
                st.warning("Invalid payload SMILES.")
            else:
                st.info("RDKit is not available, so molecule preview is disabled.")

    with preview_col2:
        st.subheader("Linker Structure")
        if linker.strip():
            img = draw_molecule(linker)
            if img is not None:
                st.image(img)

                props = get_molecular_properties(linker)

                if props:
                    st.markdown("### Molecular Properties")

                    col1, col2 = st.columns(2)

                    items = list(props.items())

                    for i, (key, value) in enumerate(items):

                        if i % 2 == 0:
                            with col1:
                                st.metric(key, value)
                        else:
                            with col2:
                                st.metric(key, value)
            elif RDKIT_AVAILABLE:
                st.warning("Invalid linker SMILES.")
            else:
                st.info("RDKit is not available, so molecule preview is disabled.")

    

        


with tab_json:
    st.subheader("Upload Single-Sample JSON")

    uploaded_json = st.file_uploader(
        "Upload JSON file",
        type=["json"]
    )

    if uploaded_json is not None:
        sample = normalize_sample_keys(json.load(uploaded_json))


        st.write("Uploaded input:")
        st.json(sample)

        missing = validate_sample(sample)

        if missing:
            st.error("Missing or empty fields: " + ", ".join(missing))
        else:
            if st.button("Predict From JSON", type="primary"):

                show_prediction(sample, seed, threshold,prediction_mode)

                st.divider()

                col1, col2 = st.columns(2)

                with col1:

                    st.subheader("Payload Structure")

                    payload_img = draw_molecule(sample["Payload_SMILES"])

                    if payload_img is not None:

                        st.image(payload_img)

                        props = get_molecular_properties(sample["Payload_SMILES"])

                        if props:

                            st.markdown("### Molecular Properties")

                            for key, value in props.items():
                                st.metric(key, value)

                with col2:

                    st.subheader("Linker Structure")

                    linker_img = draw_molecule(sample["Linker_SMILES"])

                    if linker_img is not None:

                        st.image(linker_img)

                        props = get_molecular_properties(sample["Linker_SMILES"])

                        if props:

                            st.markdown("### Molecular Properties")

                            for key, value in props.items():
                                st.metric(key, value)


with tab_csv:
    st.subheader("Batch Prediction")

    uploaded_csv = st.file_uploader(
        "Upload CSV file",
        type=["csv"]
    )

    if uploaded_csv is not None:
        df = pd.read_csv(uploaded_csv)

        st.write("Uploaded batch:")
        st.dataframe(df, use_container_width=True)

        required_columns = [
            "Heavy_Chain_Sequence",
            "Light_Chain_Sequence",
            "Antigen_Sequence",
            "Payload_SMILES",
            "Linker_SMILES",
            "DAR"
        ]

        missing_columns = [col for col in required_columns if col not in df.columns]

        if missing_columns:
            st.error("Missing CSV columns: " + ", ".join(missing_columns))
        else:
            if st.button("Run Batch Prediction", type="primary"):
                with st.spinner("Running batch predictions..."):
                    from inference import predict_sample

                    results = []

                    for idx, row in df.iterrows():
                        sample = {
                            "Heavy_Chain_Sequence": row["Heavy_Chain_Sequence"],
                            "Light_Chain_Sequence": row["Light_Chain_Sequence"],
                            "Antigen_Sequence": row["Antigen_Sequence"],
                            "Payload_SMILES": row["Payload_SMILES"],
                            "Linker_SMILES": row["Linker_SMILES"],
                            "DAR": float(row["DAR"])
                        }

                        probability, label = predict_sample(
                            sample=sample,
                            seed=seed if seed is not None else 1,
                            threshold=threshold,
                            ensemble=(prediction_mode == "Ensemble")
                        )

                        results.append({
                            "row": idx,
                            "predicted_probability": probability,
                            "predicted_label": label,
                            "prediction_text": "Active" if label == 1 else "Inactive",
                            "threshold": threshold
                        })

                    results_df = pd.DataFrame(results)

                    st.session_state["batch_results_df"] = results_df
                    st.session_state["batch_input_df"] = df.copy()

            if "batch_results_df" in st.session_state and "batch_input_df" in st.session_state:
                results_df = st.session_state["batch_results_df"]
                input_df = st.session_state["batch_input_df"]

                st.success("Batch prediction complete.")
                st.dataframe(results_df, use_container_width=True)

                st.divider()

                selected_row = st.selectbox(
                    "Preview molecule structures for row",
                    options=list(results_df["row"]),
                    key="preview_row_select"
                )

                selected_row = int(selected_row)
                selected_sample = input_df.loc[selected_row]

                col1, col2 = st.columns(2)

                with col1:
                    st.subheader("Payload Structure")

                    payload_img = draw_molecule(selected_sample["Payload_SMILES"])

                    if payload_img is not None:
                        st.image(payload_img)

                        props = get_molecular_properties(selected_sample["Payload_SMILES"])

                        if props:
                            for key, value in props.items():
                                st.metric(key, value)

                with col2:
                    st.subheader("Linker Structure")

                    linker_img = draw_molecule(selected_sample["Linker_SMILES"])

                    if linker_img is not None:
                        st.image(linker_img)

                        props = get_molecular_properties(selected_sample["Linker_SMILES"])

                        if props:
                            for key, value in props.items():
                                st.metric(key, value)

                st.download_button(
                    "Download Batch Results CSV",
                    data=results_df.to_csv(index=False),
                    file_name="abformer_batch_predictions.csv",
                    mime="text/csv"
                )
