import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer, util
from llama_cpp import Llama

st.title("Threat Event Grouping Application")

# get groupings
groupings_file = st.file_uploader("Upload Groupings Excel", type="xlsx")


@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("models/e5-large-v2")

model = load_embedding_model()

if groupings_file:
    groupings = pd.read_excel(groupings_file)
    group_embeddings = model.encode(groupings["Threat Event"].tolist(), normalize_embeddings=True)

@st.cache_resource
def load_llm():
    llm = Llama(
        model_path="models/Qwen2-0.5B-Instruct-Q8_0.gguf",
        n_ctx=4096,
        n_threads=8
    )
    return llm

llm = load_llm()

# llm used only when the avg similarity is below 85%
def ai_check_group_match(threat_event, current_group, risk_info):
    prompt = f"""
    You are a cybersecurity assistant analyzing event clustering.

    Threat Event: {threat_event}
    Proposed Group: {current_group}
    Risk Context: {risk_info}

    Respond with exactly one word:
    - "GOOD" if the threat event fits well in this group
    - "NOT GOOD" if it does not fit
    """

    full_prompt = f"""<|im_start|>system
    You are an expert cybersecurity event clustering assistant.<|im_end|>
    <|im_start|>user
    {prompt}<|im_end|>
    <|im_start|>assistant
    """

    res = llm(full_prompt, max_tokens=10)
    return res["choices"][0]["text"].strip().upper()


def ai_propose_new_group(threat_event, risk_info):
    prompt = f"""
    You are a cybersecurity assistant helping cluster threat events.

    Threat Event: {threat_event}
    Risk Context: {risk_info}
    existing groups, use only if applicable: {', '.join(groupings['Group Name'].unique().tolist())}

    Propose a short, clear name for a new threat group that this threat event should belong to.
    Respond only with the new group name, with no explanations and no special characters.
    """

    full_prompt = f"""<|im_start|>system
    You are an expert cybersecurity clustering assistant.<|im_end|>
    <|im_start|>user
    {prompt}<|im_end|>
    <|im_start|>assistant
    """

    res = llm(full_prompt, max_tokens=50)
    return res["choices"][0]["text"].strip()


uploaded_file = st.file_uploader("Upload Threat Events Excel", type="xlsx")

if uploaded_file:
    new_threats = pd.read_excel(uploaded_file)

    if "Threat Event" not in new_threats.columns:
        st.error("Uploaded file must have a 'Threat Event' column.")
        st.stop()

    results = []

    for idx, row in new_threats.iterrows():
        threat_event = row["Threat Event"]
        risk_info = row.get("Risk Scenario", "")
        
        # handle NA
        if pd.isna(threat_event) or str(threat_event).strip() == "NA" or str(threat_event).strip() == "":
            results.append({
                "ThreatEvent": threat_event,
                "BestGroup": "NA",
                "HighestAvgScore": "NA",
                "FinalGroup": "NA",
                "Indicator": "No Threat Event"
            })
            continue

        # embedding-based similarity
        event_emb = model.encode([threat_event], normalize_embeddings=True)
        sims = util.cos_sim(event_emb, group_embeddings).numpy().flatten()
        groupings["Similarity"] = sims
        avg_scores = groupings.groupby("Group Name")["Similarity"].mean()
        best_group = avg_scores.idxmax()
        best_score = avg_scores.max() * 100  # convert to percentage

        indicator = ""
        final_group = best_group

        # decision logic
        if best_score >= 90:
            indicator = "Highly Accurate"
        elif best_score >= 85:
            indicator = "Moderately Accurate"
        elif best_score >= 75:
            llm_response = ai_check_group_match(threat_event, best_group, risk_info)
            if "NOT GOOD" in llm_response:
                new_group = ai_propose_new_group(threat_event, risk_info)
                final_group = new_group
                indicator = "AI Created"
            else:
                indicator = "Must Check"
        else:
            llm_response = ai_check_group_match(threat_event, best_group, risk_info)
            if "NOT GOOD" in llm_response:
                new_group = ai_propose_new_group(threat_event, risk_info)
                final_group = new_group
                indicator = "AI Generated"
            else:
                final_group = f"AI_{best_group}"
                indicator = "AI Verified"

        results.append({
            "ThreatEvent": threat_event,
            "BestGroup": best_group,
            "HighestAvgScore": round(best_score, 2),
            "FinalGroup": final_group,
            "Indicator": indicator
        })

    st.subheader("Final Results")
    results_df = pd.DataFrame(results)
    st.dataframe(results_df)
