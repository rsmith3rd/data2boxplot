import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import io
from scipy.stats import f_oneway
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import csv
import os

# Ensure Streamlit binds to the correct port
os.environ["STREAMLIT_SERVER_PORT"] = os.environ.get("PORT", "8080")


import streamlit as st

st.set_page_config(page_title="\U0001F4E6 Data to Boxplot + ANOVA", layout="wide")
st.markdown("""
<!-- Google tag (gtag.js) -->
<script async src="https://www.googletagmanager.com/gtag/js?id=G-T32QR2N5C2"></script>
<script>
  window.dataLayer = window.dataLayer || [];
  function gtag(){dataLayer.push(arguments);}
  gtag('js', new Date());

  gtag('config', 'G-T32QR2N5C2');
</script>
""", unsafe_allow_html=True)

# --- Custom UI Styling ---
st.markdown("""
    <style>
    .block-container {
        padding-top: 1rem;
        padding-bottom: 5rem;
    }
    .stMultiSelect, .stSelectbox, .stTextInput {
        font-size: 15px;
        margin-bottom: 0.3rem;
    }
    .stFileUploader > div {
        height: 75px;
        border: 2px dashed #aaa;
        border-radius: 5px;
        display: flex;
        align-items: center;
        justify-content: center;
    }
    .stRadio > div, .stCheckbox > div {
        display: flex;
        gap: 2rem;
        flex-wrap: wrap;
    }
    .twocol {
        display: flex;
        justify-content: space-between;
        gap: 2rem;
    }
    .twocol > div {
        flex: 1;
    }
    </style>
""", unsafe_allow_html=True)

st.title("📦 Data to Boxplot + ANOVA")

# --- File upload section ---
st.markdown("## \U0001F4C1 Upload Your Data")
st.caption("Supports .csv, .xlsx, .xls files up to 200MB.")

with st.container():
    col1, col2 = st.columns([1, 1])
    with col1:
        uploaded_files = st.file_uploader(
            label="Drag and drop files here",
            type=["csv", "xlsx", "xls"],
            accept_multiple_files=True,
            label_visibility="visible"
        )

    with col2:
        delimiter = None
        autodetect = True

        if uploaded_files:
            st.markdown("### \U0001F50D Delimiter Settings (for CSVs only)")
            autodetect = st.checkbox("Auto-detect delimiter for CSVs", value=True, key="autodetect_checkbox")

            if not autodetect:
                delimiter_options = {
                    "Comma ( , )": ",",
                    "Tab (\\t)": "\t",
                    "Semicolon ( ; )": ";",
                    "Pipe ( | )": "|",
                    "Other (custom)": "custom"
                }
                delimiter_choice = st.selectbox("Choose delimiter for CSV files", list(delimiter_options.keys()), key="delimiter_choice")
                delimiter = delimiter_options[delimiter_choice]
                if delimiter == "custom":
                    delimiter = st.text_input("Enter your custom delimiter", value=",", key="custom_delim")

def detect_delimiter(file):
    try:
        sample = file.read(1024).decode('utf-8', errors='ignore')
        file.seek(0)
        sniffer = csv.Sniffer()
        dialect = sniffer.sniff(sample, delimiters=[",", ";", "\t", "|"])
        return dialect.delimiter
    except Exception:
        return None

if uploaded_files:
    st.markdown("---")
    st.markdown("### 🗂️ Column Selection and Labeling for Each File")
    dataframes = []

    for idx, file in enumerate(uploaded_files):
        st.markdown(f"#### ⚙️ File: `{file.name}`")
        try:
            if file.name.endswith(".csv"):
                used_delimiter = delimiter
                if autodetect:
                    used_delimiter = detect_delimiter(file)
                    if used_delimiter is None:
                        st.error(f"❌ Could not auto-detect delimiter for `{file.name}`. Please choose one manually.")
                        continue
                df = pd.read_csv(file, delimiter=used_delimiter)
            elif file.name.endswith((".xlsx", ".xls")):
                df = pd.read_excel(file)
            else:
                st.error(f"❌ Unsupported file type: `{file.name}`")
                continue
        except Exception as e:
            st.error(f"❌ Could not read `{file.name}`: {e}")
            continue

        if df.empty or df.shape[1] == 0:
            st.warning(f"⚠️ `{file.name}` is empty or invalid.")
            continue

        numeric_cols = df.select_dtypes(include="number").columns.tolist()
        selected_cols = st.multiselect(f"Select numeric columns from `{file.name}`", numeric_cols, key=f"cols_{idx}")
        if not selected_cols:
            st.warning("⚠️ Please select at least one column to proceed.")

        for col in selected_cols:
            with st.expander(f"⚙️ Settings for column: `{col}`", expanded=True):
                label = st.text_input(
                    f"Label for `{col}`",
                    value=f"{file.name.split('.')[0]}_{col}",
                    key=f"label_{idx}_{col}"
                )
                temp_df = pd.DataFrame({"Value": df[col].dropna()})
                temp_df["Group"] = label
                dataframes.append(temp_df)
    if dataframes:
        df_all = pd.concat(dataframes, ignore_index=True)
        st.success("✅ Data combined successfully.")

        st.download_button("📥 Download Combined Data", data=df_all.to_csv(index=False).encode(), file_name="combined_data.csv", mime="text/csv")

        if st.checkbox("🔍 Show Combined Data Table"):
            st.dataframe(df_all)

        if df_all.isnull().values.any():
            st.warning("⚠️ Missing values found.")
            if st.checkbox("Remove rows with missing values?"):
                df_all = df_all.dropna()
                st.success("✅ Removed rows with missing values.")

        y_axis = "Value"
        x_axis = "Group"
        

        plot_title = st.text_input("📝 Plot Title", value="Group Comparison")
        
        with st.container():
            col1, col2 = st.columns([1, 1])
            with col1:
                custom_x_label = st.text_input("X-axis Label", value=x_axis)
                show_sample_sizes = st.checkbox("Show sample sizes below groups?", value=True)
                if st.checkbox("🚫 Remove outliers (IQR method)?"):
                    Q1 = df_all[y_axis].quantile(0.25)
                    Q3 = df_all[y_axis].quantile(0.75)
                    IQR = Q3 - Q1
                    df_all = df_all[(df_all[y_axis] >= Q1 - 1.5 * IQR) & (df_all[y_axis] <= Q3 + 1.5 * IQR)]
                    st.success("✅ Outliers removed.")
                show_points = st.checkbox("Show individual points", value=True)
                show_means = st.checkbox("Show group means", value=False)
            with col2:
                custom_y_label = st.text_input("Y-axis Label", value=y_axis)
                flip_y = st.checkbox("Flip Y-axis (most negative on top)?")
                show_violin = st.checkbox("Overlay violin plot", value=False)
                transform = st.selectbox("Transform Y-axis?", ["None", "log10", "sqrt"])
                
        plot_width = st.slider("Plot width", 6, 20, 10)
        plot_height = st.slider("Plot height", 4, 12, 6)
        
        if transform == "log10":
            df_all[y_axis] = df_all[y_axis].apply(lambda x: np.log10(x) if x > 0 else np.nan)
            df_all = df_all.dropna()
            st.warning("⚠️ Non-positive values removed for log10.")
        elif transform == "sqrt":
            df_all[y_axis] = df_all[y_axis].apply(lambda x: np.sqrt(x) if x >= 0 else np.nan)
            df_all = df_all.dropna()
            st.warning("⚠️ Negative values removed for sqrt.")

        

        fig, ax = plt.subplots(figsize=(plot_width, plot_height))
        palette = sns.color_palette("Set2")

        if show_violin:
            sns.violinplot(x=x_axis, y=y_axis, data=df_all, inner=None, palette=palette, ax=ax, linewidth=0.5, alpha=0.3)
        sns.boxplot(x=x_axis, y=y_axis, data=df_all, palette=palette, ax=ax, width=0.3)
        if show_points:
            sns.stripplot(x=x_axis, y=y_axis, data=df_all, color="black", alpha=0.4, jitter=True, ax=ax)

        if show_means:
            means = df_all.groupby(x_axis)[y_axis].mean().reset_index()
            sns.pointplot(data=means, x=x_axis, y=y_axis, color="red", markers="D", linestyles="--", ax=ax)

        if show_sample_sizes:
            for i, group in enumerate(df_all[x_axis].unique()):
                count = df_all[df_all[x_axis] == group].shape[0]
                y_min = df_all[y_axis].min()
                ax.text(i, y_min - 0.05 * abs(y_min), f"n={count}", ha='center', va='top', fontsize=9)

        ax.set_title(plot_title)
        ax.set_xlabel(custom_x_label)
        ax.set_ylabel(custom_y_label)

        if flip_y:
            ax.invert_yaxis()

        left, center, right = st.columns([1, 6, 1])  # adjust width ratios as needed

        with center:
            st.pyplot(fig)


        buf = io.BytesIO()
        fig.savefig(buf, format="png")
        st.download_button("📥 Download Plot as PNG", data=buf.getvalue(), file_name="boxplot.png", mime="image/png")

        st.markdown("### 🧪 Run ANOVA + Tukey Post Hoc Test")
        if st.checkbox("Run Statistical Analysis"):
            try:
                model = ols(f'{y_axis} ~ C({x_axis})', data=df_all).fit()
                anova_table = sm.stats.anova_lm(model, typ=2)
                st.markdown("#### ANOVA Table")
                st.dataframe(anova_table)

                st.download_button("📥 Download ANOVA Table", data=anova_table.to_csv().encode(), file_name="anova_results.csv", mime="text/csv")

                pval = anova_table["PR(>F)"].iloc[0]
                
        
                if pval < 0.05:
                    st.success(f"✅ ANOVA significant (p = {pval:.4f}) ➜ Running Tukey HSD...")
                    tukey = pairwise_tukeyhsd(endog=df_all[y_axis], groups=df_all[x_axis], alpha=0.05)
                    tukey_data = tukey.summary().data[1:]
                    tukey_columns = tukey.summary().data[0]
                    tukey_df = pd.DataFrame(tukey_data, columns=tukey_columns)
                    #tukey_df["Reject Null?"] = tukey_df["p-adj"].apply(lambda p: "Yes" if float(p) < 0.05 else "No")


                    st.markdown("#### Tukey HSD Results")
                    st.dataframe(tukey_df)

                    st.download_button("📥 Download Tukey HSD Results", data=tukey_df.to_csv(index=False).encode(), file_name="tukey_results.csv", mime="text/csv")

                    st.markdown("### ⭐ Interpretation of Significant Comparisons")
                    group_means = df_all.groupby(x_axis)[y_axis].mean().to_dict()

                    for _, row in tukey_df.iterrows():
                        g1, g2 = row["group1"], row["group2"]
                        p_adj = float(row["p-adj"])
                        if p_adj < 0.05:
                            m1, m2 = group_means.get(g1), group_means.get(g2)
                            if m1 is None or m2 is None or m2 == 0:
                                continue
                            direction = "higher" if m1 > m2 else "lower"
                            fold_change = abs(m1 / m2)
                            diff = abs(m1 - m2)
                            if fold_change >= 1.1:
                                st.markdown(f"- **{g1}** is **{fold_change:.2f}-fold {direction}** than **{g2}** (Δ = {diff:.2f}), p = {p_adj:.4f}")
                            else:
                                pct_diff = abs((m1 - m2) / m2 * 100)
                                st.markdown(f"- **{g1}** is **{pct_diff:.1f}% {direction}** than **{g2}** (Δ = {diff:.2f}), p = {p_adj:.4f}")
                else:
                    st.info(f"ANOVA not significant (p = {pval:.4f}); Tukey not performed.")
            except Exception as e:
                st.error(f"❌ Statistical analysis failed: {e}")

else:
    with col1:
         st.info("Upload your CSV or Excel file(s) to get started.")
        
   
