"""
AgriStat Complete - Enterprise Agricultural Statistics Software
Version : 3.0.0
Designs : CRD | RCBD | Latin Square | Split-Plot | Lattice | Factorial
Author  : AgriStat Engine
"""

import streamlit as st
import warnings
import re
from typing import Optional, List

warnings.filterwarnings("ignore")

_missing_packages = []

# --- SAFE IMPORTS ---
try:
    import pandas as pd
except ImportError:
    _missing_packages.append("pandas")
    pd = None

try:
    import numpy as np
except ImportError:
    _missing_packages.append("numpy")
    np = None

try:
    import scipy.stats as sp_stats
    from scipy.stats import f_oneway, shapiro
except ImportError:
    _missing_packages.append("scipy")
    sp_stats = None

try:
    import statsmodels.formula.api as smf
    from statsmodels.stats.anova import anova_lm
    from statsmodels.stats.multicomp import pairwise_tukeyhsd
except ImportError:
    _missing_packages.append("statsmodels")
    smf = None

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
except ImportError:
    _missing_packages.append("matplotlib")
    plt = None
    sns = None

try:
    import plotly.express as px
    import plotly.graph_objects as go
except ImportError:
    _missing_packages.append("plotly")
    px = None
    go = None

# --- DEPENDENCY GUARD ---
if _missing_packages:
    st.set_page_config(page_title="AgriStat - Setup Required", page_icon="!")
    st.error("## Missing Required Packages")
    st.warning(f"Missing: {', '.join(_missing_packages)}")
    st.info(
        "Add all missing packages to requirements.txt, "
        "push to GitHub, and reboot the Streamlit app."
    )
    st.stop()


# =====================================================================
# CLASS 1 - DATA PROCESSOR
# =====================================================================
class DataProcessor:

    @staticmethod
    def load_file(uploaded_file):
        try:
            name = uploaded_file.name.lower()
            if name.endswith(".csv"):
                return pd.read_csv(uploaded_file), None
            elif name.endswith((".xlsx", ".xls")):
                return pd.read_excel(uploaded_file, engine="openpyxl"), None
            return None, "Unsupported format. Please upload CSV or Excel."
        except Exception as exc:
            return None, f"File load error: {exc}"

    @staticmethod
    def sanitize_headers(df):
        col_map = {}
        seen = {}
        for col in df.columns:
            clean = re.sub(r"[^a-zA-Z0-9]", "_", str(col)).strip("_")
            if clean and clean[0].isdigit():
                clean = "var_" + clean
            if not clean:
                clean = "col"
            base = clean
            count = seen.get(base, 0)
            if count:
                clean = f"{base}_{count}"
            seen[base] = count + 1
            col_map[str(col)] = clean
        return df.rename(columns=col_map), col_map

    @staticmethod
    def get_column_types(df):
        num = df.select_dtypes(include=np.number).columns.tolist()
        cat = df.select_dtypes(exclude=np.number).columns.tolist()
        return num, cat


# =====================================================================
# CLASS 2 - DESCRIPTIVE STATISTICS
# =====================================================================
class DescriptiveStats:

    @staticmethod
    def compute_summary(df, columns):
        records = []
        for col in columns:
            s = df[col].dropna()
            if s.empty:
                continue
            mean_val = s.mean()
            std_val = s.std()
            cv = (std_val / mean_val * 100) if mean_val != 0 else float("nan")
            if len(s) >= 3:
                try:
                    _, p_norm = shapiro(s[:5000])
                except Exception:
                    p_norm = float("nan")
            else:
                p_norm = float("nan")
            records.append({
                "Variable":        col,
                "N":               len(s),
                "Mean":            round(float(mean_val), 4),
                "Median":          round(float(s.median()), 4),
                "Std Dev":         round(float(std_val), 4),
                "CV (%)":          round(float(cv), 2),
                "Skewness":        round(float(s.skew()), 4),
                "Kurtosis":        round(float(s.kurtosis()), 4),
                "Min":             round(float(s.min()), 4),
                "Max":             round(float(s.max()), 4),
                "Shapiro-Wilk p":  round(float(p_norm), 4),
            })
        if records:
            return pd.DataFrame(records).set_index("Variable")
        return pd.DataFrame()


# =====================================================================
# CLASS 3 - INFERENTIAL STATISTICS
# =====================================================================
class InferentialStats:

    # --- SHARED PRECISION METRICS ---
    @staticmethod
    def _precision(ms_error, df_error, r, grand_mean, alpha):
        sem  = float(np.sqrt(ms_error / r))
        t_crit = float(sp_stats.t.ppf(1 - alpha / 2, df_error))
        cd   = float(t_crit * np.sqrt(2 * ms_error / r))
        cv   = (
            float((np.sqrt(ms_error) / grand_mean) * 100)
            if grand_mean != 0
            else float("nan")
        )
        return round(sem, 4), round(cd, 4), round(cv, 2)

    # --- TUKEY HSD POST-HOC ---
    @staticmethod
    def tukey_hsd(df, response, treatment, alpha):
        dc = df[[response, treatment]].dropna()
        result = pairwise_tukeyhsd(
            endog=dc[response],
            groups=dc[treatment],
            alpha=alpha,
        )
        rows = result._results_table.data
        return pd.DataFrame(rows[1:], columns=rows[0])

    # --- CRD ANOVA ---
    @staticmethod
    def run_crd(df, response, treatment, alpha=0.05):
        dc = df[[response, treatment]].dropna().copy()
        groups = [g[response].values for _, g in dc.groupby(treatment)]
        k = len(groups)
        n_total = len(dc)
        grand_mean = float(dc[response].mean())

        ss_trt   = float(sum(len(g) * (np.mean(g) - grand_mean) ** 2 for g in groups))
        ss_total = float(((dc[response] - grand_mean) ** 2).sum())
        ss_err   = ss_total - ss_trt
        df_trt   = k - 1
        df_err   = n_total - k
        ms_trt   = ss_trt / df_trt
        ms_err   = ss_err / df_err if df_err > 0 else float("nan")
        f_stat, p_val = f_oneway(*groups)

        anova_table = pd.DataFrame({
            "Source":  ["Treatment", "Error", "Total"],
            "SS":      [round(ss_trt, 4),        round(ss_err, 4),   round(ss_total, 4)],
            "df":      [df_trt,                   df_err,             n_total - 1],
            "MS":      [round(ms_trt, 4),         round(ms_err, 4),   "---"],
            "F-Value": [round(float(f_stat), 4),  "---",              "---"],
            "P-Value": [round(float(p_val),  4),  "---",              "---"],
        })

        means_table = (
            dc.groupby(treatment)[response]
            .agg(["count", "mean", "std"])
            .rename(columns={"count": "n", "mean": "Mean", "std": "Std Dev"})
            .round(4)
        )

        avg_r = float(np.mean([len(g) for g in groups]))
        sem, cd, cv = InferentialStats._precision(
            ms_err, df_err, avg_r, grand_mean, alpha
        )
        return anova_table, means_table, sem, cd, cv, float(p_val)

    # --- RCBD ANOVA ---
    @staticmethod
    def run_rcbd(df, response, treatment, block, alpha=0.05):
        dc = df[[response, treatment, block]].dropna().copy()
        formula = (
            f'Q("{response}") ~ C(Q("{block}")) + C(Q("{treatment}"))'
        )
        model = smf.ols(formula, data=dc).fit()
        raw = anova_lm(model, typ=2).reset_index()
        raw.columns = ["Source", "SS", "df", "F-Value", "P-Value"]
        raw["MS"] = raw["SS"] / raw["df"]
        raw["Source"] = raw["Source"].replace({
            f'C(Q("{block}"))':     "Block",
            f'C(Q("{treatment}"))': "Treatment",
            "Residual":             "Error",
        })

        anova_table = (
            raw[["Source", "SS", "df", "MS", "F-Value", "P-Value"]]
            .round(4)
        )

        err_row    = anova_table[anova_table["Source"] == "Error"].iloc[0]
        ms_err     = float(err_row["MS"])
        df_err     = float(err_row["df"])
        r          = float(dc[block].nunique())
        grand_mean = float(dc[response].mean())

        sem, cd, cv = InferentialStats._precision(
            ms_err, df_err, r, grand_mean, alpha
        )

        means_table = (
            dc.groupby(treatment)[response]
            .agg(["count", "mean", "std"])
            .rename(columns={"count": "n", "mean": "Mean", "std": "Std Dev"})
            .round(4)
        )
        return anova_table, means_table, sem, cd, cv

    # --- LATIN SQUARE ANOVA ---
    @staticmethod
    def run_latin_square(df, response, treatment, row_blk, col_blk, alpha=0.05):
        dc = df[[response, treatment, row_blk, col_blk]].dropna().copy()
        formula = (
            f'Q("{response}") ~ '
            f'C(Q("{row_blk}")) + '
            f'C(Q("{col_blk}")) + '
            f'C(Q("{treatment}"))'
        )
        model = smf.ols(formula, data=dc).fit()
        raw = anova_lm(model, typ=2).reset_index()
        raw.columns = ["Source", "SS", "df", "F-Value", "P-Value"]
        raw["MS"] = raw["SS"] / raw["df"]
        raw["Source"] = raw["Source"].replace({
            f'C(Q("{row_blk}"))':   "Row Block",
            f'C(Q("{col_blk}"))':   "Column Block",
            f'C(Q("{treatment}"))': "Treatment",
            "Residual":             "Error",
        })

        anova_table = (
            raw[["Source", "SS", "df", "MS", "F-Value", "P-Value"]]
            .round(4)
        )

        err_row    = anova_table[anova_table["Source"] == "Error"].iloc[0]
        ms_err     = float(err_row["MS"])
        df_err     = float(err_row["df"])
        t_levels   = float(dc[treatment].nunique())
        grand_mean = float(dc[response].mean())

        sem, cd, cv = InferentialStats._precision(
            ms_err, df_err, t_levels, grand_mean, alpha
        )

        means_table = (
            dc.groupby(treatment)[response]
            .agg(["count", "mean", "std"])
            .rename(columns={"count": "n", "mean": "Mean", "std": "Std Dev"})
            .round(4)
        )
        return anova_table, means_table, sem, cd, cv

    # --- SPLIT-PLOT ANOVA ---
    @staticmethod
    def run_split_plot(df, response, replication, main_factor, sub_factor):
        dc = df[[response, replication, main_factor, sub_factor]].dropna().copy()
        formula = (
            f'Q("{response}") ~ '
            f'C(Q("{replication}")) + '
            f'C(Q("{main_factor}")) + '
            f'C(Q("{replication}")):C(Q("{main_factor}")) + '
            f'C(Q("{sub_factor}")) + '
            f'C(Q("{main_factor}")):C(Q("{sub_factor}"))'
        )
        model = smf.ols(formula, data=dc).fit()
        raw = anova_lm(model, typ=1).reset_index()
        raw.columns = ["Source", "SS", "df", "F-Value", "P-Value"]
        raw["MS"] = raw["SS"] / raw["df"]
        raw["Source"] = raw["Source"].replace({
            f'C(Q("{replication}"))':                          "Replication",
            f'C(Q("{main_factor}"))':                          "Main Plot Factor",
            f'C(Q("{replication}")):C(Q("{main_factor}"))':    "Main Plot Error",
            f'C(Q("{sub_factor}"))':                           "Sub Plot Factor",
            f'C(Q("{main_factor}")):C(Q("{sub_factor}"))':     "Main x Sub Interaction",
            "Residual":                                        "Sub Plot Error",
        })

        lookup = raw.set_index("Source")

        ms_main_err = float(lookup.loc["Main Plot Error",       "MS"])
        df_main_err = float(lookup.loc["Main Plot Error",       "df"])
        ms_sub_err  = float(lookup.loc["Sub Plot Error",        "MS"])
        df_sub_err  = float(lookup.loc["Sub Plot Error",        "df"])
        ms_a        = float(lookup.loc["Main Plot Factor",      "MS"])
        df_a        = float(lookup.loc["Main Plot Factor",      "df"])
        ms_b        = float(lookup.loc["Sub Plot Factor",       "MS"])
        df_b        = float(lookup.loc["Sub Plot Factor",       "df"])
        ms_ab       = float(lookup.loc["Main x Sub Interaction","MS"])
        df_ab       = float(lookup.loc["Main x Sub Interaction","df"])

        f_a  = ms_a  / ms_main_err
        f_b  = ms_b  / ms_sub_err
        f_ab = ms_ab / ms_sub_err

        p_a  = float(sp_stats.f.sf(f_a,  df_a,  df_main_err))
        p_b  = float(sp_stats.f.sf(f_b,  df_b,  df_sub_err))
        p_ab = float(sp_stats.f.sf(f_ab, df_ab, df_sub_err))

        anova_table = pd.DataFrame([
            {
                "Source":     "Replication",
                "SS":         round(float(lookup.loc["Replication", "SS"]), 4),
                "df":         int(float(lookup.loc["Replication", "df"])),
                "MS":         round(float(lookup.loc["Replication", "MS"]), 4),
                "Test Error": "---",
                "F-Value":    "---",
                "P-Value":    "---",
            },
            {
                "Source":     "Main Plot Factor",
                "SS":         round(float(lookup.loc["Main Plot Factor", "SS"]), 4),
                "df":         int(df_a),
                "MS":         round(ms_a, 4),
                "Test Error": "Main Plot Error",
                "F-Value":    round(f_a, 4),
                "P-Value":    round(p_a, 4),
            },
            {
                "Source":     "Main Plot Error",
                "SS":         round(float(lookup.loc["Main Plot Error", "SS"]), 4),
                "df":         int(df_main_err),
                "MS":         round(ms_main_err, 4),
                "Test Error": "---",
                "F-Value":    "---",
                "P-Value":    "---",
            },
            {
                "Source":     "Sub Plot Factor",
                "SS":         round(float(lookup.loc["Sub Plot Factor", "SS"]), 4),
                "df":         int(df_b),
                "MS":         round(ms_b, 4),
                "Test Error": "Sub Plot Error",
                "F-Value":    round(f_b, 4),
                "P-Value":    round(p_b, 4),
            },
            {
                "Source":     "Main x Sub Interaction",
                "SS":         round(float(lookup.loc["Main x Sub Interaction", "SS"]), 4),
                "df":         int(df_ab),
                "MS":         round(ms_ab, 4),
                "Test Error": "Sub Plot Error",
                "F-Value":    round(f_ab, 4),
                "P-Value":    round(p_ab, 4),
            },
            {
                "Source":     "Sub Plot Error",
                "SS":         round(float(lookup.loc["Sub Plot Error", "SS"]), 4),
                "df":         int(df_sub_err),
                "MS":         round(ms_sub_err, 4),
                "Test Error": "---",
                "F-Value":    "---",
                "P-Value":    "---",
            },
        ])

        main_means = (
            dc.groupby(main_factor)[response]
            .agg(["count", "mean", "std"])
            .rename(columns={"count": "n", "mean": "Mean", "std": "Std Dev"})
            .round(4)
        )
        sub_means = (
            dc.groupby(sub_factor)[response]
            .agg(["count", "mean", "std"])
            .rename(columns={"count": "n", "mean": "Mean", "std": "Std Dev"})
            .round(4)
        )
        interaction_means = (
            dc.groupby([main_factor, sub_factor])[response]
            .agg(["count", "mean", "std"])
            .rename(columns={"count": "n", "mean": "Mean", "std": "Std Dev"})
            .reset_index()
            .round(4)
        )

        grand_mean  = float(dc[response].mean())
        sem_main, cd_main, cv_main = InferentialStats._precision(
            ms_main_err, df_main_err, dc[replication].nunique(), grand_mean, 0.05
        )
        sem_sub, cd_sub, cv_sub = InferentialStats._precision(
            ms_sub_err, df_sub_err, dc[replication].nunique(), grand_mean, 0.05
        )

        return (
            anova_table,
            main_means,
            sub_means,
            interaction_means,
            sem_main, cd_main, cv_main,
            sem_sub,  cd_sub,  cv_sub,
        )

    # --- LATTICE ANOVA ---
    @staticmethod
    def run_lattice(df, response, treatment, replication, block):
        dc = df[[response, treatment, replication, block]].dropna().copy()
        dc["_blk_nested"] = (
            dc[replication].astype(str) + "__" + dc[block].astype(str)
        )
        formula = (
            f'Q("{response}") ~ '
            f'C(Q("{replication}")) + '
            f'C(_blk_nested) + '
            f'C(Q("{treatment}"))'
        )
        model = smf.ols(formula, data=dc).fit()
        raw = anova_lm(model, typ=1).reset_index()
        raw.columns = ["Source", "SS", "df", "F-Value", "P-Value"]
        raw["MS"] = raw["SS"] / raw["df"]
        raw["Source"] = raw["Source"].replace({
            f'C(Q("{replication}"))': "Replication",
            "C(_blk_nested)":        "Blocks within Replication",
            f'C(Q("{treatment}"))':  "Treatment",
            "Residual":              "Error",
        })

        anova_table = (
            raw[["Source", "SS", "df", "MS", "F-Value", "P-Value"]]
            .round(4)
        )

        means_table = (
            dc.groupby(treatment)[response]
            .agg(["count", "mean", "std"])
            .rename(columns={"count": "n", "mean": "Mean", "std": "Std Dev"})
            .round(4)
        )

        err_ms     = float(anova_table[anova_table["Source"] == "Error"]["MS"].iloc[0])
        df_err     = float(anova_table[anova_table["Source"] == "Error"]["df"].iloc[0])
        grand_mean = float(dc[response].mean())
        r          = float(dc[replication].nunique())

        cv = (
            float((np.sqrt(err_ms) / grand_mean) * 100)
            if grand_mean != 0
            else float("nan")
        )

        sem, cd, _ = InferentialStats._precision(
            err_ms, df_err, r, grand_mean, 0.05
        )

        rcbd_formula = (
            f'Q("{response}") ~ '
            f'C(Q("{replication}")) + C(Q("{treatment}"))'
        )
        rcbd_model = smf.ols(rcbd_formula, data=dc).fit()
        rcbd_raw   = anova_lm(rcbd_model, typ=2).reset_index()
        rcbd_raw.columns = ["Source", "SS", "df", "F-Value", "P-Value"]
        rcbd_raw["MS"]   = rcbd_raw["SS"] / rcbd_raw["df"]
        ms_rcbd = float(
            rcbd_raw.loc[rcbd_raw["Source"] == "Residual", "MS"].iloc[0]
        )
        rel_efficiency = (ms_rcbd / err_ms * 100) if err_ms > 0 else float("nan")

        return (
            anova_table,
            means_table,
            round(rel_efficiency, 2),
            round(cv, 2),
            round(sem, 4),
            round(cd, 4),
        )

    # --- FACTORIAL ANOVA ---
    @staticmethod
    def run_factorial(df, response, factors, block=None):
        if len(factors) < 2:
            raise ValueError("Please select at least two factors.")

        use_cols = [response] + factors + ([block] if block else [])
        dc = df[use_cols].dropna().copy()
        factor_expr = " * ".join([f'C(Q("{f}"))' for f in factors])
        formula = (
            f'Q("{response}") ~ C(Q("{block}")) + {factor_expr}'
            if block
            else f'Q("{response}") ~ {factor_expr}'
        )

        model = smf.ols(formula, data=dc).fit()
        raw = anova_lm(model, typ=2).reset_index()
        raw.columns = ["Source", "SS", "df", "F-Value", "P-Value"]
        raw["MS"] = raw["SS"] / raw["df"]

        def prettify(term):
            out = str(term)
            if block:
                out = out.replace(f'C(Q("{block}"))', block)
            for f in factors:
                out = out.replace(f'C(Q("{f}"))', f)
            out = out.replace(":", " x ")
            if out == "Residual":
                out = "Error"
            return out

        raw["Source"] = raw["Source"].map(prettify)
        anova_table  = (
            raw[["Source", "SS", "df", "MS", "F-Value", "P-Value"]]
            .round(4)
        )

        means_table = (
            dc.groupby(factors)[response]
            .agg(["count", "mean", "std"])
            .rename(columns={"count": "n", "mean": "Mean", "std": "Std Dev"})
            .reset_index()
            .round(4)
        )
        return anova_table, means_table


# =====================================================================
# UI HELPER FUNCTIONS
# =====================================================================
def _render_precision_metrics(sem, cd, cv):
    st.markdown("**Precision Metrics**")
    c1, c2, c3 = st.columns(3)
    c1.metric("SEm (Standard Error of Mean)",   f"{sem:.4f}")
    c2.metric("CD at 5% (Critical Difference)", f"{cd:.4f}")
    c3.metric("CV % (Coefficient of Variation)",f"{cv:.2f} %")
    if cv <= 10:
        st.success("CV% is excellent (<=10%). High experimental precision.")
    elif cv <= 20:
        st.info("CV% is acceptable (10-20%). Good experimental precision.")
    else:
        st.warning("CV% is high (>20%). Check for data entry errors or high variability.")


def _render_anova_table(df):
    def highlight_sig(row):
        styles = [""] * len(row)
        try:
            pval = float(row["P-Value"])
            if pval < 0.01:
                styles = ["background-color: #d4edda; color: #155724"] * len(row)
            elif pval < 0.05:
                styles = ["background-color: #fff3cd; color: #856404"] * len(row)
        except (ValueError, TypeError):
            pass
        return styles
    st.dataframe(
        df.style.apply(highlight_sig, axis=1),
        use_container_width=True,
    )


def _render_post_hoc(df, response, treatment, alpha, p_val):
    if p_val < alpha:
        st.info(
            f"Treatment effect is **significant** "
            f"(p = {p_val:.4f} < alpha = {alpha}). "
            "Running Tukey HSD post-hoc test..."
        )
        try:
            tukey_df = InferentialStats.tukey_hsd(df, response, treatment, alpha)
            st.dataframe(tukey_df, use_container_width=True)
        except Exception as tukey_err:
            st.warning(f"Tukey HSD could not be computed: {tukey_err}")
    else:
        st.info(
            f"Treatment effect is **not significant** "
            f"(p = {p_val:.4f} >= alpha = {alpha}). "
            "Post-hoc test is not required."
        )


# =====================================================================
# MAIN APPLICATION CONTROLLER
# =====================================================================
def main():
    st.set_page_config(
        page_title="AgriStat Complete",
        page_icon="🌾",
        layout="wide",
    )

    st.title("🌾 AgriStat Complete - Enterprise Agricultural Statistics")
    st.caption("CRD | RCBD | Latin Square | Split-Plot | Lattice | Factorial")

    # --- SIDEBAR ---
    with st.sidebar:
        st.header("Configuration")
        uploaded_file = st.file_uploader(
            "Upload Dataset (CSV / Excel)",
            type=["csv", "xlsx", "xls"],
        )
        st.divider()
        module = st.radio(
            "Select Analysis Module:",
            [
                "Data Preview",
                "Descriptive Statistics",
                "CRD ANOVA",
                "RCBD ANOVA",
                "Latin Square ANOVA",
                "Split-Plot ANOVA",
                "Lattice ANOVA",
                "Factorial ANOVA",
            ],
        )
        st.divider()
        alpha = st.slider("Significance Level (alpha)", 0.01, 0.10, 0.05, 0.01)

    # --- NO FILE UPLOADED ---
    if not uploaded_file:
        st.info("Upload a dataset in the sidebar to begin analysis.")
        st.markdown("""
| Design | Module |
|---|---|
| Completely Randomized Design | CRD ANOVA |
| Randomized Complete Block Design | RCBD ANOVA |
| Latin Square Design | Latin Square ANOVA |
| Split-Plot Design | Split-Plot ANOVA |
| Lattice Design (Intra-block) | Lattice ANOVA |
| Factorial (2+ factors, optional block) | Factorial ANOVA |
        """)
        return

    # --- LOAD FILE ---
    raw_df, load_err = DataProcessor.load_file(uploaded_file)
    if load_err:
        st.error(load_err)
        return

    df, col_map = DataProcessor.sanitize_headers(raw_df)
    numeric_cols, _ = DataProcessor.get_column_types(df)
    all_cols = df.columns.tolist()

    if not numeric_cols:
        st.error(
            "No numeric columns found. "
            "Please check that your data contains a numeric response variable."
        )
        return

    # =================================================================
    # MODULE: DATA PREVIEW
    # =================================================================
    if module == "Data Preview":
        st.subheader("Data Preview and Validation")
        st.write(f"Shape: {df.shape[0]} rows x {df.shape[1]} columns")
        st.dataframe(df.head(100), use_container_width=True)
        missing = df.isnull().sum()
        missing = missing[missing > 0]
        if not missing.empty:
            st.warning("Missing values detected:")
            st.dataframe(
                missing.rename("Missing Count").reset_index(),
                use_container_width=True,
            )
        else:
            st.success("No missing values detected.")

    # =================================================================
    # MODULE: DESCRIPTIVE STATISTICS
    # =================================================================
    elif module == "Descriptive Statistics":
        st.subheader("Descriptive Statistics")
        sel = st.multiselect(
            "Select variables to summarise:",
            numeric_cols,
            default=numeric_cols[:4],
        )
        if sel:
            try:
                result = DescriptiveStats.compute_summary(df, sel)
                if not result.empty:
                    st.dataframe(result, use_container_width=True)
                    st.caption(
                        "Shapiro-Wilk p > 0.05 suggests normality. "
                        "CV% below 20 indicates acceptable experimental precision."
                    )
                else:
                    st.warning("No summary could be computed.")
            except Exception as e:
                st.error(f"Descriptive Stats Error: {e}")

    # =================================================================
    # MODULE: CRD ANOVA
    # =================================================================
    elif module == "CRD ANOVA":
        st.subheader("Completely Randomized Design (CRD) ANOVA")
        st.caption(
            "Suitable when all experimental units are homogeneous "
            "and no blocking structure is needed."
        )
        resp = st.selectbox("Response Variable (Y):", numeric_cols, key="crd_resp")
        trt  = st.selectbox(
            "Treatment Factor:",
            [c for c in all_cols if c != resp],
            key="crd_trt",
        )

        if st.button("Run CRD Analysis", key="run_crd"):
            try:
                anova_tb, means_tb, sem, cd, cv, p_val = InferentialStats.run_crd(
                    df, resp, trt, alpha
                )
                st.success("CRD ANOVA complete.")
                _render_precision_metrics(sem, cd, cv)
                st.write("### ANOVA Table")
                _render_anova_table(anova_tb)
                st.write("### Treatment Means")
                st.dataframe(means_tb, use_container_width=True)
                st.write("### Post-Hoc Comparison (Tukey HSD)")
                _render_post_hoc(df, resp, trt, alpha, p_val)
            except Exception as e:
                st.error(f"CRD Analysis Error: {e}")

    # =================================================================
    # MODULE: RCBD ANOVA
    # =================================================================
    elif module == "RCBD ANOVA":
        st.subheader("Randomized Complete Block Design (RCBD) ANOVA")
        st.caption(
            "Accounts for one-directional field heterogeneity "
            "by grouping experimental units into blocks."
        )
        resp = st.selectbox("Response Variable (Y):", numeric_cols, key="rcbd_resp")
        trt  = st.selectbox(
            "Treatment Factor:",
            [c for c in all_cols if c != resp],
            key="rcbd_trt",
        )
        blk  = st.selectbox(
            "Block Factor:",
            [c for c in all_cols if c not in [resp, trt]],
            key="rcbd_blk",
        )

        if st.button("Run RCBD Analysis", key="run_rcbd"):
            try:
                anova_tb, means_tb, sem, cd, cv = InferentialStats.run_rcbd(
                    df, resp, trt, blk, alpha
                )
                st.success("RCBD ANOVA complete.")
                _render_precision_metrics(sem, cd, cv)
                st.write("### ANOVA Table")
                _render_anova_table(anova_tb)
                st.write("### Treatment Means")
                st.dataframe(means_tb, use_container_width=True)
            except Exception as e:
                st.error(f"RCBD Analysis Error: {e}")

    # =================================================================
    # MODULE: LATIN SQUARE ANOVA
    # =================================================================
    elif module == "Latin Square ANOVA":
        st.subheader("Latin Square Design ANOVA")
        st.caption(
            "Simultaneously controls two orthogonal sources of variation "
            "(rows and columns) in field experiments."
        )
        resp    = st.selectbox("Response Variable (Y):", numeric_cols, key="ls_resp")
        trt     = st.selectbox(
            "Treatment Factor:",
            [c for c in all_cols if c != resp],
            key="ls_trt",
        )
        row_blk = st.selectbox(
            "Row Block:",
            [c for c in all_cols if c not in [resp, trt]],
            key="ls_row",
        )
        col_blk = st.selectbox(
            "Column Block:",
            [c for c in all_cols if c not in [resp, trt, row_blk]],
            key="ls_col",
        )

        if st.button("Run Latin Square Analysis", key="run_lsd"):
            try:
                anova_tb, means_tb, sem, cd, cv = InferentialStats.run_latin_square(
                    df, resp, trt, row_blk, col_blk, alpha
                )
                st.success("Latin Square ANOVA complete.")
                _render_precision_metrics(sem, cd, cv)
                st.write("### ANOVA Table")
                _render_anova_table(anova_tb)
                st.write("### Treatment Means")
                st.dataframe(means_tb, use_container_width=True)
            except Exception as e:
                st.error(f"Latin Square Analysis Error: {e}")

    # =================================================================
    # MODULE: SPLIT-PLOT ANOVA
    # =================================================================
    elif module == "Split-Plot ANOVA":
        st.subheader("Split-Plot Design ANOVA")
        st.caption(
            "Main plot factor tested against main plot error. "
            "Sub plot factor and interaction tested against sub plot error."
        )
        resp   = st.selectbox("Response Variable (Y):", numeric_cols, key="sp_resp")
        rep    = st.selectbox(
            "Replication / Block:",
            [c for c in all_cols if c != resp],
            key="sp_rep",
        )
        main_f = st.selectbox(
            "Main Plot Factor:",
            [c for c in all_cols if c not in [resp, rep]],
            key="sp_main",
        )
        sub_f  = st.selectbox(
            "Sub Plot Factor:",
            [c for c in all_cols if c not in [resp, rep, main_f]],
            key="sp_sub",
        )

        if st.button("Run Split-Plot Analysis", key="run_splitplot"):
            try:
                (
                    anova_tb,
                    main_means,
                    sub_means,
                    interaction_means,
                    sem_main, cd_main, cv_main,
                    sem_sub,  cd_sub,  cv_sub,
                ) = InferentialStats.run_split_plot(df, resp, rep, main_f, sub_f)

                st.success("Split-Plot ANOVA complete.")

                st.write("### Precision Metrics - Main Plot")
                _render_precision_metrics(sem_main, cd_main, cv_main)

                st.write("### Precision Metrics - Sub Plot")
                _render_precision_metrics(sem_sub, cd_sub, cv_sub)

                st.write("### ANOVA Table")
                st.dataframe(anova_tb, use_container_width=True)

                st.write(f"### Main Plot Means - {main_f}")
                st.dataframe(main_means, use_container_width=True)

                st.write(f"### Sub Plot Means - {sub_f}")
                st.dataframe(sub_means, use_container_width=True)

                st.write("### Interaction Means")
                st.dataframe(interaction_means, use_container_width=True)

            except Exception as e:
                st.error(f"Split-Plot Analysis Error: {e}")

    # =================================================================
    # MODULE: LATTICE ANOVA
    # =================================================================
    elif module == "Lattice ANOVA":
        st.subheader("Lattice Design ANOVA")
        st.caption(
            "Intra-block analysis for incomplete block designs. "
            "Relative efficiency vs RCBD is automatically computed."
        )
        resp = st.selectbox("Response Variable (Y):", numeric_cols, key="lat_resp")
        trt  = st.selectbox(
            "Treatment Factor:",
            [c for c in all_cols if c != resp],
            key="lat_trt",
        )
        rep  = st.selectbox(
            "Replication:",
            [c for c in all_cols if c not in [resp, trt]],
            key="lat_rep",
        )
        blk  = st.selectbox(
            "Incomplete Block:",
            [c for c in all_cols if c not in [resp, trt, rep]],
            key="lat_blk",
        )

        if st.button("Run Lattice Analysis", key="run_lattice"):
            try:
                anova_tb, means_tb, rel_eff, cv, sem, cd = InferentialStats.run_lattice(
                    df, resp, trt, rep, blk
                )
                st.success("Lattice ANOVA complete.")

                st.write("### Precision Metrics")
                _render_precision_metrics(sem, cd, cv)

                st.write("### Design Efficiency")
                st.metric(
                    "Relative Efficiency vs RCBD (%)",
                    f"{rel_eff:.2f}",
                )
                st.caption(
                    "Relative efficiency above 100% confirms that lattice blocking "
                    "improved precision over a standard RCBD."
                )

                st.write("### ANOVA Table")
                _render_anova_table(anova_tb)

                st.write("### Treatment Means")
                st.dataframe(means_tb, use_container_width=True)

            except Exception as e:
                st.error(f"Lattice Analysis Error: {e}")

    # =================================================================
    # MODULE: FACTORIAL ANOVA
    # =================================================================
    elif module == "Factorial ANOVA":
        st.subheader("Factorial Design ANOVA")
        st.caption(
            "Analyses main effects and all interaction effects "
            "for two or more treatment factors. "
            "Optional blocking supports factorial-in-RCBD."
        )
        resp         = st.selectbox("Response Variable (Y):", numeric_cols, key="fac_resp")
        factor_pool  = [c for c in all_cols if c != resp]
        factors      = st.multiselect(
            "Select Treatment Factors (minimum 2):",
            factor_pool,
            default=factor_pool[:2] if len(factor_pool) >= 2 else factor_pool,
            key="fac_factors",
        )
        block_pool   = ["<none>"] + [c for c in all_cols if c not in [resp] + factors]
        block_sel    = st.selectbox(
            "Optional Block / Replication Factor:",
            block_pool,
            key="fac_block",
        )
        block = None if block_sel == "<none>" else block_sel

        if st.button("Run Factorial Analysis", key="run_factorial"):
            try:
                if len(factors) < 2:
                    st.warning("Please select at least two treatment factors.")
                else:
                    anova_tb, means_tb = InferentialStats.run_factorial(
                        df, resp, factors, block
                    )
                    st.success("Factorial ANOVA complete.")
                    st.write("### ANOVA Table")
                    _render_anova_table(anova_tb)
                    st.write("### Cell Means Table")
                    st.dataframe(means_tb, use_container_width=True)
            except Exception as e:
                st.error(f"Factorial Analysis Error: {e}")


# --- ENTRY POINT ---
if __name__ == "__main__":
    main()
