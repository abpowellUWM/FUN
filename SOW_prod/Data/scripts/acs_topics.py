# SOW_prod/Data/scripts/acs_topics.py
#
# ACS 5-year Census pull -> the 15 production topic tables, written straight to
# Data/Production/ACS/ (one CSV per table, ready for Tableau). Each table is one
# line in TOPICS at the bottom. Replaces the old build_acs.py -> acs.csv -> SQL
# (DB Browser) chain.
#
# Pulls live from the Census API every run -- every year in YEARS (2020-2024).
#   export CENSUS_API_KEY=...   (free key: https://api.census.gov/data/key_signup.html)
#   python3 Data/scripts/acs_topics.py
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
from standards import (RACE_ACS, SEX, STANDARD_VALUES, check_values,  # noqa: E402
                       county_columns, map_strict)

# ACS 5-year vintages to pull: 2020 = survey years 2016-2020 ... 2024 = 2020-2024.
# Add the next year here when Census releases it (2025 5-year: expected Dec 2026).
# Consecutive 5-year periods overlap by 4 years -- compare non-overlapping
# years (e.g. 2019 vs 2024) for change over time. Earnings are in each
# year's own dollars (not inflation-adjusted across years).
YEARS = list(range(2020, 2025))
STATE_FIPS = "55"
API_KEY = (os.getenv("CENSUS_API_KEY") or "").strip().strip("'\"")  # tolerate stray spaces/quotes


def detailed_url(year):
    return f"https://api.census.gov/data/{year}/acs/acs5"


def subject_url(year):
    return f"https://api.census.gov/data/{year}/acs/acs5/subject"


OUT_DIR = Path(__file__).resolve().parents[1] / "Production" / "ACS"   # Data/Production/ACS
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ===========================================================================
# VARIABLE LIST
# ===========================================================================
def seq(prefix, start, end):
    return [f"{prefix}_{i:03d}" for i in range(start, end + 1)]


# Race A (White alone, includes Hispanic) is not pulled: the dashboard's White
# group is H (White, non-Hispanic).
def race_vars(table_prefix, rows):
    return [f"{table_prefix}{s}_{row:03d}" for s in "BCDEFGHI" for row in rows]


VAR_BASES = []
VAR_BASES += seq("B01001", 1, 49)
# DECISION B1 fix: row 17 is the true "Female:" total (row 26 is "Female:
# 35 to 44 years" -- verified against the live acs5 API). The old pull
# requested 26, mislabeling an age slice as the sex total.
VAR_BASES += race_vars("B01001", [1, 2, 17])   # 001 = each race's Total (both sexes)
VAR_BASES += seq("B12001", 1, 19)
VAR_BASES += seq("B15002", 1, 35)
VAR_BASES += seq("B20004", 2, 18)
# 001 universe; 002/003/017 below poverty (Total/Men/Women); 031/032/046 at or above (Total/Men/Women)
VAR_BASES += ["B17001_001", "B17001_002", "B17001_003", "B17001_017", "B17001_031", "B17001_032", "B17001_046"]
VAR_BASES += race_vars("B17001", [1, 2, 3, 17, 31, 32, 46])
# DECISION B4 fix: added _002/_005 (true all-workers medians, never pulled
# before) alongside the existing _003/_006 (full-time year-round medians).
VAR_BASES += ["B20017_002", "B20017_003", "B20017_005", "B20017_006"]
VAR_BASES += race_vars("B20017", [2, 3, 5, 6])
VAR_BASES += ["B22002_001", "B22002_002", "B22002_006", "B22002_007",
              "B22002_012", "B22002_013", "B22002_019", "B22002_020",
              "B22002_025", "B22002_026"]

# DECISION B3/B5 fix: pull each age bracket's "In labor force" TOTAL row
# directly, for both sexes, all the way through 75+ -- verified row-by-row
# against the live API. The old pull assumed a uniform row stride between
# brackets; the real table narrows starting at 65-69 (no Armed Forces/
# Civilian sub-rows for older ages), which the old stride math didn't
# account for, so it grabbed the wrong rows for 70-74 and never pulled 75+
# at all (male), and pulled 70-74's row believing it was 65-69 (female).
# Pulling the direct total per bracket -- instead of the old
# (bracket-total, not-in-labor-force) pair meant for subtraction -- also
# means no subtraction logic is needed downstream for either sex anymore.
# 2026-11-19: each bracket's population total (the row just before its "In
# labor force" row) is pulled too, so participation by age = In labor force /
# Population 16+ on the same row. 001 = both sexes, all ages.
MALE_LF = (4, 11, 18, 25, 32, 39, 46, 53, 60, 67, 74, 79, 84)            # Male, 13 brackets
FEMALE_LF = (90, 97, 104, 111, 118, 125, 132, 139, 146, 153, 160, 165, 170)  # Female, 13 brackets
VAR_BASES += ["B23001_001", "B23001_002", "B23001_088"]  # totals (denominators)
VAR_BASES += [f"B23001_{r:03d}" for lf in MALE_LF + FEMALE_LF for r in (lf - 1, lf)]  # bracket total + in labor force

# C23002A-I: NOT curated (see module docstring / plan doc D-out-of-scope
# note). Pulled unchanged from the original script -- every row, because
# which of the 30 rows per race suffix means what has never been derived.
# dim_demographic tags these rows "Employment / Labor Force" like the rest
# of B23001, but no LFP measure should be built on them until that
# curation happens.
for suffix in "BCDEFGHI":
    VAR_BASES += [f"C23002{suffix}_{i:03d}" for i in range(1, 31)]

# 001 all; 002/013/024/035/046 = each marital status, both sexes (Sex = Total rows)
VAR_BASES += ["B12006_001", "B12006_002", "B12006_013", "B12006_024", "B12006_035", "B12006_046",
              "B12006_003", "B12006_004", "B12006_014", "B12006_015",
              "B12006_025", "B12006_026", "B12006_036", "B12006_037",
              "B12006_047", "B12006_048",
              "B12006_008", "B12006_009", "B12006_019", "B12006_020",
              "B12006_030", "B12006_031", "B12006_041", "B12006_042",
              "B12006_052", "B12006_053"]
# Not pulled (no topic table uses them): B05003 citizenship, B20005, B22010,
# B24022, B27001 insurance, B13002 births, B25003/B25064/B25070 housing.
VAR_BASES = list(dict.fromkeys(VAR_BASES))  # de-dupe, preserve order

SUBJECT_GROUPS = ["S2411", "S2412"]

# C24010 (occupation worker counts) -- folded in from occupation_pull.py.
C24010_VARS = (
    ["C24010_001", "C24010_002",
     "C24010_003", "C24010_019", "C24010_027", "C24010_030", "C24010_034",
     "C24010_038",
     "C24010_039", "C24010_055", "C24010_063", "C24010_066", "C24010_070",
     "C24010_004", "C24010_007", "C24010_011", "C24010_016",
     "C24010_020", "C24010_021", "C24010_024", "C24010_025", "C24010_026",
     "C24010_028", "C24010_029",
     "C24010_031", "C24010_032", "C24010_033",
     "C24010_035", "C24010_036", "C24010_037",
     "C24010_040", "C24010_043", "C24010_047", "C24010_052",
     "C24010_056", "C24010_057", "C24010_060", "C24010_061", "C24010_062",
     "C24010_064", "C24010_065",
     "C24010_067", "C24010_068", "C24010_069",
     "C24010_071", "C24010_072", "C24010_073"]
    + race_vars("C24010", range(1, 14))
)
C24010_VARS = list(dict.fromkeys(C24010_VARS))

C24010_BASE_ROWS = {
    "002": ("Male", "Total", "sex_total"),
    "003": ("Male", "Management, business, science, and arts occupations", "top"),
    "019": ("Male", "Service occupations", "top"),
    "027": ("Male", "Sales and office occupations", "top"),
    "030": ("Male", "Natural resources, construction, and maintenance occupations", "top"),
    "034": ("Male", "Production, transportation, and material moving occupations", "top"),
    "038": ("Female", "Total", "sex_total"),
    "039": ("Female", "Management, business, science, and arts occupations", "top"),
    "055": ("Female", "Service occupations", "top"),
    "063": ("Female", "Sales and office occupations", "top"),
    "066": ("Female", "Natural resources, construction, and maintenance occupations", "top"),
    "070": ("Female", "Production, transportation, and material moving occupations", "top"),
    "004": ("Male", "Management, business, and financial occupations", "branch"),
    "007": ("Male", "Computer, engineering, and science occupations", "branch"),
    "011": ("Male", "Education, legal, community service, arts, and media occupations", "branch"),
    "016": ("Male", "Healthcare practitioners and technical occupations", "branch"),
    "020": ("Male", "Healthcare support occupations", "leaf"),
    "021": ("Male", "Protective service occupations", "branch"),
    "024": ("Male", "Food preparation and serving related occupations", "leaf"),
    "025": ("Male", "Building and grounds cleaning and maintenance occupations", "leaf"),
    "026": ("Male", "Personal care and service occupations", "leaf"),
    "028": ("Male", "Sales and related occupations", "leaf"),
    "029": ("Male", "Office and administrative support occupations", "leaf"),
    "031": ("Male", "Farming, fishing, and forestry occupations", "leaf"),
    "032": ("Male", "Construction and extraction occupations", "leaf"),
    "033": ("Male", "Installation, maintenance, and repair occupations", "leaf"),
    "035": ("Male", "Production occupations", "leaf"),
    "036": ("Male", "Transportation occupations", "leaf"),
    "037": ("Male", "Material moving occupations", "leaf"),
    "040": ("Female", "Management, business, and financial occupations", "branch"),
    "043": ("Female", "Computer, engineering, and science occupations", "branch"),
    "047": ("Female", "Education, legal, community service, arts, and media occupations", "branch"),
    "052": ("Female", "Healthcare practitioners and technical occupations", "branch"),
    "056": ("Female", "Healthcare support occupations", "leaf"),
    "057": ("Female", "Protective service occupations", "branch"),
    "060": ("Female", "Food preparation and serving related occupations", "leaf"),
    "061": ("Female", "Building and grounds cleaning and maintenance occupations", "leaf"),
    "062": ("Female", "Personal care and service occupations", "leaf"),
    "064": ("Female", "Sales and related occupations", "leaf"),
    "065": ("Female", "Office and administrative support occupations", "leaf"),
    "067": ("Female", "Farming, fishing, and forestry occupations", "leaf"),
    "068": ("Female", "Construction and extraction occupations", "leaf"),
    "069": ("Female", "Installation, maintenance, and repair occupations", "leaf"),
    "071": ("Female", "Production occupations", "leaf"),
    "072": ("Female", "Transportation occupations", "leaf"),
    "073": ("Female", "Material moving occupations", "leaf"),
}
C24010_RACE_ROWS = {
    "002": ("Male", "Total", "sex_total"),
    "003": ("Male", "Management, business, science, and arts occupations", "top"),
    "004": ("Male", "Service occupations", "top"),
    "005": ("Male", "Sales and office occupations", "top"),
    "006": ("Male", "Natural resources, construction, and maintenance occupations", "top"),
    "007": ("Male", "Production, transportation, and material moving occupations", "top"),
    "008": ("Female", "Total", "sex_total"),
    "009": ("Female", "Management, business, science, and arts occupations", "top"),
    "010": ("Female", "Service occupations", "top"),
    "011": ("Female", "Sales and office occupations", "top"),
    "012": ("Female", "Natural resources, construction, and maintenance occupations", "top"),
    "013": ("Female", "Production, transportation, and material moving occupations", "top"),
}

# ===========================================================================
# FETCH HELPERS -- kept close to pull_acs.py verbatim (no confirmed bugs)
# ===========================================================================
def chunk_list(items, size):
    for i in range(0, len(items), size):
        yield items[i:i + size]


def get_json(url, params=None, timeout=120):
    # Never print the request URL or params: they contain the API key.
    r = requests.get(url, params=params, timeout=timeout)
    if "Invalid Key" in r.text[:500]:
        # Census answers a bad key with a redirect to an HTML "Invalid Key"
        # page (status 200), not an error code.
        raise SystemExit(
            "Census rejected CENSUS_API_KEY (\"Invalid Key\"). Check that:\n"
            "  1. you clicked the activation link in the Census email (new keys don't work until then),\n"
            "  2. the key was pasted exactly -- no quotes, spaces or placeholder text,\n"
            "  3. you ran `export CENSUS_API_KEY=...` in the same terminal tab as this script.")
    if not r.ok:
        raise SystemExit(f"Census API error {r.status_code} from {url}: {r.text[:300]}")
    try:
        return r.json()
    except ValueError:
        raise SystemExit(f"Census API sent a non-data response from {url} "
                         f"(status {r.status_code}): {r.text[:200]!r}")


def fetch_acs(get_vars, geography_for, geography_in, base_url):
    params = {"get": ",".join(["NAME"] + get_vars), "for": geography_for, "in": geography_in, "key": API_KEY}
    rows = get_json(base_url, params=params, timeout=180)
    return pd.DataFrame(rows[1:], columns=rows[0])


def wide_to_tidy(df, year):
    geo_cols = [c for c in ["NAME", "state", "county"] if c in df.columns]
    value_cols = [c for c in df.columns if c not in geo_cols]
    long = df.melt(id_vars=geo_cols, value_vars=value_cols, var_name="variable", value_name="value")
    long["stat"] = long["variable"].str[-1].map({"E": "estimate", "M": "moe"})
    long["var_base"] = long["variable"].str[:-1]
    long = long[long["stat"].isin(["estimate", "moe"])]
    tidy = long.pivot_table(index=geo_cols + ["var_base"], columns="stat", values="value", aggfunc="first").reset_index()
    tidy.columns.name = None
    tidy["estimate"] = pd.to_numeric(tidy.get("estimate"), errors="coerce")
    tidy["moe"] = pd.to_numeric(tidy.get("moe"), errors="coerce")
    # Census sentinels for "not computed" -- convert to real nulls so they
    # can't leak into SUM()s as billions/negative-billions (confirmed bug:
    # education/race charts showing huge negative bars).
    tidy.loc[tidy["estimate"] == -666666666, "estimate"] = pd.NA
    tidy.loc[tidy["moe"].isin([-222222222, -333333333, -555555555, -666666666]), "moe"] = pd.NA
    tidy["year"] = year
    tidy["geoid"] = pd.to_numeric(tidy["state"] + tidy["county"])
    return tidy


# ===========================================================================
# VARIABLE LABELING (topic / sex / race / age / education / marital / poverty)
# DECISION B9/D9 fix: build_dims.py depended on CONTEXT/variable_mapping.txt,
# which does not exist anywhere in the project -- the old pipeline could not
# actually be run. Topic/sex/age/race/poverty_status are derived here
# directly from each variable's own Census label text, in-script, so there
# is no external file that can go missing again.
# ===========================================================================

# DECISION D1 fix: every Census race iteration gets its own distinct label.
# The old pipeline collapsed White-alone (A) and White-non-Hispanic (H) into
# one "White" label, and American Indian/Alaska Native (C) and Native
# Hawaiian/Pacific Islander (E) into one "Other" -- a real representation
# gap on a dashboard about demographic equity. This was intentional and
# documented in the old code, not a hidden bug, but the recommendation was
# to split it, so it's fixed here.
RACE_LABEL = {
    "A": "White alone", "B": "Black or African American", "C": "American Indian/Alaska Native",
    "D": "Asian", "E": "Native Hawaiian/Pacific Islander", "F": "Some Other Race",
    "G": "Two or More Races", "H": "White, non-Hispanic", "I": "Hispanic or Latino",
}

# Marital status is assigned by explicit Census row number, not by substring
# keyword match. The old keyword approach was fragile and had two confirmed
# bugs: (1) the real "Now married:" total row (B12001 _004/_013) never matched
# the assumed "Now married, except separated" string, so the true Married
# total fell through to "All" and the Married bar was built entirely from the
# nested "spouse present/absent" sub-rows -- double-counting the spouse-absent
# subtotal (+60,022); (2) B12006 has the same label mismatch. Row numbers are
# stable across the ACS vintage and unambiguous, matching the POVERTY_ROW
# pattern used below.
#
# B12001 (Marital Status by Sex, pop 15+): only the four mutually-exclusive
# top-level categories per sex are tagged; every nested sub-row (spouse
# present/absent, separated, other) and the sex/grand totals stay "All" so
# they're excluded from the marital breakdown.
B12001_MARITAL_ROW = {
    "003": "Never married", "004": "Married", "009": "Widowed", "010": "Divorced",  # Male
    "012": "Never married", "013": "Married", "018": "Widowed", "019": "Divorced",  # Female
}
# B12006 (Marital Status by Sex by Labor Force) -> labor_force_marital_status.
# "Married" here excludes separated people (Separated is its own value).
B12006_MARITAL_ROW = {
    "002": "Never married", "003": "Never married", "004": "Never married",
    "008": "Never married", "009": "Never married",
    "013": "Married (not separated)", "014": "Married (not separated)", "015": "Married (not separated)",
    "019": "Married (not separated)", "020": "Married (not separated)",
    "024": "Separated", "025": "Separated", "026": "Separated", "030": "Separated", "031": "Separated",
    "035": "Widowed", "036": "Widowed", "037": "Widowed", "041": "Widowed", "042": "Widowed",
    "046": "Divorced", "047": "Divorced", "048": "Divorced", "052": "Divorced", "053": "Divorced",
}
EDU_MAP = [
    ("Less than high school graduate", "Less than high school"), ("Less than high school diploma", "Less than high school"),
    ("No schooling completed", "Less than high school"), ("Nursery to 4th grade", "Less than high school"),
    ("5th and 6th grade", "Less than high school"), ("7th and 8th grade", "Less than high school"),
    ("9th grade", "Less than high school"), ("10th grade", "Less than high school"),
    ("11th grade", "Less than high school"), ("12th grade, no diploma", "Less than high school"),
    ("High school graduate", "High school graduate"),
    ("Some college, less than 1 year", "Some college"), ("Some college, 1 or more years, no degree", "Some college"),
    ("Some college or associate's degree", "Some college or associate's degree"),  # B20004 only
    ("Associate's degree", "Associate's degree"),
    ("Bachelor's degree", "Bachelor's degree"), ("Graduate or professional degree", "Graduate or professional degree"),
    ("Master's degree", "Master's degree"), ("Professional school degree", "Professional school degree"),
    ("Doctorate degree", "Doctorate degree"),
]

# DECISION D6 fix: same 7-bucket scheme as the crime pipeline (was 6 buckets
# with "45-64" merged and a mislabeled "35-45" -- ages 35-44 were shown
# under a label implying age 45 was included).
AGE_BINS = [0, 17, 24, 34, 44, 54, 64, 200]
AGE_LABELS = ["Under 18", "18-24", "25-34", "35-44", "45-54", "55-64", "65+"]
AGE_SORT = {label: i + 1 for i, label in enumerate(AGE_LABELS)}
EDU_ORDER = ["Less than high school", "High school graduate", "Some college", "Associate's degree",
             "Bachelor's degree", "Graduate or professional degree", "Master's degree",
             "Professional school degree", "Doctorate degree"]
EDU_SORT = {label: i + 1 for i, label in enumerate(EDU_ORDER)}
EDU_SORT["Some college or associate's degree"] = 3

# Sort orders for the topic tables' "X sort" columns
SEX_SORT = {"Women": 1, "Men": 2, "Total": 3}
RACE_SORT = {"All": 0, "White (non-Hispanic)": 1, "Black": 2, "Hispanic": 3, "Asian": 4,
             "AIAN": 5, "NHPI": 6, "Other": 7, "Two or more": 8}
MARITAL_SORT = {"All": 0, "Never married": 1, "Married": 2, "Married (not separated)": 2,
                "Separated": 3, "Widowed": 4, "Divorced": 5}


# A label piece counts as an age only when the WHOLE piece is an age band
# ("Under 5 years", "20 to 24 years", "65 years and over"). Pieces that merely
# contain a number and the word "year" ("Worked full-time, year-round ...",
# "...16 years and over with earnings") are not ages.
_AGE_SEG = re.compile(r"^(under \d+ years|\d+ (to|and) \d+ years|\d+ years( and over)?)$", re.I)


def _band_range(part):
    """(youngest, oldest) age a Census age band covers."""
    nums = [int(n) for n in re.findall(r"\d+", part)]
    if part.lower().startswith("under "):
        return 0, nums[0] - 1
    if "and over" in part.lower():
        return nums[0], 200
    return nums[0], nums[-1]


def _age_group(label):
    """The standard age group a variable belongs to -- only when the Census age band sits
    entirely inside ONE standard group. A band that spans several ("16 to 64 years",
    "16 to 19 years") is not any single group, so it stays "All"; age_detail keeps the
    Census's own band."""
    for part in label.split("!!"):
        part = part.rstrip(":").strip()
        if not _AGE_SEG.match(part):
            continue
        lo_age, hi_age = _band_range(part)
        for lo, hi, lbl in zip(AGE_BINS, AGE_BINS[1:], AGE_LABELS):
            if (lo < lo_age <= hi or lo_age == lo == 0) and (lo < hi_age <= hi or hi_age == lo == 0):
                return lbl
        return "All"
    return "All"


def _keyword_lookup(label, keyword_map):
    for keyword, value in keyword_map:
        if keyword in label:
            return value
    return "All"


def _sex(label):
    # 2026-11-08: also strip " --" -- B20017 labels read "Male --"/"Female --",
    # so men's and women's median earnings were both tagged sex = "All".
    parts = [p.rstrip(":").replace(" --", "").strip() for p in label.split("!!")]
    if "Male" in parts:
        return "Male"
    if "Female" in parts:
        return "Female"
    return "All"


def label_variables(fact):
    dim = fact[["var_base", "label"]].drop_duplicates("var_base").copy()
    table = dim["var_base"].str.extract(r"^([A-Z]\d+)")[0]
    race_suffix = dim["var_base"].str.extract(r"^[A-Z]\d+([A-I])_")[0]
    row = dim["var_base"].str.extract(r"_(\d{3})$")[0]

    dim["race"] = race_suffix.map(RACE_LABEL).fillna("All")
    dim["sex"] = dim["label"].fillna("").apply(_sex)
    dim["age_group"] = dim["label"].fillna("").apply(_age_group)
    dim["marital_status"] = "All"
    is_b12001 = table.eq("B12001")
    is_b12006 = table.eq("B12006")
    dim.loc[is_b12001, "marital_status"] = row[is_b12001].map(B12001_MARITAL_ROW).fillna("All")
    dim.loc[is_b12006, "marital_status"] = row[is_b12006].map(B12006_MARITAL_ROW).fillna("All")
    dim["education_level"] = [
        _keyword_lookup(lbl, EDU_MAP) if t in ("B15002", "B20004") else "All"
        for t, lbl in zip(table, dim["label"].fillna(""))
    ]
    dim["age_sort"] = dim["age_group"].map(AGE_SORT).fillna(0).astype(int)
    dim["education_sort"] = dim["education_level"].map(EDU_SORT).fillna(0).astype(int)

    return dim[["var_base", "sex", "race", "age_group", "age_sort",
                "marital_status", "education_level", "education_sort"]]


def label_occupation_variables(fact):
    occ = fact[fact["var_base"].str.match(r"^S241[12]_")][["var_base", "label"]].drop_duplicates("var_base").copy()
    parsed = occ["var_base"].str.extract(r"^(S24\d+)_(C0\d)_(\d{3})$")
    occ["table"], occ["col"], occ["row_number"] = parsed[0], parsed[1], parsed[2]
    occ["worker_type"] = occ["table"].map({"S2411": "All civilian workers", "S2412": "Full-time, year-round"})
    occ["sex"] = occ["col"].map({"C01": "Total", "C02": "Male", "C03": "Female"})
    occ["occupation_label"] = occ["label"].fillna("").str.split("!!").str[-1].str.rstrip(":")
    occ["occupation_level"] = "Detail"
    occ["is_median_earnings_row"] = True
    occ["race"] = "All"
    occ["occupation_nesting"] = pd.NA

    counts = fact[fact["var_base"].str.match(r"^C24010")][["var_base"]].drop_duplicates().copy()
    parsed = counts["var_base"].str.extract(r"^C24010([A-I]?)_(\d{3})$")
    counts["race_suffix"], counts["row_number"] = parsed[0].fillna(""), parsed[1]
    is_iter = counts["race_suffix"] != ""
    lookup = counts["row_number"].where(~is_iter).map(C24010_BASE_ROWS).combine_first(
        counts["row_number"].where(is_iter).map(C24010_RACE_ROWS))
    counts["sex"], counts["occupation_label"], counts["occupation_nesting"] = lookup.str[0], lookup.str[1], lookup.str[2]
    counts["race"] = counts["race_suffix"].map(RACE_LABEL).fillna("All")
    counts["table"] = "C24010"
    counts["worker_type"] = "All civilian workers"
    counts["occupation_level"] = pd.NA
    counts["is_median_earnings_row"] = False
    # DECISION D11 fix: row 001 (grand total) has no sex/label -- give it one
    # instead of silently dropping it, so it's a usable universe/denominator
    # row through the dimension relationship instead of an orphan in fact.
    grand_total = counts["sex"].isna()
    counts.loc[grand_total, ["sex", "occupation_label", "occupation_nesting"]] = ["All", "Total", "sex_total"]

    cols = ["var_base", "table", "race", "worker_type", "sex", "row_number",
            "occupation_label", "occupation_level", "occupation_nesting", "is_median_earnings_row"]
    return pd.concat([occ[cols], counts[cols]], ignore_index=True)


# ===========================================================================
# MAIN
# ===========================================================================
def pull_from_api(year):
    if not API_KEY:
        raise SystemExit("CENSUS_API_KEY is not set. Get a free key at https://api.census.gov/data/key_signup.html, "
                         "then run: export CENSUS_API_KEY=your_key")
    geography_for, geography_in = "county:*", f"state:{STATE_FIPS}"

    print(f"\n=== ACS 5-year {year} ===")
    var_meta = get_json(f"{detailed_url(year)}/variables.json")["variables"]
    label_lookup = {k: v["label"] for k, v in var_meta.items() if "label" in v}

    all_vars = VAR_BASES + C24010_VARS
    est_vars = [f"{v}E" for v in all_vars if f"{v}E" in var_meta]
    missing = [v for v in all_vars if f"{v}E" not in var_meta]
    if missing:
        print(f"[WARN] {len(missing)} variables not found in {year} acs5, skipped: {missing}")
    vars_to_get = est_vars + [v[:-1] + "M" for v in est_vars]

    print(f"Pulling {len(all_vars)} base variables...")
    chunks = []
    for i, vc in enumerate(chunk_list(vars_to_get, 40), start=1):
        print(f"  detailed chunk {i}...")
        df = fetch_acs(vc, geography_for, geography_in, detailed_url(year))
        chunks.append(df)
        time.sleep(0.2)
    detailed = chunks[0]
    for d in chunks[1:]:
        detailed = detailed.merge(d, on=[c for c in ["NAME", "state", "county"] if c in detailed.columns], how="left")
    fact = wide_to_tidy(detailed, year)

    subject_var_meta = get_json(f"{subject_url(year)}/variables.json")["variables"]
    subject_label_lookup = {k: v["label"] for k, v in subject_var_meta.items() if "label" in v}
    for group in SUBJECT_GROUPS:
        print(f"  subject group {group}...")
        df = fetch_acs([f"group({group})"], geography_for, geography_in, subject_url(year))
        df = df.loc[:, ~df.columns.duplicated()]
        sub = wide_to_tidy(df, year)
        fact = pd.concat([fact, sub], ignore_index=True)
        label_lookup.update(subject_label_lookup)
        time.sleep(0.2)

    fact["label"] = fact["var_base"].map(lambda vb: label_lookup.get(vb + "E"))
    print(f"Pulled {year}: {fact['var_base'].nunique():,} variables x {fact['geoid'].nunique()} counties")
    return fact[["geoid", "county", "NAME", "year", "var_base", "estimate", "moe", "label"]]


# ===========================================================================
# LABELED ROWS
# One row per county x year x Census variable, with readable columns
# (Sex, Race, Age group, ...) and a `measure` that becomes the topic tables'
# value column name. "All" = not broken out by that column.
# ===========================================================================
# What a row counts when nothing more specific is left in its Census label
# (these are the topic tables' measure column names).
UNIVERSE = {
    "B01001": "Population", "B12001": "Population 15+", "B12006": "Population 16+",
    "B15002": "Population 25+", "B17001": "Poverty universe", "B20004": "Median earnings",
    "B20017": "Median earnings", "B22002": "Households", "B23001": "Population 16+",
    "C23002": "Population 16+", "C24010": "Workers", "S2411": "Median earnings", "S2412": "Median earnings",
}
SUBGROUPS_17 = [  # the dashboard's "Occ Subgroups" list
    "Building and grounds cleaning and maintenance occupations",
    "Computer, engineering, and science occupations", "Construction and extraction occupations",
    "Education, legal, community service, arts, and media occupations",
    "Farming, fishing, and forestry occupations", "Food preparation and serving related occupations",
    "Healthcare practitioners and technical occupations", "Healthcare support occupations",
    "Installation, maintenance, and repair occupations",
    "Management, business, and financial occupations", "Material moving occupations",
    "Office and administrative support occupations", "Personal care and service occupations",
    "Production occupations", "Protective service occupations", "Sales and related occupations",
    "Transportation occupations",
]
MAJOR_GROUPS_5 = [
    "Management, business, science, and arts occupations", "Service occupations",
    "Sales and office occupations", "Natural resources, construction, and maintenance occupations",
    "Production, transportation, and material moving occupations",
]


def _segments(label):
    segs = [x.strip().rstrip(":").replace(" --", "").strip() for x in str(label).split("!!")]
    return [x for x in segs if x and x != "Estimate"]


def _measure(table, label, sex_col, marital, education):
    segs = _segments(label)
    if table.startswith("S24"):
        return UNIVERSE[table]
    if table == "C24010":
        return UNIVERSE[table]
    keep = []
    for x in segs:
        if x.startswith("Total") or x.startswith("Median earnings in the past 12 months"):
            continue
        if x in ("Male", "Female") or _AGE_SEG.match(x) or x.startswith("Worked full-time, year-round"):
            continue
        if table in ("B15002", "B20004") and education != "All":
            continue  # the education_level column carries it
        if table in ("B12001", "B12006") and marital != "All" and x != "In labor force":
            continue  # the marital_status column carries it
        if table == "B22002" and (x.startswith(("With children", "No children", "Other family"))
                                  or "householder" in x):
            continue  # children / household_type / sex columns carry it
        keep.append(x)
    if table == "C23002":  # "In labor force > Civilian > Employed" -> "Employed"
        last = keep[-1] if keep else ""
        return {"": "Population 16+", "Civilian": "Civilian labor force"}.get(last, last)
    path = " > ".join(keep)
    if table == "B17001":
        return ("Below poverty" if "below poverty" in path else
                "At or above poverty" if "at or above poverty" in path else UNIVERSE[table])
    if table == "B22002":
        if "did not receive" in path:
            return "Households not receiving SNAP"
        if "received" in path:
            return "Households receiving SNAP"
    return path or UNIVERSE.get(table, "Total")


def check_years_match(raw):
    """Every year must cover all 72 counties, and a variable must mean the same
    thing in every year (same Census label, ignoring the dollar year), because
    all the labeling below is done once, from the latest year's labels."""
    counties = raw.groupby("year")["geoid"].nunique()
    if (counties != 72).any():
        raise ValueError(f"Expected 72 counties every year, got {counties.to_dict()}")
    norm = (raw.drop_duplicates(["year", "var_base"])
            .assign(norm=lambda d: d["label"].fillna("")
                    .str.replace(r"\(in \d{4} inflation-adjusted dollars\)", "(in YYYY dollars)", regex=True)
                    .str.replace(":", "").str.replace(r"\s+", " ", regex=True).str.strip()))
    differ = norm.groupby("var_base")["norm"].nunique()
    if (differ > 1).any():
        bad = norm[norm["var_base"].isin(differ[differ > 1].index)].sort_values(["var_base", "year"])
        print(bad[["var_base", "year", "label"]].to_string())
        raise ValueError(f"{(differ > 1).sum()} Census variable(s) changed meaning between years")
    per_year = norm.groupby("var_base")["year"].nunique()
    partial = per_year[per_year < raw["year"].nunique()]
    if len(partial):
        print(f"[WARN] {len(partial)} variable(s) missing in some years: {sorted(partial.index)[:20]}")


def drop_ratio_rows(raw):
    """Remove the Census's pre-computed "Women's earnings as a percentage of men's" column
    (column C04 of S2411/S2412). It is a ratio of two medians that are already in the table
    (columns C02 and C03), and this file holds source values only -- no calculated rows."""
    is_c04 = raw["var_base"].str.match(r"^S241[12]_C04_")
    ratio_label = raw["label"].fillna("").str.contains("percentage of men", case=False)
    if not (is_c04 == ratio_label).all():
        raise ValueError("S2411/S2412 column C04 no longer lines up with the 'percentage of men' label -- check before dropping")
    print(f"Dropped {int(is_c04.sum()):,} pre-computed ratio rows (S2411/S2412 column C04)")
    return raw[~is_c04]


def build_acs_table(raw):
    check_years_match(raw)
    raw = drop_ratio_rows(raw)
    latest = raw[raw["year"] == raw["year"].max()]
    dim = label_variables(latest).set_index("var_base")
    occ = label_occupation_variables(latest).set_index("var_base")

    v = (raw.sort_values("year", ascending=False)[["var_base", "label"]]
         .drop_duplicates("var_base").set_index("var_base"))
    v["table_id"] = v.index.str.extract(r"^([A-Z]\d+)", expand=False)
    v["race"] = map_strict(dim["race"].reindex(v.index), RACE_ACS, "ACS race")
    v["sex"] = map_strict(dim["sex"].reindex(v.index), SEX, "ACS sex")
    is_occ = v.index.isin(occ.index)
    occ_sex = occ["sex"].replace({"Female % of Male": "Female"})
    v.loc[is_occ, "sex"] = map_strict(occ_sex.reindex(v.index[is_occ]), SEX, "occupation sex")
    v.loc[is_occ, "race"] = map_strict(occ["race"].reindex(v.index[is_occ]), RACE_ACS, "occupation race")
    lbl = v["label"].fillna("")
    v.loc[lbl.str.contains("Female householder"), "sex"] = "Women"
    v.loc[lbl.str.contains("Male householder"), "sex"] = "Men"

    for col in ["age_group", "age_sort", "education_level", "education_sort", "marital_status"]:
        v[col] = dim[col].reindex(v.index)
    # Census's own age band ("15 to 17 years"), for calcs the 7 standard
    # groups can't express -- e.g. women 15-44 for the birth rate.
    v["age_detail"] = [next((x for x in _segments(l) if _AGE_SEG.match(x)), "All") for l in lbl]
    v.loc[is_occ, ["age_group", "education_level", "marital_status"]] = "All"
    v.loc[is_occ, ["age_sort", "education_sort"]] = 0

    v["children"] = np.select([lbl.str.contains("With children under 18"), lbl.str.contains("No children under 18")],
                              ["With children", "No children"], "All")

    occ_label = occ["occupation_label"].reindex(v.index)
    v["occupation"] = np.where(is_occ, occ_label.replace({"Total": "All occupations"}), "All")
    v.loc[is_occ & occ_label.str.contains("population 16 years and over with earnings", na=False), "occupation"] = "All occupations"
    v["occupation_level"] = np.select(
        [~is_occ, v["occupation"].eq("All occupations"), v["occupation"].isin(MAJOR_GROUPS_5),
         v["occupation"].isin(SUBGROUPS_17)],
        ["All", "Total", "Major group", "Subgroup"], "Detail")

    ftyr = lbl.str.contains("Worked full-time, year-round") | v["table_id"].eq("S2412")
    all_workers = v["table_id"].isin(["S2411", "C24010", "B20004"]) | (v["table_id"].eq("B20017") & ~ftyr)
    v["worker_type"] = np.select([ftyr, all_workers], ["Full-time, year-round", "All workers"], "All")

    v["measure"] = [_measure(t, l, s, m, e) for t, l, s, m, e in
                    zip(v["table_id"], v["label"], v["sex"], v["marital_status"], v["education_level"])]
    out = raw[["geoid", "year", "var_base", "estimate", "moe", "label"]].merge(
        v.drop(columns="label"), left_on="var_base", right_index=True)
    # census_label keeps each year's own wording (e.g. "in 2021 inflation-adjusted dollars")
    out["census_label"] = (out["label"].fillna("").str.replace(r"^Estimate!!", "", regex=True)
                           .str.replace("!!", " > ").str.replace(":", ""))
    out["county"], out["county_fips"] = county_columns(out["geoid"])
    out = out.rename(columns={"var_base": "var_code"})
    out = out[["county", "county_fips", "year", "measure", "sex", "race", "age_group", "age_sort", "age_detail",
               "education_level", "education_sort", "marital_status", "children",
               "occupation", "occupation_level", "worker_type", "estimate", "moe",
               "table_id", "var_code", "census_label"]]
    out[["age_sort", "education_sort"]] = out[["age_sort", "education_sort"]].astype(int)

    check_values(out, {**STANDARD_VALUES, "race": list(RACE_ACS.values()),
                       "sex": ["Women", "Men", "Total"]}, "acs")
    if out["county"].eq("Unknown").any():
        raise ValueError("ACS rows without a county")

    # Topic-table columns: Census table with race suffix, education detail, and the sort orders
    out["census_table"] = out["var_code"].str.split("_").str[0]          # B17001B, S2411
    is_edu = out["table_id"].eq("B15002") & out["education_level"].ne("All")
    out["education_detail"] = np.where(is_edu, out["census_label"].str.split(" > ").str[-1], "All")
    out["education_detail_sort"] = np.where(is_edu, out["var_code"].str[-3:].astype(int), 0)
    occ_sort = out[out["table_id"].eq("S2411")].groupby("occupation")["var_code"].min().str[-3:].astype(int)
    out["occupation_sort"] = out["occupation"].map(occ_sort)              # S2411 row order, all occupation tables
    out["sex_sort"] = out["sex"].map(SEX_SORT)
    out["race_sort"] = out["race"].map(RACE_SORT)
    out["marital_sort"] = out["marital_status"].map(MARITAL_SORT)
    out["age_detail_sort"] = out["age_detail"].map(
        lambda x: -1 if x == "All" else 0 if x.startswith("Under") else int(x.split()[0]))

    return out.rename(columns={
        "sex": "Sex", "sex_sort": "Sex sort", "race": "Race", "race_sort": "Race sort",
        "age_group": "Age group", "age_sort": "Age sort", "age_detail": "Age detail", "age_detail_sort": "Age detail sort",
        "education_level": "Education level", "education_sort": "Education sort",
        "education_detail": "Education detail", "education_detail_sort": "Education detail sort",
        "marital_status": "Marital status", "marital_sort": "Marital sort", "children": "Children",
        "occupation": "Occupation", "occupation_level": "Occupation level", "occupation_sort": "Occupation sort",
        "worker_type": "Worker type", "census_table": "Census table"})


# ===========================================================================
# TOPIC TABLES -- one line each (replaces the old Topics/SQL/*.sql files)
#   name, Census table(s), row filter, breakdown columns, measures[, Is total rule]
# Each breakdown column gets its "X sort" column automatically (SORT_OF).
# Is total rule: default = Sex is Total or any breakdown is "All"; None = no column.
# ===========================================================================
LABOR = ["Population 16+", "In labor force", "Civilian labor force", "Employed", "Unemployed",
         "In Armed Forces", "Not in labor force"]
TOPICS = [
    ("population_age", ["B01001"], lambda d: d["Race"].eq("All"),
     ["Sex", "Age group", "Age detail"], ["Population"]),
    # race groups overlap (Hispanic is any race): never add them up
    ("population_race", ["B01001"], lambda d: d["Race"].ne("All") | d["Age detail"].eq("All"),
     ["Sex", "Race"], ["Population"]),
    ("population_education", ["B15002"], None,
     ["Sex", "Education level", "Education detail"], ["Population 25+"]),
    ("population_marital_status", ["B12001"], lambda d: d["measure"].eq("Population 15+"),
     ["Sex", "Marital status"], ["Population 15+"]),
    # medians: never add or average
    ("median_earnings_race", ["B20017"], None,
     ["Sex", "Race", "Worker type"], ["Median earnings"]),
    ("median_earnings_education", ["B20004"], None,
     ["Sex", "Education level"], ["Median earnings"]),
    ("median_earnings_occupation", ["S2411", "S2412"], None,
     ["Sex", "Occupation", "Occupation level", "Worker type"], ["Median earnings"]),
    ("workers_occupation", ["C24010"], lambda d: d["Race"].eq("All"),
     ["Sex", "Occupation", "Occupation level"], ["Workers"],
     lambda t: t["Sex"].eq("Total") | t["Occupation level"].ne("Subgroup")),
    # race is only published for the 5 major groups
    ("workers_race_occupation", ["C24010"],
     lambda d: d["Race"].ne("All") | d["Occupation level"].isin(["Total", "Major group"]),
     ["Sex", "Race", "Occupation", "Occupation level"], ["Workers"]),
    ("labor_force_age", ["B23001"], None,
     ["Sex", "Age group", "Age detail"], ["Population 16+", "In labor force"]),
    ("labor_force_race_age", ["C23002"], None,
     ["Sex", "Race", "Age group", "Age detail"], LABOR),
    ("labor_force_marital_status", ["B12006"], None,
     ["Sex", "Marital status"], ["Population 16+", "In labor force"]),
    ("poverty_race", ["B17001"], None,
     ["Sex", "Race"], ["Below poverty", "At or above poverty", "Poverty universe"]),
    # SNAP rate = receiving / households
    ("snap_households", ["B22002"], lambda d: d["Sex"].eq("Total"),
     [], ["Households", "Households receiving SNAP"], None),
    # single householders only: Women + Men is NOT all households
    ("snap_single_householders_children", ["B22002"], lambda d: d["Sex"].ne("Total"),
     ["Sex", "Children"], ["Households receiving SNAP", "Households not receiving SNAP"], None),
]
SORT_OF = {"Sex": "Sex sort", "Race": "Race sort", "Age group": "Age sort", "Age detail": "Age detail sort",
           "Education level": "Education sort", "Education detail": "Education detail sort",
           "Marital status": "Marital sort", "Occupation": "Occupation sort"}


def default_total(t):
    cols = [c for c in ["Race", "Age detail", "Education level", "Marital status", "Occupation"] if c in t]
    return t["Sex"].eq("Total") | t[cols].isin(["All", "All occupations"]).any(axis=1)


def write_topic_tables(acs):
    out_dir = OUT_DIR
    geo = ["county", "county_fips", "year"]
    print(f"\nWriting topic tables to {out_dir}")
    for name, tables, where, dims, measures, *total in TOPICS:
        total = total[0] if total else default_total
        d = acs[acs["table_id"].isin(tables)]
        d = d[where(d)] if where else d
        cols = [c for x in dims for c in (x, SORT_OF.get(x)) if c]      # each breakdown + its sort
        keys = geo + cols + ["Census table"]
        t = d[keys].drop_duplicates()
        for m in measures:                                               # measures side by side
            part = d[d["measure"].eq(m)].groupby(keys, dropna=False)[["estimate", "moe"]].max()
            part.columns = [m, f"{m} MOE"]
            t = t.merge(part.reset_index(), on=keys, how="left")
        if total:
            t.insert(len(geo) + len(cols), "Is total", np.where(total(t), "Yes", "No"))
        if measures == ["Median earnings"]:
            t.insert(0, "Geography level", "County")
        num = t.select_dtypes("number").columns
        t[num] = t[num].round().astype("Int64")

        per = t.groupby(["county_fips", "year"]).size()
        if t.duplicated(geo + dims).any() or len(per) != 72 * t["year"].nunique() or per.nunique() != 1:
            raise ValueError(f"{name}: duplicate rows or uneven rows per county-year "
                             f"({per.value_counts().to_dict()})")
        t.sort_values(geo + cols).to_csv(out_dir / f"{name}.csv", index=False)
        print(f"  {name:36s} {len(t):>7,} rows  ({per.iloc[0]} per county-year)")


# ===========================================================================
# MAIN
# ===========================================================================
def run():
    raw = pd.concat([pull_from_api(year) for year in YEARS], ignore_index=True)
    write_topic_tables(build_acs_table(raw))


if __name__ == "__main__":
    run()
