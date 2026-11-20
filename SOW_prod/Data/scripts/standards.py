
import os
from pathlib import Path

import pandas as pd

# Raw data lives in Data/Raw (this file is in Data/scripts). Set WWC_RAW_DATA to use another folder.
RAW_DATA_DIR = Path(os.getenv("WWC_RAW_DATA") or Path(__file__).resolve().parents[1] / "Raw")

def require_raw_data(*subpaths):
    """Stop with a clear message if the raw data isn't where RAW_DATA_DIR says."""
    for sub in subpaths:
        if not (RAW_DATA_DIR / sub).exists():
            raise SystemExit(f"Raw data not found: {RAW_DATA_DIR / sub}\n"
                             f"Edit RAW_DATA_DIR in standards.py (or set WWC_RAW_DATA) to the folder that holds {sub}.")

YEAR = 2024

COUNTY_BY_FIPS = {
    "55001": "Adams", "55003": "Ashland", "55005": "Barron", "55007": "Bayfield", "55009": "Brown",
    "55011": "Buffalo", "55013": "Burnett", "55015": "Calumet", "55017": "Chippewa", "55019": "Clark",
    "55021": "Columbia", "55023": "Crawford", "55025": "Dane", "55027": "Dodge", "55029": "Door",
    "55031": "Douglas", "55033": "Dunn", "55035": "Eau Claire", "55037": "Florence", "55039": "Fond du Lac",
    "55041": "Forest", "55043": "Grant", "55045": "Green", "55047": "Green Lake", "55049": "Iowa",
    "55051": "Iron", "55053": "Jackson", "55055": "Jefferson", "55057": "Juneau", "55059": "Kenosha",
    "55061": "Kewaunee", "55063": "La Crosse", "55065": "Lafayette", "55067": "Langlade", "55069": "Lincoln",
    "55071": "Manitowoc", "55073": "Marathon", "55075": "Marinette", "55077": "Marquette", "55078": "Menominee",
    "55079": "Milwaukee", "55081": "Monroe", "55083": "Oconto", "55085": "Oneida", "55087": "Outagamie",
    "55089": "Ozaukee", "55091": "Pepin", "55093": "Pierce", "55095": "Polk", "55097": "Portage",
    "55099": "Price", "55101": "Racine", "55103": "Richland", "55105": "Rock", "55107": "Rusk",
    "55109": "St. Croix", "55111": "Sauk", "55113": "Sawyer", "55115": "Shawano", "55117": "Sheboygan",
    "55119": "Taylor", "55121": "Trempealeau", "55123": "Vernon", "55125": "Vilas", "55127": "Walworth",
    "55129": "Washburn", "55131": "Washington", "55133": "Waukesha", "55135": "Waupaca", "55137": "Waushara",
    "55139": "Winnebago", "55141": "Wood",
}

SEX = {"F": "Women", "Female": "Women", "M": "Men", "Male": "Men", "All": "Total", "Total": "Total"}

# Short labels, matching what the 10/27 dashboard already displays.
RACE_ACS = {  # Census race-iteration labels from build_acs.RACE_LABEL
    "All": "All",
    "White alone": "White",
    "White, non-Hispanic": "White (non-Hispanic)",
    "Black or African American": "Black",
    "Hispanic or Latino": "Hispanic",
    "Asian": "Asian",
    "American Indian/Alaska Native": "AIAN",
    "Native Hawaiian/Pacific Islander": "NHPI",
    "Some Other Race": "Other",
    "Two or More Races": "Two or more",
}
RACE_FBI = {  # NIBRS REF_RACE.race_desc
    "White": "White",
    "Black or African American": "Black",
    "Asian": "Asian",
    "American Indian or Alaska Native": "AIAN",
    "Native Hawaiian or Other Pacific Islander": "NHPI",
    "Unknown": "Unknown",
}
RACE_WISH = {  # WISH birth categories; every group except Hispanic is non-Hispanic
    "Non-Hispanic White": "White",
    "Non-Hispanic Black": "Black",
    "Non-Hispanic American Indian/Alaska Native": "AIAN",
    "Non-Hispanic Laotian/Hmong": "Laotian/Hmong",
    "Non-Hispanic Other Asian/Pacific Islander/Native Hawaiian": "Other Asian/NHPI",
    "Non-Hispanic Other": "Other",
    "Hispanic": "Hispanic",
    "Non-Hispanic Multi-race": "Two or more",
    "Missing": "Unknown",
}

AGE_SORT = {"All": 0, "Under 18": 1, "18-24": 2, "25-34": 3, "35-44": 4, "45-54": 5,
            "55-64": 6, "65+": 7, "Unknown": 99}


def map_strict(series, mapping, what):
    """Map values and fail loudly on anything unmapped, so a new raw value
    can never slip through as a blank."""
    out = series.map(mapping)
    bad = sorted(series[out.isna() & series.notna()].astype(str).unique())
    if bad:
        raise ValueError(f"Unmapped {what} value(s): {bad}")
    return out


def county_columns(geoid):
    """geoid (int/float/str, may be null) -> (county, county_fips)."""
    fips = pd.to_numeric(geoid, errors="coerce").astype("Int64").astype(str).replace("<NA>", "")
    county = fips.map(COUNTY_BY_FIPS)
    bad = sorted(fips[county.isna() & (fips != "")].unique())
    if bad:
        raise ValueError(f"FIPS not in COUNTY_BY_FIPS: {bad}")
    return county.fillna("Unknown"), fips


def check_values(df, allowed, table):
    """Assert each standard column only holds its allowed values."""
    for col, values in allowed.items():
        if col in df.columns:
            extra = set(df[col].dropna().unique()) - set(values)
            if extra:
                raise ValueError(f"{table}.{col} has non-standard values: {sorted(extra)}")


STANDARD_VALUES = {
    "sex": ["Women", "Men", "Total", "Unknown"],
    "county": list(COUNTY_BY_FIPS.values()) + ["Unknown"],
}
