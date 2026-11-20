# SOW_prod/Data/scripts/build_births.py
#
# WISH birth counts -> Data/Production/WISH/births.csv (one table, same standard columns as the other sources).
#
# Reads Data/Raw/wish_births/wish_births_2020_2024.csv, one row per
#   year x county x category,   columns: year, county, race, age, education, marital_status, birth_count
# (WISH has no API: the file was pulled from the WISH Birth Counts Module in the browser, one query per
# breakdown and year -- county by age / education (9 categories) / race-ethnicity / marital status.
# WISH values are kept exactly as shown: a number, "X" = suppressed (1-4 births), "." = zero.)
# Each row is ONE breakdown: the other three category columns say "All"; a row with all four "All" is the
# county's exact total. To add a year, add its rows to that CSV (same layout) and re-run.
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from standards import (RACE_WISH, RAW_DATA_DIR, STANDARD_VALUES, check_values,  # noqa: E402
                       county_columns, map_strict, require_raw_data)

require_raw_data("wish_births/wish_births_2020_2024.csv")
RAW_FILE = RAW_DATA_DIR / "wish_births" / "wish_births_2020_2024.csv"
OUT_DIR = str(Path(__file__).resolve().parents[1] / "Production" / "WISH")   # Data/Production/WISH

WI_COUNTY_FIPS = {
    "ADAMS": "001", "ASHLAND": "003", "BARRON": "005", "BAYFIELD": "007",
    "BROWN": "009", "BUFFALO": "011", "BURNETT": "013", "CALUMET": "015",
    "CHIPPEWA": "017", "CLARK": "019", "COLUMBIA": "021", "CRAWFORD": "023",
    "DANE": "025", "DODGE": "027", "DOOR": "029", "DOUGLAS": "031",
    "DUNN": "033", "EAU CLAIRE": "035", "FLORENCE": "037", "FOND DU LAC": "039",
    "FOREST": "041", "GRANT": "043", "GREEN": "045", "GREEN LAKE": "047",
    "IOWA": "049", "IRON": "051", "JACKSON": "053", "JEFFERSON": "055",
    "JUNEAU": "057", "KENOSHA": "059", "KEWAUNEE": "061", "LA CROSSE": "063",
    "LAFAYETTE": "065", "LANGLADE": "067", "LINCOLN": "069", "MANITOWOC": "071",
    "MARATHON": "073", "MARINETTE": "075", "MARQUETTE": "077", "MENOMINEE": "078",
    "MILWAUKEE": "079", "MONROE": "081", "OCONTO": "083", "ONEIDA": "085",
    "OUTAGAMIE": "087", "OZAUKEE": "089", "PEPIN": "091", "PIERCE": "093",
    "POLK": "095", "PORTAGE": "097", "PRICE": "099", "RACINE": "101",
    "RICHLAND": "103", "ROCK": "105", "RUSK": "107", "ST CROIX": "109",
    "SAUK": "111", "SAWYER": "113", "SHAWANO": "115", "SHEBOYGAN": "117",
    "TAYLOR": "119", "TREMPEALEAU": "121", "VERNON": "123", "VILAS": "125",
    "WALWORTH": "127", "WASHBURN": "129", "WASHINGTON": "131", "WAUKESHA": "133",
    "WAUPACA": "135", "WAUSHARA": "137", "WINNEBAGO": "139", "WOOD": "141",
}

# Category order = WISH column order. Age and education have a real order (sort 1..n, Missing = 99);
# race has none, so its sort is just the source order.
AGE_ORDER = ["Under 15 yrs", "15-17 yrs", "18-19 yrs", "20-24 yrs", "25-29 yrs", "30-34 yrs", "35-39 yrs",
             "40-44 yrs", "45+ yrs", "Missing"]
EDU_ORDER = ["8th Grade or Less", "9th-12th Grade-No Diploma", "High School Graduate or GED Completed",
             "Some College Credit-No Degree", "Associate Degree", "Bachelor's Degree", "Master's Degree",
             "Doctorate or Professional Degree", "Missing"]
RACE_ORDER = ["Non-Hispanic White", "Non-Hispanic Black", "Non-Hispanic American Indian/Alaska Native",
              "Non-Hispanic Laotian/Hmong", "Non-Hispanic Other Asian/Pacific Islander/Native Hawaiian",
              "Non-Hispanic Other", "Hispanic", "Non-Hispanic Multi-race", "Missing"]
SORT = lambda order: {c: (99 if c == "Missing" else i + 1) for i, c in enumerate(order)}  # noqa: E731
AGE_SORT_W, EDU_SORT_W, RACE_SORT_W = SORT(AGE_ORDER), SORT(EDU_ORDER), SORT(RACE_ORDER)

AGE_LABEL = {  # drop " yrs" so births match the ACS / crime age style
    "Under 15 yrs": "Under 15", "15-17 yrs": "15-17", "18-19 yrs": "18-19", "20-24 yrs": "20-24",
    "25-29 yrs": "25-29", "30-34 yrs": "30-34", "35-39 yrs": "35-39", "40-44 yrs": "40-44",
    "45+ yrs": "45+", "Missing": "Unknown",
}
EDUCATION_LABEL = {
    "8th Grade or Less": "8th grade or less",
    "9th-12th Grade-No Diploma": "9th-12th grade, no diploma",
    "High School Graduate or GED Completed": "High school graduate or GED",
    "Some College Credit-No Degree": "Some college, no degree",
    "Associate Degree": "Associate's degree",
    "Bachelor's Degree": "Bachelor's degree",
    "Master's Degree": "Master's degree",
    "Doctorate or Professional Degree": "Doctorate or professional degree",
    "Missing": "Unknown",
}
MARITAL_LABEL = {"Married": "Married", "Not Married": "Not married", "Missing": "Unknown"}


def parse_birth_value(raw):
    """WISH: 'X' = suppressed count between 1 and 4 (blank + is_suppressed); '.' = a true zero."""
    value = str(raw).strip()
    if value.upper() == "X":
        return np.nan, True
    if value == ".":
        return 0, False
    return int(value.replace(",", "")), False


raw = pd.read_csv(RAW_FILE, dtype=str, keep_default_na=False)
expected = ["year", "county", "race", "age", "education", "marital_status", "birth_count"]
if list(raw.columns) != expected:
    raise SystemExit(f"{RAW_FILE.name} must have columns {expected}, found {list(raw.columns)}")

# county: WISH writes names in capitals ("ST. CROIX"); MISSING = no county of residence (-> Unknown)
county_upper = raw["county"].str.upper().str.replace(".", "", regex=False).str.strip()
geoid = pd.to_numeric("55" + county_upper.map(WI_COUNTY_FIPS), errors="coerce").astype("Int64")
bad = sorted(set(county_upper[geoid.isna() & county_upper.ne("MISSING")]))
if bad:
    raise ValueError(f"WISH counties not in the county list: {bad}")

# which breakdown a row is: exactly one of the four category columns is not "All" (none = the total)
detail = raw[["age", "education", "marital_status", "race"]].ne("All")
if (detail.sum(axis=1) > 1).any():
    raise ValueError("a WISH row has more than one breakdown filled in")
breakdown = np.select([detail["age"], detail["education"], detail["marital_status"], detail["race"]],
                      ["Age", "Education", "Marital status", "Race"], "Total")

values = [parse_birth_value(v) for v in raw["birth_count"]]

births = pd.DataFrame()
births["county"], births["county_fips"] = county_columns(geoid)
births["year"] = raw["year"].astype(int)
births["sex"] = "Women"
births["breakdown"] = breakdown
for col in ["age_group", "education_level", "marital_status", "race"]:
    births[col] = "All"
births["age_sort"] = 0
births["education_sort"] = 0
is_age, is_edu = births["breakdown"].eq("Age"), births["breakdown"].eq("Education")
is_mar, is_race = births["breakdown"].eq("Marital status"), births["breakdown"].eq("Race")
births.loc[is_age, "age_group"] = map_strict(raw.loc[is_age, "age"], AGE_LABEL, "birth age")
births.loc[is_age, "age_sort"] = raw.loc[is_age, "age"].map(AGE_SORT_W)
births.loc[is_edu, "education_level"] = map_strict(raw.loc[is_edu, "education"], EDUCATION_LABEL, "birth education")
births.loc[is_edu, "education_sort"] = raw.loc[is_edu, "education"].map(EDU_SORT_W)
births.loc[is_mar, "marital_status"] = map_strict(raw.loc[is_mar, "marital_status"], MARITAL_LABEL, "birth marital")
race_clean = raw["race"].str.replace("/ ", "/", regex=False)  # WISH writes "Pacific Islander/ Native Hawaiian"
births.loc[is_race, "race"] = map_strict(race_clean[is_race], RACE_WISH, "birth race")
births["race_sort"] = race_clean.map(RACE_SORT_W).where(is_race, 0).astype(int)
# Suppressed = 1-4 births, value hidden by WISH. birth_count stays blank so it is never summed.
births["birth_count"] = pd.array([v[0] for v in values], dtype="Int64")
births["is_suppressed"] = [v[1] for v in values]

check_values(births, {**STANDARD_VALUES, "sex": ["Women"]}, "births")

print(f"births: {births.shape[0]:,} rows x {births.shape[1]} columns; years {sorted(births['year'].unique())}")
print(f"rows with county = Unknown (no county of residence): {births['county'].eq('Unknown').sum():,}")
print(f"suppressed cells: {int(births['is_suppressed'].sum()):,}")
_tot = births[births["breakdown"].eq("Total") & births["county"].ne("Unknown")]
print("Total rows (real counties) per year:", _tot.groupby("year")["birth_count"].sum().to_dict())

births.to_csv(f"{OUT_DIR}/births.csv", index=False)
print(f"Saved births.csv to {OUT_DIR}")
