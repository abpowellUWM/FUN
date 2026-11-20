# SOW_prod/Data/scripts/build_crime.py
#
# Raw NIBRS tables -> two dashboard tables, in one pass (no fact/dim step),
# for every year in YEARS, stacked with a `year` column, written to Data/Production/FBI/:
#   crime_people.csv    one row per year x county x Victim/Offender x sex x age x race
#   crime_offenses.csv  one row per year x county x offense x victim sex/age/race x relationship
#   crime_coverage.csv  one row per year: agencies in the FBI file / agencies with any incident
#   crime_coverage_county.csv  one row per year x county: how many agencies filed an incident there
# Two tables because people and offenses are different units (one victim can
# have several offenses) -- never add person_count and offense_count together.
#
# Reads (never writes): Data/Raw/fbi_nibrs/ (FBI Crime Data Explorer, Wisconsin
# NIBRS downloads, one NIBRS_<year>/ folder per year). County splits come from
# Data/reference/agency_county_splits.csv,
# a copy of reference_data/agency_county_splits.xlsx with an ORI column and
# three added departments (the original file is unchanged).
# Same domestic-violence logic as scripts/fbi/build_crime_facts.py, which is
# left unchanged. Changes confirmed with the user (see README): "ST. CROIX"
# match; age "00" -> Unknown; 2020 infant ages read from age_id; splits
# matched by ORI; one random seed per department and year.
import re
import sys
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from standards import (AGE_SORT, COUNTY_BY_FIPS, RACE_FBI, RAW_DATA_DIR, STANDARD_VALUES,  # noqa: E402
                       check_values, map_strict, require_raw_data)

require_raw_data("fbi_nibrs")
NIBRS = RAW_DATA_DIR / "fbi_nibrs"
SPLITS_CSV = HERE.parent / "reference" / "agency_county_splits.csv"   # Data/reference
OUT_DIR = HERE.parent / "Production" / "FBI"                         # Data/Production/FBI
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Folder holding each year's raw NIBRS CSVs. Add a year here when you download it.
RAW_DIRS = {
    2020: NIBRS / "NIBRS_2020",
    2021: NIBRS / "NIBRS_2021",
    2022: NIBRS / "NIBRS_2022",
    2023: NIBRS / "NIBRS_2023",
    2024: NIBRS / "NIBRS_2024",
}
YEARS = sorted(RAW_DIRS)
# Which agencies are in the output. False (since 2026-11-11, user decision) = every agency that filed
# any incident in a year is included in that year, so agencies that start reporting later appear from
# their first year (more agencies report each year -- see crime_coverage.csv). True = only the agencies
# that filed at least one incident (any crime type) in EVERY year in YEARS, for a like-for-like trend.
ONLY_AGENCIES_REPORTING_ALL_YEARS = False
RANDOM_SEED = 42  # combined with year + department ORI + victim/offender, see assign_county()


# ===========================================================================
# 1. DEFINITIONS
# FBI/NIBRS Domestic and Family Violence: force, threat, coercion or a Crime
# Against Property by a current/former spouse, parent or guardian, someone the
# victim shares a child with, a current/former romantic or intimate partner,
# or someone similarly situated. NIBRS user manual:
# https://le.fbi.gov/file-repository/nibrs-user-manual-2025-0-062625.pdf/view
# ===========================================================================
DV_RELATIONSHIP_CODES = [
    "SE",  # Victim Was Spouse
    "CS",  # Victim Was Common-Law Spouse
    "XS",  # Victim Was Ex-Spouse
    "XR",  # Victim Was Ex-Relationship (Ex-Boyfriend/Girlfriend)
    "BG",  # Victim Was Boyfriend/Girlfriend
    "CF",  # Victim Was Child of Boyfriend/Girlfriend
    "HR",  # Homosexual Relationship (discontinued -- may appear in historical data)
    "PA",  # Victim Was Parent
    "CH",  # Victim Was Child
    "SP",  # Victim Was Stepparent
    "SC",  # Victim Was Stepchild
    "FC",  # Victim Was Foster Child (code exists from 2023)
    "FP",  # Victim Was Foster Parent (code exists from 2023)
]

DV_OFFENSES = {  # offense_code -> offense_category
    "09A": "Homicide", "09B": "Homicide",
    "11A": "Sex Offenses", "11B": "Sex Offenses", "11C": "Sex Offenses", "11D": "Sex Offenses",
    "36A": "Sex Offenses",
    "100": "Kidnapping",
    "13A": "Assault", "13B": "Assault", "13C": "Assault",
    "120": "Property", "210": "Property", "220": "Property", "240": "Property",
    "250": "Property", "270": "Property", "290": "Property",
    "23D": "Larceny/Theft", "23F": "Larceny/Theft", "23G": "Larceny/Theft", "23H": "Larceny/Theft",
}
CATEGORY_SORT = {"Homicide": 1, "Sex Offenses": 2, "Kidnapping": 3, "Assault": 4,
                 "Property": 5, "Larceny/Theft": 6}

# NIBRS "Victim Was X" = the victim was the offender's X.
RELATIONSHIP_LABEL = {
    "Victim Was Spouse": "Spouse",
    "Victim Was Common-Law Spouse": "Common-law spouse",
    "Victim was Ex-Spouse": "Ex-spouse",
    "Victim Was Boyfriend/Girlfriend": "Boyfriend/girlfriend",
    "Victim Was Ex-Relationship (Ex-Boyfriend/Girlfriend)": "Ex-boyfriend/girlfriend",
    "Homosexual Relationship": "Same-sex partner",
    "Victim Was Child": "Child",
    "Victim Was Stepchild": "Stepchild",
    "Victim Was Child of Boyfriend or Girlfriend": "Child of boyfriend/girlfriend",
    "Victim Was Parent": "Parent",
    "Victim Was Stepparent": "Stepparent",
    "Victim was Foster Parent": "Foster parent",
    "Victim was Foster Child": "Foster child",
}
INTIMATE_PARTNER = {"Spouse", "Common-law spouse", "Ex-spouse", "Boyfriend/girlfriend",
                    "Ex-boyfriend/girlfriend", "Same-sex partner"}

SEX_LABEL = {"F": "Women", "M": "Men"}  # U, X, blank -> Unknown
AGE_BINS = [0, 17, 24, 34, 44, 54, 64, 200]
AGE_LABELS = ["Under 18", "18-24", "25-34", "35-44", "45-54", "55-64", "65+"]
# NIBRS age codes (NIBRS_AGE.csv): NN/NB/BB = under 1 year -> age 0;
# "99" = over 98 -> 65+; "00" = Unknown and "NS" = Not Specified -> Unknown
# (per documentation/decisions/2026-07-12_fbi_unknown-demographics.md; the
# original script read "00" as age 0 / Under 18 -- fixed 2026-11-09).
INFANT_CODES = {"NN", "NB", "BB"}
UNKNOWN_AGE_CODES = {"00", "NS"}


# ===========================================================================
# 2. COUNTY ASSIGNMENT (same rule every year)
# Each person gets the county of the reporting agency. Agencies that serve
# several counties get each person randomly assigned to one of their
# counties, weighted by 2020 Census population share (e.g. Appleton ~83%
# Outagamie, ~15% Calumet, ~2% Winnebago). Departments are matched by ORI,
# the FBI's permanent department ID -- not by name, because the Eau Claire
# County Sheriff and Eau Claire city police share the name "Eau Claire".
# The same shares are used for every year. No county -> "Unknown".
# ===========================================================================
FIPS_BY_COUNTY = {name.upper().replace(".", ""): fips for fips, name in COUNTY_BY_FIPS.items()}


def normalize_county(name):
    if pd.isna(name) or str(name).strip() == "":
        return None
    name = re.sub(r"\bST\.\s*", "ST ", str(name).upper().strip())
    return re.sub(r"\s+", " ", name).strip()


splits = pd.read_csv(SPLITS_CSV, dtype={"ori": str})
split_rows = [
    (r["ori"], str(r[f"county_{slot}"]).upper(), r[f"county{slot}_%"])
    for _, r in splits.iterrows() for slot in (1, 2, 3)
    if pd.notna(r[f"county_{slot}"]) and pd.notna(r[f"county{slot}_%"])
]
# Colby-Abbotsford has one row per city under one ORI: averaged, as before.
weights = (pd.DataFrame(split_rows, columns=["ori", "county", "weight"])
           .groupby(["ori", "county"], as_index=False)["weight"].mean())
# Normalize before matching: the splits file spells "ST. CROIX". The original
# script matched the raw spelling, so people randomly assigned to St. Croix
# lost their county (became Unknown). Fixed here 2026-11-08.
weights["county_fips"] = weights["county"].apply(normalize_county).map(FIPS_BY_COUNTY)
if weights["county_fips"].isna().any():
    raise ValueError(f"Split counties not recognized: {sorted(weights.loc[weights['county_fips'].isna(), 'county'])}")
MULTI_COUNTY = {ori: (list(g["county_fips"]), list(g["weight"])) for ori, g in weights.groupby("ori")}


# ===========================================================================
# 3. ONE YEAR: raw tables -> labeled people and offenses
# ===========================================================================
def read_raw(year, name, cols=None):
    """Read one raw NIBRS table. Column names are lowercased (2020 uses CAPITALS)
    and file names are matched case-insensitively."""
    folder = RAW_DIRS[year]
    matches = [p for p in folder.iterdir() if p.is_file() and p.name.lower() == name.lower()]
    if not matches:
        raise FileNotFoundError(f"{name} not found in {folder}")
    df = pd.read_csv(matches[0], dtype=str, low_memory=False)
    df.columns = df.columns.str.lower()
    return df[cols] if cols else df


def age_years(people, age_codes):
    """Age in years, or NaN = Unknown. 2021+ put age codes (NN, 00, NS...) in
    age_num. 2020 leaves age_num blank and stores the code in age_id, so look
    it up in that year's NIBRS_AGE table (infants -> Under 18, over 98 -> 65+)."""
    code = people["age_num"].str.strip()
    if "age_id" in people:
        code = code.fillna(people["age_id"].map(age_codes))
    code = code.where(~code.isin(INFANT_CODES), "0")
    code = code.where(~code.isin(UNKNOWN_AGE_CODES))
    return pd.to_numeric(code, errors="coerce")


def reporting_oris(year):
    """ORIs of the agencies that filed at least one incident of ANY type in `year`
    (not only domestic violence -- 'is this agency reporting at all')."""
    incident = read_raw(year, "NIBRS_incident.csv", ["agency_id"])
    agencies = read_raw(year, "agencies.csv", ["agency_id", "ori"])
    ids = set(pd.to_numeric(incident["agency_id"]).dropna())
    agencies["agency_id"] = pd.to_numeric(agencies["agency_id"])
    return set(agencies.loc[agencies["agency_id"].isin(ids), "ori"].dropna().str.strip())


def build_year(year, offense_names, keep_oris):
    victim = read_raw(year, "NIBRS_VICTIM.csv")
    offender = read_raw(year, "NIBRS_OFFENDER.csv")
    offense = read_raw(year, "NIBRS_OFFENSE.csv")
    if "offense_code" not in offense:
        # 2020 stores offense_type_id instead of the code; translate it with
        # that year's own NIBRS_OFFENSE_TYPE table (same codes, e.g. 13B).
        types = read_raw(year, "NIBRS_OFFENSE_TYPE.csv", ["offense_type_id", "offense_code"])
        offense = offense.merge(types, on="offense_type_id", how="left")
    offense = offense[["offense_id", "offense_code"]]
    victim_offense = read_raw(year, "NIBRS_VICTIM_OFFENSE.csv", ["victim_id", "offense_id"])
    victim_offender = read_raw(year, "NIBRS_VICTIM_OFFENDER_REL.csv", ["victim_id", "offender_id", "relationship_id"])
    relationship = read_raw(year, "NIBRS_RELATIONSHIP.csv", ["relationship_id", "relationship_code", "relationship_name"])
    incident = read_raw(year, "NIBRS_incident.csv", ["incident_id", "agency_id"])
    agencies = read_raw(year, "agencies.csv", ["agency_id", "ori", "pub_agency_name", "county_name"])
    race = read_raw(year, "REF_RACE.csv", ["race_id", "race_desc"])  # race_id numbering differs in 2020
    age_codes = read_raw(year, "NIBRS_AGE.csv", ["age_id", "age_code"]).set_index("age_id")["age_code"]

    if "offense_code" not in offense or offense["offense_code"].isna().all():
        raise ValueError(f"{year}: NIBRS_OFFENSE.csv has no offense_code column")
    for col in ("victim_id", "offender_id", "offense_id", "incident_id", "relationship_id"):
        for df in (victim, offender, victim_offense, victim_offender, incident, offense, relationship):
            if col in df:
                df[col] = pd.to_numeric(df[col])

    # --- county lookup for this year's agencies
    agencies["pub_agency_name"] = agencies["pub_agency_name"].str.strip()
    agencies["agency_id"] = pd.to_numeric(agencies["agency_id"])
    agencies["county_fips"] = agencies["county_name"].apply(normalize_county).map(FIPS_BY_COUNTY)
    incident["agency_id"] = pd.to_numeric(incident["agency_id"])
    # Optional agency rule (keep_oris is None = off, every agency stays). When on, only agencies that
    # reported in every year stay. Everything below -- the DV definition, the county assignment --
    # runs unchanged on what is left.
    agencies["ori"] = agencies["ori"].str.strip()
    if keep_oris is None:
        kept_agency_ids = set(agencies["agency_id"])
        in_scope = pd.Series(True, index=agencies.index)
    else:
        kept_agency_ids = set(agencies.loc[agencies["ori"].isin(keep_oris), "agency_id"])
        in_scope = agencies["ori"].isin(keep_oris)
        kept_incidents = set(incident.loc[incident["agency_id"].isin(kept_agency_ids), "incident_id"])
        victim = victim[victim["incident_id"].isin(kept_incidents)].copy()
        offender = offender[offender["incident_id"].isin(kept_incidents)].copy()
    no_county = agencies[in_scope & ~agencies["ori"].isin(MULTI_COUNTY) & agencies["county_fips"].isna()]
    no_county = sorted(no_county["pub_agency_name"].fillna("?") + " (" + no_county["county_name"].fillna("blank") + ")")

    def assign_county(people, person_type):
        people = people.merge(incident, on="incident_id", how="left")
        people = people.merge(agencies[["agency_id", "ori", "county_fips"]], on="agency_id", how="left")
        for ori, (fips_list, w) in MULTI_COUNTY.items():
            mask = (people["ori"] == ori).values
            if mask.any():
                # One fixed random stream per department, year and person type, so
                # adding/removing a department never moves anyone at another one.
                rng = np.random.default_rng([RANDOM_SEED, year, zlib.crc32(ori.encode()),
                                             0 if person_type == "Victim" else 1])
                people.loc[mask, "county_fips"] = rng.choice(fips_list, size=mask.sum(), p=w)
        people["county"] = people["county_fips"].map(COUNTY_BY_FIPS).fillna("Unknown")
        people["county_fips"] = people["county_fips"].where(people["county"] != "Unknown", "")
        return people

    def label_people(people):
        people = people.merge(race, on="race_id", how="left")
        people["race"] = map_strict(people["race_desc"].fillna("Unknown"), RACE_FBI, f"{year} race")
        people["sex"] = people["sex_code"].map(SEX_LABEL).fillna("Unknown")
        age = age_years(people, age_codes)
        people["age_group"] = (pd.cut(age, bins=AGE_BINS, labels=AGE_LABELS, right=True, include_lowest=True)
                               .astype(str).replace("nan", "Unknown"))
        people["age_sort"] = map_strict(people["age_group"], AGE_SORT, "age_group")
        people["year"] = year
        return people

    # --- DV victims: a DV relationship to an offender AND at least one DV offense
    dv_rel_ids = set(relationship.loc[relationship["relationship_code"].isin(DV_RELATIONSHIP_CODES), "relationship_id"])
    dv_offense_ids = set(offense.loc[offense["offense_code"].isin(DV_OFFENSES), "offense_id"])
    dv_victim_ids = (set(victim_offender.loc[victim_offender["relationship_id"].isin(dv_rel_ids), "victim_id"])
                     & set(victim_offense.loc[victim_offense["offense_id"].isin(dv_offense_ids), "victim_id"])
                     & set(victim["victim_id"]))
    dv_links = victim_offender[victim_offender["victim_id"].isin(dv_victim_ids)
                               & victim_offender["relationship_id"].isin(dv_rel_ids)]

    victims = assign_county(label_people(victim[victim["victim_id"].isin(dv_victim_ids)].copy()), "Victim")
    offenders = assign_county(label_people(offender[offender["offender_id"].isin(set(dv_links["offender_id"]))].copy()),
                              "Offender")

    people = (
        pd.concat([victims.assign(person_type="Victim"), offenders.assign(person_type="Offender")], ignore_index=True)
        .groupby(["county", "county_fips", "year", "person_type", "sex", "age_group", "age_sort", "race"])
        .size().reset_index(name="person_count")
    )

    # --- offenses: one count per victim x distinct DV offense; each victim gets
    # one relationship (their lowest-numbered DV-linked offender)
    victim_relationship = (
        dv_links.merge(relationship[["relationship_id", "relationship_name"]], on="relationship_id", how="left")
        .sort_values("offender_id").drop_duplicates("victim_id")[["victim_id", "relationship_name"]]
    )
    vo = (
        victim_offense[victim_offense["victim_id"].isin(dv_victim_ids) & victim_offense["offense_id"].isin(dv_offense_ids)]
        .merge(offense, on="offense_id", how="left")
        .drop_duplicates(["victim_id", "offense_code"])
        .merge(victims[["victim_id", "county", "county_fips", "year", "sex", "age_group", "age_sort", "race"]],
               on="victim_id", how="left")
        .merge(victim_relationship, on="victim_id", how="left")
    )
    vo["offense_category"] = vo["offense_code"].map(DV_OFFENSES)
    vo["offense_category_sort"] = vo["offense_category"].map(CATEGORY_SORT)
    vo["offense"] = map_strict(vo["offense_code"], offense_names, "offense_code")
    vo["relationship"] = map_strict(vo["relationship_name"], RELATIONSHIP_LABEL, f"{year} relationship")
    vo["relationship_group"] = np.where(vo["relationship"].isin(INTIMATE_PARTNER), "Intimate partner", "Family")
    offenses = (
        vo.groupby(["county", "county_fips", "year", "offense_category", "offense_category_sort", "offense_code",
                    "offense", "sex", "age_group", "age_sort", "race", "relationship", "relationship_group"])
        .size().reset_index(name="offense_count")
    )

    n_victims = people.loc[people["person_type"] == "Victim", "person_count"].sum()
    assert n_victims == len(dv_victim_ids), f"{year}: every DV victim must be counted exactly once"
    reporting = incident["agency_id"].nunique() if keep_oris is None else len(kept_agency_ids & set(incident["agency_id"]))
    print(f"{year}: {n_victims:,} victims | "
          f"{people.loc[people['person_type'] == 'Offender', 'person_count'].sum():,} offenders | "
          f"{offenses['offense_count'].sum():,} offenses | {reporting} agencies | "
          f"county Unknown: {people.loc[people['county'] == 'Unknown', 'person_count'].sum():,} people")
    if no_county:
        print(f"      agencies with no single Wisconsin county and no split (-> Unknown): {no_county}")
    coverage = {"year": year, "agencies_in_fbi_file": agencies["agency_id"].nunique(),
                "agencies_with_incidents": incident["agency_id"].nunique(),
                "agencies_in_output": reporting}
    return people, offenses, coverage


# ===========================================================================
# 4. ALL YEARS, CHECKS, SAVE
# ===========================================================================
# One name per offense code across years: the latest year's FBI name.
# 11D was "Fondling" through 2022 and "Criminal Sexual Contact" from 2023 --
# the FBI broadened the definition (any body part; also sexual degradation or
# humiliation; forcing the victim to touch the offender). Kept as one offense
# per user 2026-11-09, with a caveat in the README.
latest = read_raw(max(YEARS), "NIBRS_OFFENSE_TYPE.csv", ["offense_code", "offense_name"])
OFFENSE_NAMES = latest.drop_duplicates("offense_code").set_index("offense_code")["offense_name"]

# The same department under two FBI ORIs: River Falls PD was WI0480100 (Pierce County numbering, through
# 2020) and got WI0561100 (St. Croix County numbering) from 2021. For the county coverage counts they are
# one agency, so it is not counted twice. (The people tables are not affected: both ORIs are in the splits file.)
SAME_AGENCY_ORI = {"WI0480100": "WI0561100"}


def reporting_agency_counties(year):
    """One row per (ORI, county_fips) for the agencies that filed at least one incident in `year`.
    County rule is the same as for people: a multi-county department is listed under every county in the
    splits file, a single-county agency under its own county, and an agency with no county under "" (Unknown)."""
    incident = read_raw(year, "NIBRS_incident.csv", ["agency_id"])
    agencies = read_raw(year, "agencies.csv", ["agency_id", "ori", "county_name"])
    agencies["ori"] = agencies["ori"].str.strip()
    reporting = agencies[agencies["agency_id"].isin(set(incident["agency_id"]))].drop_duplicates("ori")
    out = []
    for ori, county_name in zip(reporting["ori"], reporting["county_name"]):
        if ori in MULTI_COUNTY:
            out += [(ori, fips) for fips in MULTI_COUNTY[ori][0]]
        else:
            out.append((ori, FIPS_BY_COUNTY.get(normalize_county(county_name)) or ""))
    out = [(SAME_AGENCY_ORI.get(ori, ori), fips) for ori, fips in out]
    return pd.DataFrame(out, columns=["ori", "county_fips"]).drop_duplicates()


def county_coverage(years):
    """Agencies that filed at least one incident (any crime type) per year x county, so a blank or missing
    county-year can be told apart: 0 agencies = no data (not zero violence). A multi-county department counts
    in each county it serves. Two constant columns say how many of the county's agencies reported in every
    year / in at least one year, so a trend can be limited to counties whose agencies did not change."""
    per_year = {y: reporting_agency_counties(y) for y in years}
    all_years = set.intersection(*[set(d["ori"]) for d in per_year.values()])
    pairs = pd.concat(per_year.values()).drop_duplicates()
    rows = []
    for y in years:
        d = per_year[y]
        for fips, name in list(COUNTY_BY_FIPS.items()) + [("", "Unknown")]:
            ever = pairs.loc[pairs["county_fips"] == fips, "ori"]
            rows.append({"county": name, "county_fips": fips, "year": y,
                         "agencies_reporting": int((d["county_fips"] == fips).sum()),
                         "agencies_reporting_all_years": int(ever.isin(all_years).sum()),
                         "agencies_reporting_any_year": int(ever.nunique())})
    return pd.DataFrame(rows)


# Optional agency rule (see ONLY_AGENCIES_REPORTING_ALL_YEARS at the top). Matched by ORI, which stays
# the same from year to year (agency ids do not).
if ONLY_AGENCIES_REPORTING_ALL_YEARS:
    KEEP_ORIS = set.intersection(*[reporting_oris(year) for year in YEARS])
    print(f"Agencies that reported in all {len(YEARS)} years ({YEARS[0]}-{YEARS[-1]}): {len(KEEP_ORIS)}")
else:
    KEEP_ORIS = None
    print("All agencies included (ONLY_AGENCIES_REPORTING_ALL_YEARS = False)")

results = [build_year(year, OFFENSE_NAMES, KEEP_ORIS) for year in YEARS]
crime_people = pd.concat([r[0] for r in results], ignore_index=True)
crime_offenses = pd.concat([r[1] for r in results], ignore_index=True)
# Coverage: more agencies report each year, so part of any year-over-year
# rise is more reporting, not more violence.
crime_coverage = pd.DataFrame([r[2] for r in results])
crime_coverage_county = county_coverage(YEARS)

for name, df in [("crime_people", crime_people), ("crime_offenses", crime_offenses)]:
    check_values(df, {**STANDARD_VALUES, "race": list(RACE_FBI.values())}, name)

crime_people.to_csv(OUT_DIR / "crime_people.csv", index=False)
crime_offenses.to_csv(OUT_DIR / "crime_offenses.csv", index=False)
crime_coverage.to_csv(OUT_DIR / "crime_coverage.csv", index=False)
crime_coverage_county.to_csv(OUT_DIR / "crime_coverage_county.csv", index=False)
print(f"\nSaved crime_people.csv ({len(crime_people):,} rows), crime_offenses.csv "
      f"({len(crime_offenses):,} rows), crime_coverage.csv and crime_coverage_county.csv "
      f"({len(crime_coverage_county):,} rows) to {OUT_DIR}")
print(crime_coverage.to_string(index=False))
