# SOW_prod/Data/scripts/build_tableau_ts.py
#
# Builds the one table behind Tableau/SOW_TS.twb (time series, 2020-2024):
#   Victims / Offenders  domestic & family violence people (FBI) per 100,000 people of that sex (ACS)
#   Bachelor's degrees   people 25+ whose highest degree is a bachelor's (ACS)
#   Births               births (WISH) per 1,000 women in that age group (ACS)
#
# Each row keeps the count and its population, so Tableau works out the rate as
# SUM(count) / SUM(population) * per -- correct for the state or any set of counties.
#
# Run after the three source scripts:  python3 Data/scripts/build_tableau_ts.py
# Writes Data/Production/Tableau/sow_ts.csv
from pathlib import Path

import pandas as pd

PROD = Path(__file__).resolve().parents[1] / "Production"
OUT = PROD / "Tableau" / "sow_ts.csv"
GEO = ["county", "county_fips", "year"]

pop = pd.read_csv(PROD / "ACS" / "population_age.csv", dtype={"county_fips": str})
pop = pop.rename(columns={"Sex": "sex"})

# ---- Victims and offenders per 100,000 people of the same sex ----
people_by_sex = pop[(pop["Age group"] == "All") & pop["sex"].isin(["Men", "Women"])][GEO + ["sex", "Population"]]
crime = pd.read_csv(PROD / "FBI" / "crime_people.csv", dtype={"county_fips": str})
crime = crime[crime["sex"].isin(["Men", "Women"])]
crime_parts = []
for person in ["Victim", "Offender"]:
    c = (crime[crime["person_type"] == person].groupby(GEO + ["sex"], dropna=False)["person_count"].sum().reset_index())
    # start from every county so counties with no reported people still count in the population
    known = people_by_sex.merge(c[c["county_fips"].notna()].drop(columns="county"), on=["county_fips", "year", "sex"], how="left")
    unknown = c[c["county_fips"].isna()]   # county Unknown: counted in the state total, no population
    part = pd.concat([known, unknown], ignore_index=True)
    part["topic"] = person + "s"
    part["count"] = part["person_count"].fillna(0)
    part["population"] = part["Population"]
    part["per"] = 100_000
    crime_parts.append(part)

# ---- Bachelor's degrees (count only) ----
edu = pd.read_csv(PROD / "ACS" / "population_education.csv", dtype={"county_fips": str})
edu = edu[(edu["Education level"] == "Bachelor's degree") & edu["Sex"].isin(["Men", "Women"])]
bach = edu[GEO].assign(sex=edu["Sex"], topic="Bachelor's degrees", count=edu["Population 25+"])

# ---- Births per 1,000 women in the age group ----
# WISH age group -> the ACS age rows for women of that age.
# Under 15 uses women 10-14 and 45+ uses women 45-49 (the usual birth-rate convention);
# "All ages" uses women 15-44 (the general fertility rate).
AGE_ROWS = {
    "Under 15": ["10 to 14 years"],
    "15-17": ["15 to 17 years"],
    "18-19": ["18 and 19 years"],
    "20-24": ["20 years", "21 years", "22 to 24 years"],
    "25-29": ["25 to 29 years"],
    "30-34": ["30 to 34 years"],
    "35-39": ["35 to 39 years"],
    "40-44": ["40 to 44 years"],
    "45+": ["45 to 49 years"],
}
AGE_ROWS["All ages"] = [r for g in ["15-17", "18-19", "20-24", "25-29", "30-34", "35-39", "40-44"] for r in AGE_ROWS[g]]
AGE_SORT = {g: i for i, g in enumerate(AGE_ROWS)}

women = pop[(pop["sex"] == "Women") & (pop["Is total"] == "No")]
women_by_group = pd.concat(
    [women[women["Age detail"].isin(rows)].groupby(GEO)["Population"].sum().reset_index().assign(age_group=g)
     for g, rows in AGE_ROWS.items()], ignore_index=True)

births = pd.read_csv(PROD / "WISH" / "births.csv", dtype={"county_fips": str})
births = births[births["breakdown"].isin(["Age", "Total"])].copy()
births["age_group"] = births["age_group"].replace({"All": "All ages"})
births = births[births["age_group"].isin(AGE_ROWS)]   # drops Unknown age
births = births.groupby(GEO + ["age_group"], dropna=False)["birth_count"].sum().reset_index()   # suppressed counts count as 0
known = women_by_group.merge(births[births["county_fips"].notna()].drop(columns="county"),
                             on=["county_fips", "year", "age_group"], how="left")
birth = pd.concat([known, births[births["county_fips"].isna()]], ignore_index=True)
birth = birth.assign(topic="Births", sex="Women", count=birth["birth_count"].fillna(0),
                     population=birth["Population"], per=1_000)
birth["age_sort"] = birth["age_group"].map(AGE_SORT)

# ---- one table ----
COLS = ["topic", "county", "county_fips", "year", "sex", "age_group", "age_sort", "count", "population", "per"]
ts = pd.concat(crime_parts + [bach, birth], ignore_index=True).reindex(columns=COLS)
ts["county"] = ts["county"].fillna("Unknown")
for c in ["count", "population", "per", "age_sort"]:
    ts[c] = ts[c].astype("Int64")

# checks: every county-year present for every topic, and no duplicate rows
keys = ["topic", "county", "year", "sex", "age_group"]
assert not ts.duplicated(keys).any(), "duplicate rows"
counts = ts[ts["county"] != "Unknown"].groupby(["topic", "year"])["county_fips"].nunique()
assert (counts == 72).all(), counts[counts != 72]

OUT.parent.mkdir(parents=True, exist_ok=True)
ts.sort_values(["topic", "year", "county", "sex", "age_sort"]).to_csv(OUT, index=False)

state = ts.groupby(["topic", "sex", "age_group", "year"], dropna=False)[["count", "population"]].sum()
state["rate"] = (state["count"] / state["population"] * ts.groupby(["topic"])["per"].first().reindex(
    state.index.get_level_values("topic")).values).round(1)
print(state.to_string())
print(f"\nSaved {OUT} ({len(ts):,} rows)")
