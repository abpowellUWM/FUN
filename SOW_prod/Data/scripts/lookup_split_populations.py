# SOW_prod/Data/scripts/lookup_split_populations.py
#
# One-time lookup: 2020 Census population of each city/village, split by the
# counties it sits in -- the same numbers behind
# reference_data/agency_county_splits.xlsx ("Raw" sheet). Used to add police
# departments that serve several counties but aren't in that file yet.
#
# Run from a terminal where CENSUS_API_KEY is set:
#   python3 Data/scripts/lookup_split_populations.py
# Writes Data/reference/split_population_lookup_2020.csv
# The key is read from the environment and never printed or saved.
import os
from pathlib import Path

import pandas as pd
import requests

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "reference" / "split_population_lookup_2020.csv"   # Data/reference
URL = "https://api.census.gov/data/2020/dec/pl"  # 2020 Census redistricting data, P1_001N = total population

# Places to look up. Eau Claire and Appleton are already in the splits file;
# they're included only to confirm the existing shares came from this table.
PLACES = {
    "Wisconsin Dells": "Wisconsin Dells city",
    "Brooklyn": "Brooklyn village",
    "Ashland": "Ashland city",
    "Eau Claire (check)": "Eau Claire city",
    "Appleton (check)": "Appleton city",
}

key = (os.getenv("CENSUS_API_KEY") or "").strip().strip("'\"")
if not key:
    raise SystemExit("CENSUS_API_KEY is not set in this terminal. Run: export CENSUS_API_KEY=your_key")


def get(params):
    r = requests.get(URL, params={**params, "key": key}, timeout=60)
    if "Invalid Key" in r.text[:500] or "missing_key" in r.url:
        raise SystemExit("Census rejected CENSUS_API_KEY.")
    if not r.ok:
        raise SystemExit(f"Census API error {r.status_code}: {r.text[:200]}")
    rows = r.json()
    return pd.DataFrame(rows[1:], columns=rows[0])


places = get({"get": "NAME", "for": "place:*", "in": "state:55"})
rows = []
for agency, place_name in PLACES.items():
    match = places[places["NAME"] == f"{place_name}, Wisconsin"]
    if len(match) != 1:
        raise SystemExit(f"Could not find exactly one place named '{place_name}, Wisconsin' (found {len(match)})")
    code = match["place"].iloc[0]
    parts = get({"get": "NAME,P1_001N", "for": "county (or part):*", "in": f"state:55 place:{code}"})
    parts["population"] = parts["P1_001N"].astype(int)
    total = parts["population"].sum()
    for _, p in parts.iterrows():
        rows.append({"agency_name": agency, "place": place_name, "place_fips": code,
                     "county_part": p["NAME"], "population": p["population"],
                     "share": round(p["population"] / total, 6) if total else None})

out = pd.DataFrame(rows)
OUT.parent.mkdir(exist_ok=True)
out.to_csv(OUT, index=False)
print(out.to_string(index=False))
print(f"\nSaved {OUT}")
