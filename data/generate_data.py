"""
SouqPulse synthetic data generator.

Generates a fictional classifieds-platform property listings dataset with
two kinds of injected ground truth, so the pipeline's outputs can be
measured against a known answer instead of eyeballed:

  1. Duplicate listings: the same underlying property re-posted by a
     different agent with a reworded title/description and a slightly
     different price, days later. Real classifieds platforms deal with
     this constantly (the same apartment posted by three different
     brokers). Ground truth pairs are written to duplicates_truth.csv.

  2. Suspicious-pricing listings: a small number of listings priced far
     below the going rate for their area/type, the kind of "too good to
     be true" pattern platforms use as one fraud-triage signal. Ground
     truth flags are written into the main table.

None of this is real Bayut/Dubizzle data. Area names are real Dubai
neighborhoods (public geography), but every listing, agent, price and
photo count below is randomly generated.
"""
import csv
import os
import random
from datetime import datetime, timedelta

random.seed(42)

OUT_DIR = os.path.join(os.path.dirname(__file__), "raw")
os.makedirs(OUT_DIR, exist_ok=True)

          # name                lat        lon    annual AED / sqft (rent baseline)
AREAS = [
    ("Dubai Marina", 25.0805, 55.1403, 105),
    ("Jumeirah Village Circle", 25.0575, 55.2093, 62),
    ("Business Bay", 25.1872, 55.2631, 92),
    ("Downtown Dubai", 25.1972, 55.2744, 130),
    ("Jumeirah Lake Towers", 25.0693, 55.1424, 88),
    ("Arabian Ranches", 25.0512, 55.2704, 78),
    ("Al Furjan", 25.0234, 55.1487, 60),
    ("Dubai Silicon Oasis", 25.1234, 55.3823, 48),
    ("Palm Jumeirah", 25.1124, 55.1390, 155),
    ("Dubai Sports City", 25.0453, 55.2178, 52),
    ("Mirdif", 25.2168, 55.4211, 55),
    ("Discovery Gardens", 25.0398, 55.1444, 58),
]

PROPERTY_TYPES = {
    "Studio": (1, 1, (380, 550)),
    "1BR Apartment": (1, 2, (650, 950)),
    "2BR Apartment": (2, 3, (950, 1400)),
    "3BR Apartment": (3, 4, (1400, 2100)),
    "Townhouse": (3, 4, (1800, 2600)),
    "Villa": (4, 6, (2800, 5200)),
}

AGENTS = [
    "Layla Haddad", "Omar Farouk", "Priya Nair", "Michael Costa",
    "Fatima Al-Suwaidi", "Daniel Reyes", "Anjali Menon", "Yusuf Karimi",
    "Sofia Marchetti", "Ravi Shankar", "Noor Abdullah", "Chen Wei",
    "Grace Okafor", "Hassan Ali", "Elena Popescu",
]

TITLE_TEMPLATES = [
    "Spacious {ptype} in {area} | Ready to Move",
    "{ptype} for Rent in {area} - Prime Location",
    "Stunning {ptype} | {area} | Vacant Soon",
    "Best Deal! {ptype} in {area}",
    "{ptype} with Full Amenities - {area}",
    "Upgraded {ptype} in {area}, Motivated Seller",
    "Exclusive {ptype} in {area} | High Floor",
    "Chiller Free {ptype} in {area}",
]

REWORD_TEMPLATES = [
    "Must See! {ptype} Available in {area}",
    "{area} {ptype} - Negotiable Price",
    "Brand New Listing: {ptype} in {area}",
    "Hot Deal - {ptype} in {area}, Call Now",
    "{ptype} in {area} | Well Maintained",
]

DESC_BITS = [
    "Close to metro station and community mall.",
    "Recently renovated kitchen and bathrooms.",
    "Balcony with community view.",
    "Covered parking included.",
    "Near international schools.",
    "Built-in wardrobes throughout.",
    "Shared pool and gym access.",
    "Pet friendly building.",
    "Available for immediate move-in.",
    "Landlord maintained, no agency fees.",
]

START_DATE = datetime(2026, 1, 1)
DAYS_SPAN = 240
N_BASE_LISTINGS = 2200
DUPLICATE_RATE = 0.14
SUSPICIOUS_RATE = 0.03


def random_date():
    return START_DATE + timedelta(days=random.randint(0, DAYS_SPAN))


def make_price(ptype, area_annual_psf, size, furnished, building_age, floor):
    """area_annual_psf is AED of annual rent per sqft for that area.
    Dubai listings are conventionally priced as annual rent; we compute
    that and then derive the monthly figure actually stored.

    Furnished status, building age, and floor all have small, real
    (but noisy) effects on price -- exactly the kind of secondary
    signal a valuation model should pick up on beyond the obvious
    area/size/type relationship."""
    psf = area_annual_psf * random.uniform(0.85, 1.2)
    annual_rent = psf * size

    if furnished:
        annual_rent *= random.uniform(1.03, 1.09)
    annual_rent *= (1 - building_age * random.uniform(0.003, 0.007))
    if floor and floor > 0:
        annual_rent *= (1 + min(floor, 40) * random.uniform(0.0008, 0.0018))

    monthly_rent = annual_rent / 12 * random.uniform(0.97, 1.03)
    return round(monthly_rent / 250) * 250


def make_title(ptype, area, template_pool):
    return random.choice(template_pool).format(ptype=ptype, area=area)


def make_description():
    n = random.randint(2, 4)
    return " ".join(random.sample(DESC_BITS, n))


def make_base_listing(listing_id):
    area, lat, lon, base_psf = random.choice(AREAS)
    ptype = random.choice(list(PROPERTY_TYPES.keys()))
    beds_min, beds_max, size_range = PROPERTY_TYPES[ptype]
    size = random.randint(*size_range)
    furnished = random.random() < 0.38
    building_age = random.randint(0, 22)
    is_tower_type = ptype in ("Studio", "1BR Apartment", "2BR Apartment", "3BR Apartment")
    floor = random.randint(1, 42) if is_tower_type else 0
    price = make_price(ptype, base_psf, size, furnished, building_age, floor)
    listed = random_date()
    agent = random.choice(AGENTS)
    return {
        "listing_id": listing_id,
        "title": make_title(ptype, area, TITLE_TEMPLATES),
        "description": make_description(),
        "property_type": ptype,
        "area": area,
        "lat": round(lat + random.uniform(-0.025, 0.025), 5),
        "lon": round(lon + random.uniform(-0.025, 0.025), 5),
        "bedrooms": random.randint(beds_min, beds_max) if ptype != "Studio" else 0,
        "size_sqft": size,
        "furnished": int(furnished),
        "building_age_years": building_age,
        "floor": floor,
        "price_aed_month": int(price),
        "agent_name": agent,
        "listed_date": listed.strftime("%Y-%m-%d"),
        "status": "active",
        "is_suspicious_truth": 0,
        "dup_group_truth": listing_id,  # own group unless later marked as dup
    }


def make_duplicate(original, new_id, dup_index):
    area, lat, lon, base_psf = next(a for a in AREAS if a[0] == original["area"])
    price_drift = random.uniform(-0.06, 0.06)
    new_price = int(round(original["price_aed_month"] * (1 + price_drift) / 500) * 500)
    delay = random.randint(2, 21)
    new_date = datetime.strptime(original["listed_date"], "%Y-%m-%d") + timedelta(days=delay)
    other_agents = [a for a in AGENTS if a != original["agent_name"]]
    return {
        "listing_id": new_id,
        "title": make_title(original["property_type"], original["area"], REWORD_TEMPLATES),
        "description": make_description(),
        "property_type": original["property_type"],
        "area": original["area"],
        "lat": round(original["lat"] + random.uniform(-0.001, 0.001), 5),
        "lon": round(original["lon"] + random.uniform(-0.001, 0.001), 5),
        "bedrooms": original["bedrooms"],
        "size_sqft": original["size_sqft"] + random.choice([-15, 0, 0, 15]),
        "furnished": original["furnished"],
        "building_age_years": original["building_age_years"],
        "floor": original["floor"],
        "price_aed_month": new_price,
        "agent_name": random.choice(other_agents),
        "listed_date": new_date.strftime("%Y-%m-%d"),
        "status": "active",
        "is_suspicious_truth": 0,
        "dup_group_truth": original["dup_group_truth"],
    }


def make_suspicious(listing_id):
    row = make_base_listing(listing_id)
    row["price_aed_month"] = int(row["price_aed_month"] * random.uniform(0.35, 0.55))
    row["title"] = "URGENT SALE " + row["title"]
    row["is_suspicious_truth"] = 1
    return row


def generate():
    rows = []
    next_id = 1

    n_dupe_sources = int(N_BASE_LISTINGS * DUPLICATE_RATE)
    n_suspicious = int(N_BASE_LISTINGS * SUSPICIOUS_RATE)

    base_rows = []
    for _ in range(N_BASE_LISTINGS):
        r = make_base_listing(next_id)
        base_rows.append(r)
        next_id += 1

    rows.extend(base_rows)

    dupe_sources = random.sample(base_rows, n_dupe_sources)
    for i, src in enumerate(dupe_sources):
        dup = make_duplicate(src, next_id, i)
        rows.append(dup)
        next_id += 1
        # ~18% of duplicated listings get re-posted a THIRD time too,
        # to make sure the dedup engine has to handle clusters of >2, not
        # just pairs.
        if random.random() < 0.18:
            dup2 = make_duplicate(src, next_id, i)
            rows.append(dup2)
            next_id += 1

    for _ in range(n_suspicious):
        rows.append(make_suspicious(next_id))
        next_id += 1

    random.shuffle(rows)

    fieldnames = list(rows[0].keys())
    out_path = os.path.join(OUT_DIR, "listings.csv")
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    n_dupe_groups = sum(1 for r in rows if r["dup_group_truth"] != r["listing_id"]) 
    print(f"Wrote {len(rows)} listings to {out_path}")
    print(f"  base listings:        {N_BASE_LISTINGS}")
    print(f"  duplicate postings:   {len(rows) - N_BASE_LISTINGS - n_suspicious}")
    print(f"  suspicious-price:     {n_suspicious}")
    return out_path


if __name__ == "__main__":
    generate()
