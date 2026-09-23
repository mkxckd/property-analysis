"""
Duplicate-listing detection (entity resolution) for SouqPulse.

The problem: the same property gets re-posted, often by a different agent,
with a reworded title and a slightly different price. Left unhandled this
inflates supply counts, skews the price index, and erodes buyer trust
(the exact issue classifieds platforms build "verified/trusted seller"
badges to fight).

Approach: blocking + weighted similarity, not a single global comparison.

  1. BLOCKING - comparing every listing to every other listing is O(n^2)
     and mostly wasted work, since a Studio in Mirdif can never duplicate
     a Villa in Palm Jumeirah. We only compare listings that share the
     same area, property_type, and bedroom count. This cuts the number
     of pairwise comparisons by several orders of magnitude on realistic
     data while losing no true duplicates (duplicates always share these
     fields, by construction of the problem itself).

  2. SIMILARITY - within a block, pairs are scored on three signals:
       - text similarity: TF-IDF cosine similarity over title+description
       - price similarity: closeness of monthly price (duplicates drift
         in price, they don't match exactly)
       - size similarity: closeness of size_sqft
     combined into one weighted score.

  3. CLUSTERING - pairs above the match threshold are merged with a
     union-find structure so that A-B and B-C duplicate pairs collapse
     into a single {A, B, C} cluster, not two separate pairs.

Evaluated in evaluate.py against the ground-truth dup_group_truth column
that generate_data.py wrote, using precision/recall/F1 on pairs.
"""
import itertools
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

TEXT_WEIGHT = 0.45
PRICE_WEIGHT = 0.30
SIZE_WEIGHT = 0.25
MATCH_THRESHOLD = 0.55

# A pair can only be SCORED if it clears ALL of these gates first. Any one
# signal alone is too weak in a dense listings block: geo-proximity alone
# collides by chance when many listings are scattered across one
# neighborhood; price/size alone collide because a whole block's range is
# narrow to begin with; text alone collides because boilerplate phrasing
# repeats across unrelated listings. Requiring geo AND price AND size to
# all independently look like "the same unit" before text similarity is
# even considered is what keeps coincidental collisions out -- three
# independent signals agreeing by chance is far rarer than any one of them
# agreeing by chance.
MAX_DISTANCE_METERS_GATE = 300
MIN_PRICE_SIM_GATE = 0.90
MIN_SIZE_SIM_GATE = 0.95


class UnionFind:
    def __init__(self, ids):
        self.parent = {i: i for i in ids}

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[ra] = rb


@dataclass
class DedupResult:
    pairs: list = field(default_factory=list)          # (id_a, id_b, score)
    clusters: dict = field(default_factory=dict)        # cluster_id -> [listing_ids]
    listing_to_cluster: dict = field(default_factory=dict)


def _block_key(row):
    return (row["area"], row["property_type"], row["bedrooms"])


def _price_similarity(p1, p2):
    return 1 - min(abs(p1 - p2) / max(p1, p2), 1.0)


def _size_similarity(s1, s2):
    return 1 - min(abs(s1 - s2) / max(s1, s2), 1.0)


def _distance_meters(lat1, lon1, lat2, lon2):
    """Flat-earth approximation, accurate enough at city scale."""
    dlat_m = (lat1 - lat2) * 111_000
    dlon_m = (lon1 - lon2) * 111_000 * np.cos(np.radians((lat1 + lat2) / 2))
    return float(np.hypot(dlat_m, dlon_m))


def find_duplicates(df: pd.DataFrame) -> DedupResult:
    df = df.reset_index(drop=True)
    df["_text"] = (df["title"].fillna("") + " " + df["description"].fillna(""))
    blocks = df.groupby([df["area"], df["property_type"], df["bedrooms"]]).groups

    result = DedupResult()
    uf = UnionFind(df["listing_id"].tolist())

    for block_key, idx in blocks.items():
        idx = list(idx)
        if len(idx) < 2:
            continue

        sub = df.loc[idx]
        vectorizer = TfidfVectorizer(stop_words="english")
        try:
            tfidf = vectorizer.fit_transform(sub["_text"])
            text_sim = cosine_similarity(tfidf)
        except ValueError:
            # block's text was empty/degenerate after stopword removal
            text_sim = np.eye(len(idx))

        prices = sub["price_aed_month"].values
        sizes = sub["size_sqft"].values
        lats = sub["lat"].values
        lons = sub["lon"].values
        listing_ids = sub["listing_id"].values

        for i, j in itertools.combinations(range(len(idx)), 2):
            dist_m = _distance_meters(lats[i], lons[i], lats[j], lons[j])
            if dist_m > MAX_DISTANCE_METERS_GATE:
                continue

            p_sim = _price_similarity(prices[i], prices[j])
            if p_sim < MIN_PRICE_SIM_GATE:
                continue

            s_sim = _size_similarity(sizes[i], sizes[j])
            if s_sim < MIN_SIZE_SIM_GATE:
                continue

            t_sim = text_sim[i, j]
            score = TEXT_WEIGHT * t_sim + PRICE_WEIGHT * p_sim + SIZE_WEIGHT * s_sim

            if score >= MATCH_THRESHOLD:
                id_a, id_b = int(listing_ids[i]), int(listing_ids[j])
                result.pairs.append((id_a, id_b, round(float(score), 4)))
                uf.union(id_a, id_b)

    # materialize clusters from union-find
    cluster_members = {}
    for listing_id in df["listing_id"]:
        root = uf.find(listing_id)
        cluster_members.setdefault(root, []).append(int(listing_id))

    cluster_id = 0
    for root, members in cluster_members.items():
        if len(members) > 1:
            cid = f"cluster_{cluster_id}"
            result.clusters[cid] = sorted(members)
            for m in members:
                result.listing_to_cluster[m] = cid
            cluster_id += 1

    return result
