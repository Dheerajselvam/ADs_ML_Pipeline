import csv
import random
from datetime import datetime


random.seed(42)


N = 20000
OUTPUT = "data/raw/synthetic_ads.csv"


ages = ["18-24","25-34","35-44","45+"]
geos = ["IN","US","UK","CN"]
interests = ["sports","tech","finance","fashion"]
creatives = ["banner","video","native"]
devices = ["mobile","desktop","tablet"]




def base_ctr(age, interest, creative, hour):
    ctr = 0.01
    if interest == "tech": ctr += 0.02
    if creative == "video": ctr += 0.015
    if age == "25-34": ctr += 0.01
    if 18 <= hour <= 22: ctr += 0.005
    return min(0.4, ctr)




with open(OUTPUT, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
    "impression_id", "timestamp", "user_id", "age_bucket",
    "geo", "interests", "ad_id", "advertiser_id",
    "creative_type", "device", "hour_of_day",
    "bid", "clicked", "revenue"
    ])


    for i in range(N):
        hour = random.randint(0, 23)
        age = random.choice(ages)
        interest = random.choice(interests)
        creative = random.choice(creatives)
        pctr = base_ctr(age, interest, creative, hour)


        clicked = 1 if random.random() < pctr else 0
        revenue = round(random.uniform(0.1, 3.0), 2) if clicked else 0.0


        writer.writerow([
            f"imp_{i}",
            datetime(2025, 12, 9, hour).isoformat(),
            random.randint(0, 499),
            age,
            random.choice(geos),
            interest,
            random.randint(0, 199),
            random.randint(0, 19),
            creative,
            random.choice(devices),
            hour,
            round(random.uniform(0.05, 2.0), 3),
            clicked,
            revenue
        ])


print(f"✅ Generated {N} rows at {OUTPUT}")