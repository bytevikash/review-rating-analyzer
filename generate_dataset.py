import pandas as pd
import random

random.seed(42)

# --------------------------------------------------
# Positive Reviews
# --------------------------------------------------

positive_reviews = [
    "This product is absolutely amazing",
    "I love this product",
    "Excellent quality and great performance",
    "Very good product and excellent quality",
    "Amazing product and fast delivery",
    "Really happy with this purchase",
    "The quality is excellent",
    "Great product for the price",
    "Very satisfied with my purchase",
    "The product works perfectly",
    "Fantastic product and good quality",
    "I am very happy with this product",
    "Excellent performance",
    "Good quality and fast delivery",
    "Highly satisfied with the product",
    "This product exceeded my expectations",
    "Wonderful product and great service",
    "The product is reliable and useful",
    "Amazing quality and fast delivery",
    "Very impressive product",
    "Very nice product",
    "Nice product",
    "Really nice product",
    "Very good product",
    "Really good product",
    "Great product",
    "Excellent product",
    "Awesome product",
    "I really like this product",
    "I am happy with this product",
    "This is a great purchase",
    "This product is worth the money",
    "The product quality is very good",
    "The product quality is amazing",
    "The product works very well",
    "I am completely satisfied",
    "This was a wonderful purchase",
    "The product is fantastic",
    "The product is perfect",
    "I would definitely recommend this product"
]


# --------------------------------------------------
# Neutral Reviews
# --------------------------------------------------

neutral_reviews = [
    "The product is okay",
    "The product works fine",
    "It is an average product",
    "The quality is average",
    "The product is acceptable",
    "It is okay for the price",
    "Nothing special about this product",
    "The product is neither good nor bad",
    "It works as expected",
    "The quality could be better",
    "The product is satisfactory",
    "The price is okay",
    "It is a decent product",
    "Average experience with this product",
    "The product is fine for normal use",
    "It does the basic job",
    "The product is reasonably good",
    "The experience was normal",
    "It is an ordinary product",
    "The product meets basic expectations",
    "The product is fine",
    "The product is okay for the price",
    "It works normally",
    "Average quality product",
    "The product is acceptable for basic use",
    "The experience was average",
    "It is neither impressive nor disappointing",
    "The product performs normally",
    "The product is decent",
    "It is a standard product"
]


# --------------------------------------------------
# Negative Reviews
# --------------------------------------------------

negative_reviews = [
    "Very poor quality and waste of money",
    "I hate this product",
    "The product is terrible",
    "Very disappointing experience",
    "The product stopped working",
    "Poor quality product",
    "I am not satisfied with this purchase",
    "The product is completely useless",
    "Terrible performance",
    "Very bad product",
    "The quality is extremely poor",
    "Waste of money",
    "The product broke after one day",
    "Very disappointed with the product",
    "The product does not work properly",
    "Bad quality and poor performance",
    "I regret buying this product",
    "The product is unreliable",
    "Poor experience with this product",
    "I would not recommend this product",
    "Very bad quality",
    "Really bad product",
    "Terrible product",
    "Extremely disappointing product",
    "The product is awful",
    "I am unhappy with this purchase",
    "The product is not good",
    "The product quality is terrible",
    "The product failed quickly",
    "This product is a waste of money"
]


ratings = {
    "positive": [4, 5],
    "neutral": [3],
    "negative": [1, 2]
}

categories = [
    "Electronics",
    "Clothing",
    "Home",
    "Books",
    "Beauty"
]

variations = [
    "",
    " Overall, I am satisfied.",
    " I would buy it again.",
    " The experience was good.",
    " I expected more.",
    " After using it for some time, this is my opinion.",
    " The product arrived on time.",
    " This is my honest review."
]


# --------------------------------------------------
# Create Dataset
# --------------------------------------------------

data = []
review_id = 1

review_groups = [
    ("positive", positive_reviews),
    ("neutral", neutral_reviews),
    ("negative", negative_reviews)
]

for sentiment, review_list in review_groups:

    for i in range(200):

        review = random.choice(review_list)

        # Add a variation
        review += random.choice(variations)

        rating = random.choice(ratings[sentiment])
        category = random.choice(categories)

        data.append({
            "review_id": review_id,
            "review": review,
            "rating": rating,
            "sentiment": sentiment,
            "date": f"2026-{random.randint(1, 3):02d}-{random.randint(1, 28):02d}",
            "category": category
        })

        review_id += 1


# --------------------------------------------------
# Shuffle Dataset
# --------------------------------------------------

df = pd.DataFrame(data)

df = df.sample(
    frac=1,
    random_state=42
).reset_index(drop=True)


# --------------------------------------------------
# Save Dataset
# --------------------------------------------------

df.to_csv(
    "data/reviews.csv",
    index=False
)

print("Dataset created successfully!")
print("Total reviews:", len(df))

print("\nSentiment distribution:")
print(df["sentiment"].value_counts())