"""
Data ingestion script to load products from dataset into Qdrant.

Usage:
    python scripts/ingest_data.py --dataset ../dataset/sampled_posts
"""

import json
import sys
import argparse
from pathlib import Path
from datetime import datetime
import uuid

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from app.models.schemas import Product
from app.services.retriever import MultiQueryRetriever
from app.core.dependencies import get_qdrant


def load_posts_from_json(file_path: Path) -> list:
    """Load posts from JSON file"""
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def post_to_product(post: dict, channel_name: str) -> Product:
    """Convert a post to a Product"""
    # Extract product info from post text
    text = post.get("text", "")
    post_id = str(post.get("id", uuid.uuid4()))

    # Simple parsing - in production you'd use more sophisticated extraction
    # Extract price
    price = None
    if "*HE'F" in text or "EÌDÌHF" in text:
        # Try to extract price (simplified)
        import re
        price_match = re.search(r"(\d+(?:\d+)*)\s*(?:*HE'F|EÌDÌHF)", text)
        if price_match:
            price_str = price_match.group(1).replace("", "")
            price = int(price_str)
            if "EÌDÌHF" in text:
                price *= 1_000_000

    # Create product
    product = Product(
        id=f"{channel_name}_{post_id}",
        name=text[:100] if text else f"E-5HD {post_id}",  # Use first 100 chars as name
        description=text,
        price=price or 0,
        currency="*HE'F",
        availability=True,
        metadata={
            "channel": channel_name,
            "post_id": post_id,
            "date": post.get("date"),
            "views": post.get("views"),
        },
        created_at=datetime.now(),
        updated_at=datetime.now(),
    )

    return product


def ingest_from_directory(dataset_dir: Path, qdrant_client):
    """Ingest all JSON files from dataset directory"""
    retriever = MultiQueryRetriever(qdrant_client)

    json_files = list(dataset_dir.glob("*.json"))
    print(f"Found {len(json_files)} JSON files")

    total_products = 0

    for json_file in json_files:
        print(f"\nProcessing {json_file.name}...")
        channel_name = json_file.stem

        try:
            posts = load_posts_from_json(json_file)
            print(f"  Found {len(posts)} posts")

            for post in posts:
                try:
                    product = post_to_product(post, channel_name)
                    retriever.add_product(product)
                    total_products += 1
                except Exception as e:
                    print(f"  Error processing post {post.get('id')}: {e}")

            print(f"   Ingested {len(posts)} products from {channel_name}")

        except Exception as e:
            print(f"   Error processing file {json_file.name}: {e}")

    print(f"\n Total products ingested: {total_products}")


def main():
    parser = argparse.ArgumentParser(description="Ingest posts from dataset into Qdrant")
    parser.add_argument(
        "--dataset",
        type=str,
        default="../dataset/sampled_posts",
        help="Path to dataset directory containing JSON files",
    )
    args = parser.parse_args()

    dataset_dir = Path(args.dataset)
    if not dataset_dir.exists():
        print(f"Error: Dataset directory not found: {dataset_dir}")
        sys.exit(1)

    print("Connecting to Qdrant...")
    qdrant_client = get_qdrant()

    print(f"Starting ingestion from {dataset_dir}")
    ingest_from_directory(dataset_dir, qdrant_client)

    print("\n Ingestion complete!")


if __name__ == "__main__":
    main()
