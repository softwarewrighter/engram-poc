"""
Generate exact-match key→value dataset for Engram memory evaluation.

This creates data where hash-based memory should excel:
- Same input always maps to same output
- No generalization needed - pure recall
- Deterministic lookup is the optimal solution

Usage:
    python -m src.data_gen.generate_exact_match --num-examples 5000
"""

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple


# Knowledge bases for exact-match tasks
CAPITALS = {
    "France": "Paris",
    "Germany": "Berlin",
    "Japan": "Tokyo",
    "Italy": "Rome",
    "Spain": "Madrid",
    "Canada": "Ottawa",
    "Australia": "Canberra",
    "Brazil": "Brasilia",
    "India": "New Delhi",
    "China": "Beijing",
    "Russia": "Moscow",
    "Egypt": "Cairo",
    "Mexico": "Mexico City",
    "Argentina": "Buenos Aires",
    "South Korea": "Seoul",
    "Thailand": "Bangkok",
    "Vietnam": "Hanoi",
    "Poland": "Warsaw",
    "Netherlands": "Amsterdam",
    "Belgium": "Brussels",
    "Sweden": "Stockholm",
    "Norway": "Oslo",
    "Denmark": "Copenhagen",
    "Finland": "Helsinki",
    "Greece": "Athens",
    "Turkey": "Ankara",
    "Iran": "Tehran",
    "Iraq": "Baghdad",
    "Pakistan": "Islamabad",
    "Indonesia": "Jakarta",
}

ELEMENTS = {
    "H": ("Hydrogen", "1"),
    "He": ("Helium", "2"),
    "Li": ("Lithium", "3"),
    "Be": ("Beryllium", "4"),
    "B": ("Boron", "5"),
    "C": ("Carbon", "6"),
    "N": ("Nitrogen", "7"),
    "O": ("Oxygen", "8"),
    "F": ("Fluorine", "9"),
    "Ne": ("Neon", "10"),
    "Na": ("Sodium", "11"),
    "Mg": ("Magnesium", "12"),
    "Al": ("Aluminum", "13"),
    "Si": ("Silicon", "14"),
    "P": ("Phosphorus", "15"),
    "S": ("Sulfur", "16"),
    "Cl": ("Chlorine", "17"),
    "Ar": ("Argon", "18"),
    "K": ("Potassium", "19"),
    "Ca": ("Calcium", "20"),
    "Fe": ("Iron", "26"),
    "Cu": ("Copper", "29"),
    "Zn": ("Zinc", "30"),
    "Ag": ("Silver", "47"),
    "Au": ("Gold", "79"),
    "Pb": ("Lead", "82"),
    "U": ("Uranium", "92"),
}

PORTS = {
    "HTTP": "80",
    "HTTPS": "443",
    "SSH": "22",
    "FTP": "21",
    "SMTP": "25",
    "DNS": "53",
    "MySQL": "3306",
    "PostgreSQL": "5432",
    "MongoDB": "27017",
    "Redis": "6379",
    "Elasticsearch": "9200",
    "RabbitMQ": "5672",
    "Memcached": "11211",
    "Cassandra": "9042",
    "Neo4j": "7474",
    "InfluxDB": "8086",
    "Grafana": "3000",
    "Prometheus": "9090",
    "Jenkins": "8080",
    "Docker": "2375",
}

HTTP_CODES = {
    "200": "OK",
    "201": "Created",
    "204": "No Content",
    "301": "Moved Permanently",
    "302": "Found",
    "304": "Not Modified",
    "400": "Bad Request",
    "401": "Unauthorized",
    "403": "Forbidden",
    "404": "Not Found",
    "405": "Method Not Allowed",
    "408": "Request Timeout",
    "429": "Too Many Requests",
    "500": "Internal Server Error",
    "502": "Bad Gateway",
    "503": "Service Unavailable",
    "504": "Gateway Timeout",
}

UNITS = {
    "1 kilometer": "1000 meters",
    "1 mile": "1.609 kilometers",
    "1 kilogram": "1000 grams",
    "1 pound": "0.453 kilograms",
    "1 liter": "1000 milliliters",
    "1 gallon": "3.785 liters",
    "1 hour": "3600 seconds",
    "1 day": "86400 seconds",
    "1 week": "604800 seconds",
    "1 byte": "8 bits",
    "1 kilobyte": "1024 bytes",
    "1 megabyte": "1024 kilobytes",
    "1 gigabyte": "1024 megabytes",
    "1 terabyte": "1024 gigabytes",
}

ACRONYMS = {
    "API": "Application Programming Interface",
    "CPU": "Central Processing Unit",
    "GPU": "Graphics Processing Unit",
    "RAM": "Random Access Memory",
    "ROM": "Read Only Memory",
    "SSD": "Solid State Drive",
    "HDD": "Hard Disk Drive",
    "URL": "Uniform Resource Locator",
    "HTML": "HyperText Markup Language",
    "CSS": "Cascading Style Sheets",
    "JSON": "JavaScript Object Notation",
    "XML": "eXtensible Markup Language",
    "SQL": "Structured Query Language",
    "TCP": "Transmission Control Protocol",
    "UDP": "User Datagram Protocol",
    "HTTP": "HyperText Transfer Protocol",
    "HTTPS": "HyperText Transfer Protocol Secure",
    "SSH": "Secure Shell",
    "FTP": "File Transfer Protocol",
    "DNS": "Domain Name System",
    "IP": "Internet Protocol",
    "LAN": "Local Area Network",
    "WAN": "Wide Area Network",
    "VPN": "Virtual Private Network",
    "AI": "Artificial Intelligence",
    "ML": "Machine Learning",
    "NLP": "Natural Language Processing",
    "CNN": "Convolutional Neural Network",
    "RNN": "Recurrent Neural Network",
    "GAN": "Generative Adversarial Network",
}


def generate_capital_examples() -> List[Dict]:
    """Generate country→capital examples."""
    examples = []
    for country, capital in CAPITALS.items():
        examples.append({
            "messages": [
                {"role": "user", "content": f"CAPITAL:{country}"},
                {"role": "assistant", "content": f" {capital}"}
            ],
            "category": "capital",
            "key": f"CAPITAL:{country}",
            "value": capital,
        })
    return examples


def generate_element_examples() -> List[Dict]:
    """Generate element symbol→name and symbol→number examples."""
    examples = []
    for symbol, (name, number) in ELEMENTS.items():
        # Symbol → Name
        examples.append({
            "messages": [
                {"role": "user", "content": f"ELEMENT_NAME:{symbol}"},
                {"role": "assistant", "content": f" {name}"}
            ],
            "category": "element_name",
            "key": f"ELEMENT_NAME:{symbol}",
            "value": name,
        })
        # Symbol → Number
        examples.append({
            "messages": [
                {"role": "user", "content": f"ELEMENT_NUM:{symbol}"},
                {"role": "assistant", "content": f" {number}"}
            ],
            "category": "element_number",
            "key": f"ELEMENT_NUM:{symbol}",
            "value": number,
        })
    return examples


def generate_port_examples() -> List[Dict]:
    """Generate service→port examples."""
    examples = []
    for service, port in PORTS.items():
        examples.append({
            "messages": [
                {"role": "user", "content": f"PORT:{service}"},
                {"role": "assistant", "content": f" {port}"}
            ],
            "category": "port",
            "key": f"PORT:{service}",
            "value": port,
        })
    return examples


def generate_http_code_examples() -> List[Dict]:
    """Generate HTTP code→meaning examples."""
    examples = []
    for code, meaning in HTTP_CODES.items():
        examples.append({
            "messages": [
                {"role": "user", "content": f"HTTP:{code}"},
                {"role": "assistant", "content": f" {meaning}"}
            ],
            "category": "http_code",
            "key": f"HTTP:{code}",
            "value": meaning,
        })
    return examples


def generate_unit_examples() -> List[Dict]:
    """Generate unit conversion examples."""
    examples = []
    for unit_from, unit_to in UNITS.items():
        examples.append({
            "messages": [
                {"role": "user", "content": f"CONVERT:{unit_from}"},
                {"role": "assistant", "content": f" {unit_to}"}
            ],
            "category": "unit_conversion",
            "key": f"CONVERT:{unit_from}",
            "value": unit_to,
        })
    return examples


def generate_acronym_examples() -> List[Dict]:
    """Generate acronym→expansion examples."""
    examples = []
    for acronym, expansion in ACRONYMS.items():
        examples.append({
            "messages": [
                {"role": "user", "content": f"ACRONYM:{acronym}"},
                {"role": "assistant", "content": f" {expansion}"}
            ],
            "category": "acronym",
            "key": f"ACRONYM:{acronym}",
            "value": expansion,
        })
    return examples


def generate_synthetic_kv(num_examples: int, key_prefix: str = "KEY") -> List[Dict]:
    """Generate synthetic key→value pairs for additional training data."""
    examples = []
    # Use deterministic values so same key always maps to same value
    for i in range(num_examples):
        key = f"{key_prefix}_{i:05d}"
        # Generate a deterministic "random" value based on key
        random.seed(hash(key) % (2**32))
        value_type = random.choice(["number", "word", "code"])

        if value_type == "number":
            value = str(random.randint(1, 99999))
        elif value_type == "word":
            words = ["alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta"]
            value = random.choice(words) + "_" + str(random.randint(1, 999))
        else:
            value = f"0x{random.randint(0, 0xFFFFFF):06X}"

        examples.append({
            "messages": [
                {"role": "user", "content": f"{key_prefix}:{key}"},
                {"role": "assistant", "content": f" {value}"}
            ],
            "category": "synthetic",
            "key": f"{key_prefix}:{key}",
            "value": value,
        })

    return examples


def generate_dataset(
    num_examples: int = 5000,
    include_synthetic: bool = True,
    seed: int = 42,
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """Generate train/valid/test splits for exact-match dataset."""
    random.seed(seed)

    # Collect all real-world examples
    all_examples = []
    all_examples.extend(generate_capital_examples())
    all_examples.extend(generate_element_examples())
    all_examples.extend(generate_port_examples())
    all_examples.extend(generate_http_code_examples())
    all_examples.extend(generate_unit_examples())
    all_examples.extend(generate_acronym_examples())

    print(f"Generated {len(all_examples)} real-world examples")

    # Add synthetic examples if needed
    if include_synthetic and len(all_examples) < num_examples:
        synthetic_count = num_examples - len(all_examples)
        synthetic = generate_synthetic_kv(synthetic_count, "SYN")
        all_examples.extend(synthetic)
        print(f"Added {synthetic_count} synthetic examples")

    # Shuffle
    random.shuffle(all_examples)

    # Split: 80% train, 10% valid, 10% test
    n = len(all_examples)
    train_end = int(0.8 * n)
    valid_end = int(0.9 * n)

    train = all_examples[:train_end]
    valid = all_examples[train_end:valid_end]
    test = all_examples[valid_end:]

    # For exact-match evaluation, we want to test on SEEN keys
    # So we duplicate some training examples into test
    # This tests pure recall, not generalization
    seen_test = random.sample(train, min(len(test), len(train) // 4))
    test.extend(seen_test)

    print(f"Split: {len(train)} train, {len(valid)} valid, {len(test)} test")
    print(f"  (test includes {len(seen_test)} seen examples for recall evaluation)")

    return train, valid, test


def main():
    parser = argparse.ArgumentParser(description="Generate exact-match dataset")
    parser.add_argument("--num-examples", type=int, default=500,
                        help="Target number of examples")
    parser.add_argument("--output-dir", default="data",
                        help="Output directory")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--no-synthetic", action="store_true",
                        help="Don't add synthetic examples")

    args = parser.parse_args()

    # Generate dataset
    train, valid, test = generate_dataset(
        num_examples=args.num_examples,
        include_synthetic=not args.no_synthetic,
        seed=args.seed,
    )

    # Save to files
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    for name, data in [("train_exact", train), ("valid_exact", valid), ("test_exact", test)]:
        path = output_dir / f"{name}.jsonl"
        with open(path, "w") as f:
            for example in data:
                f.write(json.dumps(example) + "\n")
        print(f"Saved {len(data)} examples to {path}")

    # Print category distribution
    print("\nCategory distribution (train):")
    categories = {}
    for ex in train:
        cat = ex.get("category", "unknown")
        categories[cat] = categories.get(cat, 0) + 1
    for cat, count in sorted(categories.items(), key=lambda x: -x[1]):
        print(f"  {cat}: {count}")


if __name__ == "__main__":
    main()
