from datasets import load_dataset


def load_truthful_qa(num_samples=50, split="validation"):
    """Load samples from the TruthfulQA dataset."""
    if num_samples == "all":
        dataset = load_dataset("EleutherAI/truthful_qa_binary", split=split)
    else:
        dataset = load_dataset(
            "EleutherAI/truthful_qa_binary", split=f"{split}[:{num_samples}]"
        )
    return dataset


def load_faitheval(num_samples=50, split="test"):
    """Load samples from the FaithEval dataset."""
    dataset = load_dataset("Salesforce/FaithEval-counterfactual-v1.0", split=split)
    if num_samples != "all":
        dataset = dataset.select(range(int(num_samples)))
    return dataset


def load_spa_vl(num_samples=50, split="validation"):
    """Load samples from the SPA-VL dataset."""
    dataset = load_dataset("sqrti/SPA-VL", split)[split]
    if num_samples != "all":
        dataset = dataset.select(range(int(num_samples)))
    return dataset


def load_mmhal_bench(num_samples=50, split="test"):
    """Load samples from the MMHal-Bench dataset."""
    dataset = load_dataset("MMHal-Bench.py")[split]
    if num_samples != "all":
        dataset = dataset.select(range(int(num_samples)))
    return dataset


def load_data(dataset_name, num_samples=50, split="validation"):
    """Generic loader to fetch datasets by name."""
    if dataset_name == "truthfulqa":
        return load_truthful_qa(num_samples=num_samples, split=split)
    elif dataset_name == "faitheval":
        return load_faitheval(num_samples=num_samples, split=split)
    elif dataset_name == "spa-vl":
        return load_spa_vl(num_samples=num_samples, split=split)
    elif dataset_name == "mmhal-bench":
        return load_mmhal_bench(num_samples=num_samples, split=split)
    else:
        raise ValueError(f"Dataset '{dataset_name}' is not supported.")