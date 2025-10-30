# Complete Fine-Tuning Workflow Guide

## Synthesized from Week 6 (Day 1-5)

This comprehensive guide merges all techniques learned across Week 6 for creating a successful fine-tuning workflow.

---

## 📋 TABLE OF CONTENTS

1. [Environment Setup](#1-environment-setup)
2. [Data Acquisition](#2-data-acquisition)
3. [Data Exploration & Analysis](#3-data-exploration--analysis)
4. [Data Curation & Cleaning](#4-data-curation--cleaning)
5. [Data Processing & Tokenization](#5-data-processing--tokenization)
6. [Dataset Balancing & Sampling](#6-dataset-balancing--sampling)
7. [Train/Test Split](#7-traintest-split)
8. [Baseline Model Creation](#8-baseline-model-creation)
9. [Evaluation Framework](#9-evaluation-framework)
10. [Fine-Tuning Preparation](#10-fine-tuning-preparation)
11. [Fine-Tuning Execution](#11-fine-tuning-execution)
12. [Model Evaluation](#12-model-evaluation)
13. [Best Practices & Tips](#13-best-practices--tips)

---

## 1. ENVIRONMENT SETUP

### 1.1 Install Required Libraries

```python
# Core libraries
pip install datasets transformers openai anthropic python-dotenv
pip install matplotlib pandas numpy scikit-learn tqdm
pip install huggingface-hub wandb

# NLP libraries
pip install gensim scipy
```

### 1.2 Configure API Keys

```python
import os
from dotenv import load_dotenv
from huggingface_hub import login

# Load environment variables
load_dotenv(override=True)
os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')
os.environ['ANTHROPIC_API_KEY'] = os.getenv('ANTHROPIC_API_KEY')
os.environ['HF_TOKEN'] = os.getenv('HF_TOKEN')

# Login to HuggingFace
hf_token = os.environ['HF_TOKEN']
login(hf_token, add_to_git_credential=True)
```

### 1.3 Import Essential Modules

```python
import matplotlib.pyplot as plt
from datasets import load_dataset, Dataset, DatasetDict
from transformers import AutoTokenizer
from openai import OpenAI
import pickle
import json
import random
import numpy as np
from collections import Counter, defaultdict
```

---

## 2. DATA ACQUISITION

### 2.1 Load Dataset from HuggingFace

```python
# Load specific dataset
dataset = load_dataset(
    "dataset-name/repo-name",
    "subset-name",
    split="full",  # or "train", "test", etc.
    trust_remote_code=True
)

# For small bandwidth - load subset first
dataset = load_dataset(
    "dataset-name/repo-name",
    split="full[:1000]",  # First 1000 examples
    trust_remote_code=True
)
```

### 2.2 Alternative: Load Multiple Related Datasets

```python
dataset_names = [
    "Category1",
    "Category2",
    "Category3"
]

all_items = []
for dataset_name in dataset_names:
    loader = ItemLoader(dataset_name)
    all_items.extend(loader.load())
```

---

## 3. DATA EXPLORATION & ANALYSIS

### 3.1 Basic Statistics

```python
print(f"Total samples: {len(dataset):,}")

# Check data structure
print(dataset[0].keys())
print(dataset[0])
```

### 3.2 Analyze Key Fields

```python
# Count non-null values
valid_count = 0
for datapoint in dataset:
    if datapoint["target_field"]:  # e.g., "price"
        valid_count += 1

print(f"Valid entries: {valid_count:,} ({valid_count/len(dataset)*100:.1f}%)")
```

### 3.3 Distribution Analysis

```python
# Collect values for analysis
values = []
text_lengths = []

for datapoint in dataset:
    try:
        value = float(datapoint["target_field"])
        if value > 0:
            values.append(value)
            # Concatenate relevant text fields
            text = str(datapoint["field1"]) + str(datapoint["field2"])
            text_lengths.append(len(text))
    except ValueError:
        pass
```

### 3.4 Visualization

```python
# Plot distributions
plt.figure(figsize=(15, 6))

# Value distribution
plt.subplot(1, 2, 1)
plt.hist(values, bins=50, color="lightblue")
plt.title(f"Value Distribution (Avg: {np.mean(values):.2f})")
plt.xlabel("Value")
plt.ylabel("Count")

# Length distribution
plt.subplot(1, 2, 2)
plt.hist(text_lengths, bins=50, color="lightgreen")
plt.title(f"Text Length Distribution (Avg: {np.mean(text_lengths):.0f})")
plt.xlabel("Length (chars)")
plt.ylabel("Count")

plt.tight_layout()
plt.show()
```

---

## 4. DATA CURATION & CLEANING

### 4.1 Create Data Class for Structure

```python
class DataItem:
    """Structured representation of a training example"""

    def __init__(self, raw_data, label):
        self.raw_data = raw_data
        self.label = label
        self.cleaned_text = None
        self.token_count = 0
        self.include = False

        self.process()

    def scrub_text(self, text):
        """Clean and normalize text"""
        import re
        # Remove special characters
        text = re.sub(r'[:\[\]"{}【】\s]+', ' ', text).strip()
        # Remove product codes (7+ chars with numbers)
        words = text.split()
        clean_words = [w for w in words if len(w) < 7 or not any(c.isdigit() for c in w)]
        return " ".join(clean_words)

    def process(self):
        """Process and validate the data item"""
        # Extract relevant fields
        text = self.extract_text()

        # Apply cleaning
        self.cleaned_text = self.scrub_text(text)

        # Validate
        if self.validate():
            self.include = True

    def extract_text(self):
        """Extract and combine relevant text fields"""
        parts = []
        if self.raw_data.get('title'):
            parts.append(self.raw_data['title'])
        if self.raw_data.get('description'):
            parts.append('\n'.join(self.raw_data['description']))
        if self.raw_data.get('features'):
            parts.append('\n'.join(self.raw_data['features']))
        return '\n'.join(parts)

    def validate(self):
        """Check if item meets quality criteria"""
        return len(self.cleaned_text) >= 300  # Minimum character count
```

### 4.2 Filter and Process Dataset

```python
MIN_VALUE = 1
MAX_VALUE = 999
processed_items = []

for datapoint in dataset:
    try:
        value = float(datapoint["target_field"])
        if MIN_VALUE <= value <= MAX_VALUE:
            item = DataItem(datapoint, value)
            if item.include:
                processed_items.append(item)
    except (ValueError, KeyError):
        continue

print(f"Processed items: {len(processed_items):,}")
```

---

## 5. DATA PROCESSING & TOKENIZATION

### 5.1 Initialize Tokenizer

```python
from transformers import AutoTokenizer

# Choose base model (important for consistency)
BASE_MODEL = "meta-llama/Meta-Llama-3.1-8B"  # or your chosen model
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
```

### 5.2 Tokenize and Truncate

```python
MIN_TOKENS = 150
MAX_TOKENS = 160  # Adjust based on your needs

def process_with_tokenization(text):
    """Tokenize and truncate text to fit within limits"""
    # Tokenize
    tokens = tokenizer.encode(text, add_special_tokens=False)

    # Validate length
    if len(tokens) < MIN_TOKENS:
        return None

    # Truncate if needed
    if len(tokens) > MAX_TOKENS:
        tokens = tokens[:MAX_TOKENS]

    # Decode back to text
    truncated_text = tokenizer.decode(tokens)

    return truncated_text, len(tokens)
```

### 5.3 Create Prompts

```python
def create_prompt(item, include_answer=True):
    """Create a training prompt from the item"""
    # Question/instruction
    prompt = "How much does this cost to the nearest dollar?\n\n"

    # Add the processed text
    prompt += item.cleaned_text + "\n\n"

    # Add answer format
    if include_answer:
        prompt += f"Price is ${round(item.label)}.00"
    else:
        prompt += "Price is $"

    return prompt

# Apply to all items
for item in processed_items:
    item.prompt = create_prompt(item, include_answer=True)
    item.test_prompt = create_prompt(item, include_answer=False)
```

---

## 6. DATASET BALANCING & SAMPLING

### 6.1 Analyze Distribution by Category

```python
# Count by category
category_counts = Counter()
for item in processed_items:
    category_counts[item.category] += 1

# Visualize
categories = list(category_counts.keys())
counts = [category_counts[cat] for cat in categories]

plt.figure(figsize=(15, 6))
plt.bar(categories, counts, color="goldenrod")
plt.title('Category Distribution')
plt.xlabel('Categories')
plt.ylabel('Count')
plt.xticks(rotation=30, ha='right')
for i, v in enumerate(counts):
    plt.text(i, v, f"{v:,}", ha='center', va='bottom')
plt.show()
```

### 6.2 Balance by Target Value

```python
# Group by value/label
value_slots = defaultdict(list)
for item in processed_items:
    rounded_value = round(item.label)
    value_slots[rounded_value].append(item)

# Sample to balance distribution
np.random.seed(42)
random.seed(42)

balanced_sample = []
MAX_PER_SLOT = 1200

for value in range(MIN_VALUE, MAX_VALUE + 1):
    slot = value_slots[value]

    if len(slot) <= MAX_PER_SLOT:
        balanced_sample.extend(slot)
    else:
        # Apply weights if needed (e.g., prefer certain categories)
        weights = np.array([
            5 if item.category != 'OverrepresentedCategory' else 1
            for item in slot
        ])
        weights = weights / np.sum(weights)

        # Sample
        selected_indices = np.random.choice(
            len(slot),
            size=MAX_PER_SLOT,
            replace=False,
            p=weights
        )
        selected = [slot[i] for i in selected_indices]
        balanced_sample.extend(selected)

print(f"Balanced sample size: {len(balanced_sample):,}")
```

### 6.3 Verify Balancing

```python
# Plot new distribution
values = [item.label for item in balanced_sample]

plt.figure(figsize=(15, 6))
plt.hist(values, bins=50, color="darkblue", rwidth=0.7)
plt.title(f"Balanced Distribution (Avg: {np.mean(values):.2f})")
plt.xlabel("Value")
plt.ylabel("Count")
plt.show()
```

---

## 7. TRAIN/TEST SPLIT

### 7.1 Split Dataset

```python
# Shuffle
random.seed(42)
random.shuffle(balanced_sample)

# Split
TRAIN_SIZE = 400_000  # Adjust based on your needs
TEST_SIZE = 2_000

train_items = balanced_sample[:TRAIN_SIZE]
test_items = balanced_sample[TRAIN_SIZE:TRAIN_SIZE + TEST_SIZE]

print(f"Training set: {len(train_items):,}")
print(f"Test set: {len(test_items):,}")
```

### 7.2 Save for Later Use

```python
# Save as pickle files
import pickle

with open('train.pkl', 'wb') as f:
    pickle.dump(train_items, f)

with open('test.pkl', 'wb') as f:
    pickle.dump(test_items, f)

print("✓ Datasets saved to pickle files")
```

### 7.3 Upload to HuggingFace (Optional)

```python
# Convert to HF Dataset format
train_dataset = Dataset.from_dict({
    "text": [item.prompt for item in train_items],
    "label": [item.label for item in train_items]
})

test_dataset = Dataset.from_dict({
    "text": [item.test_prompt for item in test_items],
    "label": [item.label for item in test_items]
})

dataset_dict = DatasetDict({
    "train": train_dataset,
    "test": test_dataset
})

# Push to hub
HF_USER = "your-username"
DATASET_NAME = f"{HF_USER}/your-dataset-name"
dataset_dict.push_to_hub(DATASET_NAME, private=True)
```

---

## 8. BASELINE MODEL CREATION

### 8.1 Simple Baselines

```python
# Random baseline
def random_predictor(item):
    return random.randrange(MIN_VALUE, MAX_VALUE)

# Constant (average) baseline
training_values = [item.label for item in train_items]
average_value = np.mean(training_values)

def constant_predictor(item):
    return average_value
```

### 8.2 Traditional ML - Linear Regression

```python
from sklearn.linear_model import LinearRegression
from sklearn.feature_extraction.text import CountVectorizer

# Prepare data
documents = [item.test_prompt for item in train_items]
labels = np.array([item.label for item in train_items])

# Bag of Words
vectorizer = CountVectorizer(max_features=1000, stop_words='english')
X_train = vectorizer.fit_transform(documents)

# Train
lr_model = LinearRegression()
lr_model.fit(X_train, labels)

# Predictor function
def lr_predictor(item):
    x = vectorizer.transform([item.test_prompt])
    return max(0, lr_model.predict(x)[0])
```

### 8.3 Word2Vec + ML

```python
from gensim.models import Word2Vec
from gensim.utils import simple_preprocess

# Train Word2Vec
processed_docs = [simple_preprocess(doc) for doc in documents]
w2v_model = Word2Vec(
    sentences=processed_docs,
    vector_size=400,
    window=5,
    min_count=1,
    workers=8
)

# Create document vectors
def document_vector(doc):
    doc_words = simple_preprocess(doc)
    word_vectors = [w2v_model.wv[word] for word in doc_words if word in w2v_model.wv]
    return np.mean(word_vectors, axis=0) if word_vectors else np.zeros(w2v_model.vector_size)

X_w2v = np.array([document_vector(doc) for doc in documents])

# Train Random Forest
from sklearn.ensemble import RandomForestRegressor

rf_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=8)
rf_model.fit(X_w2v, labels)

def rf_predictor(item):
    doc_vec = document_vector(item.test_prompt)
    return max(0, rf_model.predict([doc_vec])[0])
```

---

## 9. EVALUATION FRAMEWORK

### 9.1 Create Test Harness

```python
import math

class Tester:
    """Unified testing framework for all models"""

    def __init__(self, predictor, data, title=None, size=250):
        self.predictor = predictor
        self.data = data
        self.title = title or predictor.__name__
        self.size = size
        self.guesses = []
        self.truths = []
        self.errors = []
        self.sles = []  # Squared Log Errors
        self.colors = []

    def color_for(self, error, truth):
        """Determine color based on error magnitude"""
        if error < 40 or error / truth < 0.2:
            return "green"
        elif error < 80 or error / truth < 0.4:
            return "orange"
        else:
            return "red"

    def run_datapoint(self, i):
        """Evaluate a single datapoint"""
        datapoint = self.data[i]
        guess = self.predictor(datapoint)
        truth = datapoint.label
        error = abs(guess - truth)

        # Calculate Squared Log Error
        log_error = math.log(truth + 1) - math.log(guess + 1)
        sle = log_error ** 2

        color = self.color_for(error, truth)

        self.guesses.append(guess)
        self.truths.append(truth)
        self.errors.append(error)
        self.sles.append(sle)
        self.colors.append(color)

        print(f"{i+1}: Guess: ${guess:.2f} Truth: ${truth:.2f} Error: ${error:.2f}")

    def chart(self, title):
        """Create scatter plot of predictions vs truth"""
        plt.figure(figsize=(12, 8))
        max_val = max(max(self.truths), max(self.guesses))

        # Perfect prediction line
        plt.plot([0, max_val], [0, max_val], color='deepskyblue', lw=2, alpha=0.6)

        # Scatter plot
        plt.scatter(self.truths, self.guesses, s=3, c=self.colors)
        plt.xlabel('Ground Truth')
        plt.ylabel('Model Estimate')
        plt.xlim(0, max_val)
        plt.ylim(0, max_val)
        plt.title(title)
        plt.show()

    def report(self):
        """Generate final report with metrics"""
        avg_error = np.mean(self.errors)
        rmsle = math.sqrt(np.mean(self.sles))
        hits = sum(1 for c in self.colors if c == "green")

        print(f"\n{'='*60}")
        print(f"Model: {self.title}")
        print(f"Average Error: ${avg_error:.2f}")
        print(f"RMSLE: {rmsle:.4f}")
        print(f"Accuracy (within threshold): {hits/self.size*100:.1f}%")
        print(f"{'='*60}\n")

        title = f"{self.title} | Error=${avg_error:.2f} | RMSLE={rmsle:.2f} | Hits={hits/self.size*100:.1f}%"
        self.chart(title)

    def run(self):
        """Run the complete evaluation"""
        for i in range(self.size):
            self.run_datapoint(i)
        self.report()

    @classmethod
    def test(cls, function, data, size=250):
        """Convenience method to test a predictor function"""
        cls(function, data, size=size).run()
```

### 9.2 Use the Tester

```python
# Test any model
Tester.test(random_predictor, test_items)
Tester.test(lr_predictor, test_items)
Tester.test(rf_predictor, test_items)
```

---

## 10. FINE-TUNING PREPARATION

### 10.1 Select Training Subset

```python
# For OpenAI: 50-200 examples recommended for small tasks
FINE_TUNE_SIZE = 200
VALIDATION_SIZE = 50

fine_tune_train = train_items[:FINE_TUNE_SIZE]
fine_tune_val = train_items[FINE_TUNE_SIZE:FINE_TUNE_SIZE + VALIDATION_SIZE]
```

### 10.2 Format for OpenAI Fine-Tuning

```python
def create_messages(item, include_answer=True):
    """Create OpenAI chat format messages"""
    system_msg = "You estimate prices of items. Reply only with the price, no explanation"

    # Clean up the user prompt
    user_prompt = item.test_prompt.replace(" to the nearest dollar", "")
    user_prompt = user_prompt.replace("\n\nPrice is $", "")

    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_prompt}
    ]

    if include_answer:
        messages.append({
            "role": "assistant",
            "content": f"Price is ${item.label:.2f}"
        })
    else:
        messages.append({
            "role": "assistant",
            "content": "Price is $"
        })

    return messages
```

### 10.3 Create JSONL Files

```python
def create_jsonl(items, filename):
    """Convert items to JSONL format for OpenAI"""
    with open(filename, 'w') as f:
        for item in items:
            messages = create_messages(item, include_answer=True)
            json_line = json.dumps({"messages": messages})
            f.write(json_line + '\n')

    print(f"✓ Created {filename} with {len(items)} examples")

# Create files
create_jsonl(fine_tune_train, "fine_tune_train.jsonl")
create_jsonl(fine_tune_val, "fine_tune_validation.jsonl")
```

### 10.4 Upload to OpenAI

```python
client = OpenAI()

# Upload training file
with open("fine_tune_train.jsonl", "rb") as f:
    train_file = client.files.create(file=f, purpose="fine-tune")

# Upload validation file
with open("fine_tune_validation.jsonl", "rb") as f:
    val_file = client.files.create(file=f, purpose="fine-tune")

print(f"Training file ID: {train_file.id}")
print(f"Validation file ID: {val_file.id}")
```

---

## 11. FINE-TUNING EXECUTION

### 11.1 Setup Weights & Biases (Optional but Recommended)

```python
# 1. Create account at https://wandb.ai
# 2. Get API key from Settings
# 3. Add to OpenAI dashboard at platform.openai.com/account/organization

wandb_integration = {
    "type": "wandb",
    "wandb": {"project": "your-project-name"}
}
```

### 11.2 Create Fine-Tuning Job

```python
# Create the job
job = client.fine_tuning.jobs.create(
    training_file=train_file.id,
    validation_file=val_file.id,
    model="gpt-4o-mini-2024-07-18",  # or "gpt-3.5-turbo-0125"
    seed=42,
    hyperparameters={
        "n_epochs": 3  # Adjust based on your needs
    },
    integrations=[wandb_integration],  # Optional
    suffix="your-model-name"
)

print(f"✓ Fine-tuning job created!")
print(f"Job ID: {job.id}")
print(f"Status: {job.status}")
```

### 11.3 Monitor Training

```python
# Check status
job_status = client.fine_tuning.jobs.retrieve(job.id)
print(f"Status: {job_status.status}")

# List recent events
events = client.fine_tuning.jobs.list_events(
    fine_tuning_job_id=job.id,
    limit=10
)
for event in events.data:
    print(event)

# Get all jobs
all_jobs = client.fine_tuning.jobs.list(limit=10)
for j in all_jobs.data:
    print(f"{j.id}: {j.status}")
```

### 11.4 Sync with Weights & Biases

```python
import wandb
from wandb.integration.openai.fine_tuning import WandbLogger

# Login
wandb.login()

# Sync the job
WandbLogger.sync(
    fine_tune_job_id=job.id,
    project="your-project-name"
)
```

---

## 12. MODEL EVALUATION

### 12.1 Get Fine-Tuned Model Name

```python
# Wait for training to complete, then retrieve
completed_job = client.fine_tuning.jobs.retrieve(job.id)
fine_tuned_model = completed_job.fine_tuned_model

print(f"✓ Fine-tuned model: {fine_tuned_model}")
```

### 12.2 Create Predictor Function

```python
import re

def extract_price(text):
    """Extract numeric price from text"""
    text = text.replace('$', '').replace(',', '')
    match = re.search(r"[-+]?\d*\.\d+|\d+", text)
    return float(match.group()) if match else 0

def fine_tuned_predictor(item):
    """Use fine-tuned model to predict"""
    messages = create_messages(item, include_answer=False)

    response = client.chat.completions.create(
        model=fine_tuned_model,
        messages=messages,
        seed=42,
        max_tokens=7,
        temperature=0  # Use 0 for consistent predictions
    )

    reply = response.choices[0].message.content
    return extract_price(reply)
```

### 12.3 Evaluate Performance

```python
# Test on your test set
Tester.test(fine_tuned_predictor, test_items, size=250)

# Compare with baseline
print("\n" + "="*60)
print("COMPARISON")
print("="*60)
Tester.test(constant_predictor, test_items, size=250)
Tester.test(fine_tuned_predictor, test_items, size=250)
```

### 12.4 Cost Analysis

```python
def estimate_cost(num_predictions, model="gpt-4o-mini"):
    """Estimate API costs"""
    # Approximate token counts
    avg_input_tokens = 200
    avg_output_tokens = 5

    # Pricing (per 1M tokens) - adjust based on current pricing
    if model == "gpt-4o-mini":
        input_cost_per_1m = 0.30
        output_cost_per_1m = 1.20

    input_cost = (num_predictions * avg_input_tokens / 1_000_000) * input_cost_per_1m
    output_cost = (num_predictions * avg_output_tokens / 1_000_000) * output_cost_per_1m

    total_cost = input_cost + output_cost

    print(f"Estimated cost for {num_predictions:,} predictions:")
    print(f"  Input: ${input_cost:.4f}")
    print(f"  Output: ${output_cost:.4f}")
    print(f"  Total: ${total_cost:.4f}")

    return total_cost

estimate_cost(250)
```

---

## 13. BEST PRACTICES & TIPS

### 13.1 Data Curation Tips

✅ **Quality over quantity**: 200 high-quality examples often better than 2000 noisy ones
✅ **Balance your dataset**: Avoid skewed distributions
✅ **Clean thoroughly**: Remove irrelevant characters, product codes, etc.
✅ **Consistent formatting**: Ensure prompts follow same structure
✅ **Token awareness**: Know your model's tokenization behavior

### 13.2 Tokenization Best Practices

✅ **Match your base model**: Use same tokenizer you'll fine-tune
✅ **Set limits**: MIN_TOKENS and MAX_TOKENS based on task
✅ **Validate token counts**: Ensure training and inference match
✅ **Test special tokens**: Understand how numbers/symbols tokenize

### 13.3 Training Tips

✅ **Start small**: Test with 50-200 examples first
✅ **Use validation set**: Monitor overfitting
✅ **Track with W&B**: Visualize training metrics
✅ **Set random seeds**: Ensure reproducibility
✅ **Save checkpoints**: Use pickle for intermediate results

### 13.4 Evaluation Best Practices

✅ **Multiple metrics**: Don't rely on single metric (use RMSLE, MAE, accuracy)
✅ **Visual inspection**: Scatter plots reveal patterns
✅ **Error analysis**: Examine failures to improve
✅ **Compare baselines**: Always beat simple models first
✅ **Test on holdout**: Never evaluate on training data

### 13.5 Cost Optimization

✅ **Start with mini models**: gpt-4o-mini before gpt-4o
✅ **Limit epochs**: 1-3 often sufficient
✅ **Reduce dataset**: Sample intelligently
✅ **Cache results**: Save predictions to avoid re-running
✅ **Monitor usage**: Track API costs in real-time

### 13.6 Common Pitfalls to Avoid

❌ **Data leakage**: Don't include answers in test prompts
❌ **Inconsistent preprocessing**: Match training and inference
❌ **Overfitting**: Too many epochs on small dataset
❌ **Ignoring distribution**: Unbalanced classes hurt performance
❌ **Poor prompts**: Vague instructions yield poor results

### 13.7 Hyperparameter Tuning

```python
# Key hyperparameters to experiment with:
hyperparameters = {
    "n_epochs": [1, 2, 3],           # Number of training passes
    "learning_rate_multiplier": [0.5, 1, 2],  # Learning rate adjustment
    "batch_size": [1, 2, 4]          # Samples per batch (if supported)
}

# Test systematically
for n_epochs in [1, 2, 3]:
    job = client.fine_tuning.jobs.create(
        training_file=train_file.id,
        model="gpt-4o-mini-2024-07-18",
        hyperparameters={"n_epochs": n_epochs},
        suffix=f"model-epochs-{n_epochs}"
    )
    # Monitor and compare results
```

### 13.8 Workflow Checklist

**Before Fine-Tuning:**

- [ ] Data explored and visualized
- [ ] Data cleaned and validated
- [ ] Token counts verified
- [ ] Train/test split created
- [ ] Baseline models tested
- [ ] Evaluation framework ready
- [ ] Costs estimated

**During Fine-Tuning:**

- [ ] Validation set included
- [ ] Monitoring enabled (W&B)
- [ ] Random seeds set
- [ ] Hyperparameters documented
- [ ] Progress tracked

**After Fine-Tuning:**

- [ ] Model evaluated on test set
- [ ] Results compared to baselines
- [ ] Error analysis completed
- [ ] Costs reviewed
- [ ] Model saved/documented

---

## 14. COMPLETE WORKFLOW EXAMPLE

Here's a complete end-to-end example:

```python
# 1. Setup
from openai import OpenAI
import pickle
import random
import numpy as np

client = OpenAI()
random.seed(42)
np.random.seed(42)

# 2. Load data
with open('train.pkl', 'rb') as f:
    train = pickle.load(f)
with open('test.pkl', 'rb') as f:
    test = pickle.load(f)

# 3. Prepare fine-tuning data
fine_tune_train = train[:200]
fine_tune_val = train[200:250]

# 4. Create JSONL
def create_jsonl(items, filename):
    with open(filename, 'w') as f:
        for item in items:
            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": item.test_prompt},
                {"role": "assistant", "content": str(item.label)}
            ]
            f.write(json.dumps({"messages": messages}) + '\n')

create_jsonl(fine_tune_train, "train.jsonl")
create_jsonl(fine_tune_val, "val.jsonl")

# 5. Upload
with open("train.jsonl", "rb") as f:
    train_file = client.files.create(file=f, purpose="fine-tune")
with open("val.jsonl", "rb") as f:
    val_file = client.files.create(file=f, purpose="fine-tune")

# 6. Fine-tune
job = client.fine_tuning.jobs.create(
    training_file=train_file.id,
    validation_file=val_file.id,
    model="gpt-4o-mini-2024-07-18",
    hyperparameters={"n_epochs": 3}
)

# 7. Wait and retrieve
# ... wait for completion ...
job_status = client.fine_tuning.jobs.retrieve(job.id)
model_name = job_status.fine_tuned_model

# 8. Test
def predictor(item):
    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": item.test_prompt}
        ]
    )
    return float(response.choices[0].message.content)

# 9. Evaluate
Tester.test(predictor, test)
```

---

## 15. ADDITIONAL RESOURCES

### Documentation

- [OpenAI Fine-Tuning Guide](https://platform.openai.com/docs/guides/fine-tuning)
- [HuggingFace Datasets](https://huggingface.co/docs/datasets)
- [Weights & Biases](https://docs.wandb.ai/)

### Tools

- **Weights & Biases**: Track experiments
- **HuggingFace Hub**: Store datasets/models
- **Jupyter**: Interactive development
- **Git**: Version control

### Community

- OpenAI Community Forum
- HuggingFace Forums
- Reddit: r/MachineLearning
- Discord: Various AI communities

---

## Summary

This workflow synthesizes all techniques from Week 6:

1. **Environment setup** with proper API keys
2. **Data acquisition** from HuggingFace
3. **Exploration** with visualization
4. **Curation** with cleaning and filtering
5. **Tokenization** with proper truncation
6. **Balancing** for better distribution
7. **Splitting** into train/test sets
8. **Baseline** models for comparison
9. **Evaluation** framework for testing
10. **Preparation** of JSONL format
11. **Fine-tuning** on OpenAI
12. **Evaluation** of results

Follow this workflow for any fine-tuning project!

---

_Created from Week 6 materials (Day 1-5)_
_Last updated: 2025_
