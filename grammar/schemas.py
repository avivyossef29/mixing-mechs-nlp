import re
import pandas as pd
from grammar.grammar import Schema, Templates, Query, TaskFactory

PEOPLE_NAMES = [
    "John", "Mary", "Carla", "Bob", "Sam", "Alex", "Emma", "David",
    "Sarah", "Michael", "Lisa", "James", "Anna", "Daniel", "Sophie",
    "Chris", "Rachel", "Tom", "Nina", "Peter", "Laura", "George", "Mark",
]

BOX_VARS = [f"box_{i}" for i in range(1, 101)]
PROG_VARS = [f"var_{i}" for i in range(1, 101)]

HOUSEHOLD_ITEMS = [
    "egg", "fan", "tea", "engine", "plate", "gift", "wire", "watch", "cross", "boat", 
    "game", "rose", "shell", "seed", "magnet", "suit", "ticket", "glass", "tie", "card", 
    "brain", "fig", "wheel", "machine", "note", "drink", "bread", "camera", "bill", 
    "chemical", "clock", "flower", "creature", "rock", "plant", "sheet", "leaf", 
    "block", "newspaper", "disk", "boot", "medicine", "coffee", "book", "ball", 
    "string", "fish", "crown", "branch", "phone", "plane", "apple", "cup", "bell", 
    "brick", "document", "file", "bus", "bag", "drug", "pot", "computer", "mirror", 
    "stone", "radio", "dress", "meat", "train", "bomb", "letter", "guitar", "hat", 
    "map", "magazine", "coat", "television", "painting", "picture", "milk", "pipe", 
    "ice", "key"
]

SCHEMA_BOXES = Schema(
    name="boxes",
    items={"Object": HOUSEHOLD_ITEMS, "Box": BOX_VARS},
    templates=Templates(
        prefix="",
        definitions={
            "row_default": "the {Object} is in {Box}",
            "ordering_01": "the {Object} is in {Box}",
        },
        queries={
            "Q:Box A:Object": Query(
                question="Respond in one word, only the answer and nothing else: What does {Box} contain?",
                answer_category="Object",
            ),
            "Q:Object A:Box": Query(
                question="Respond in one word, only the answer and nothing else: Which identifier is the {Object} in?",
                answer_category="Box",
            ),
        },
        capitalize_first_clause=True,
    ),
    max_new_tokens=5,
    # Substring checker for robust matching
    checker=lambda neural, causal: causal.strip().lower() in neural.strip().lower(),
    matchers=[
        lambda s: re.match(f"^ ?({'|'.join(map(re.escape, HOUSEHOLD_ITEMS))})$", s) is not None,
        lambda s: re.match(r"^ ?box_\d+$", s) is not None,
    ],
)

# --- Updated Schema: Programming Dict (100 variables) ---
SCHEMA_PROGRAMMING_PEOPLE_DICT = Schema(
    name="programming_people_dict",
    items={
        "VariableName": PROG_VARS, 
        "Name": PEOPLE_NAMES, 
        "Country": ["USA", "Israel", "UK", "Canada", "Germany", "France", "Japan"]
    },
    templates=Templates(
        definitions={
            "row_default": '{VariableName} = {{"name": " {Name}", "country": " {Country}"}}',
            "ordering_012": '{VariableName} = {{"name": " {Name}", "country": " {Country}"}}',
        },
        queries={
            "default": Query(
                question='Respond in one word, only the answer and nothing else: What is the country in variable {VariableName} where name="{Name}"?',
                answer_category="Country",
            ),
        },
        capitalize_first_clause=False,
        prefix="The following are dictionary variables in Python: ",
    ),
    matchers=[
        lambda s: re.match(r"^ ?var_\d+$", s) is not None,
        lambda s: re.match(f"^ ?({'|'.join(map(re.escape, PEOPLE_NAMES))})$", s) is not None,
    ],
)

if __name__ == "__main__":
    schemas = [
        SCHEMA_BOXES, 
        SCHEMA_PROGRAMMING_PEOPLE_DICT
    ]

    rows = []
    for schema in schemas:
        task_factory = TaskFactory(schema)
        # Generate 100 instances to fill the context
        task_instance = task_factory.create_task_instance(num_instances=100)
        task = task_instance.generate_task(definition_key="row_default", query_instance_idx=0)
        
        final_form = f"{task['context']} {task['question']}"
        rows.append({
            "Name": schema.name, 
            "Task": final_form, 
            "Answer": task["answer"]
        })

    df = pd.DataFrame(rows)
    # df.to_csv("experiment_results_100.csv", index=False)
    print("Sample task for SCHEMA_BOXES:")
    print(df[df['Name'] == 'boxes']['Task'].values[0][:200] + "...")
