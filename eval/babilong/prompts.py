"""
BABILong task-specific prompts and label definitions.

Based on the official BABILong evaluation protocol (RMT-team/babilong).
"""

# Valid answer labels per task (for label matching evaluation)
TASK_LABELS = {
    "qa1": {"bathroom", "bedroom", "garden", "hallway", "kitchen", "office"},
    "qa2": {"bathroom", "bedroom", "garden", "hallway", "kitchen", "office"},
    "qa3": {"bathroom", "bedroom", "garden", "hallway", "kitchen", "office"},
    "qa4": {"bathroom", "bedroom", "garden", "hallway", "kitchen", "office"},
    "qa5": {"bill", "fred", "jeff", "mary", "apple", "football", "milk"},
    "qa6": {"yes", "no"},
    "qa7": {"none", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"},
    "qa8": {"apple", "football", "milk", "nothing"},
    "qa9": {"yes", "no"},
    "qa10": {"yes", "no", "maybe"},
}

# Task-specific instructions
TASK_INSTRUCTIONS = {
    "qa1": "Answer the question based on the context. The answer is a single location.",
    "qa2": "Answer the question based on the context. The answer is a single location.",
    "qa3": "Answer the question based on the context. The answer is a single location.",
    "qa4": "Answer the question based on the context. The answer is a single location.",
    "qa5": "Answer the question based on the context. The answer is a single word.",
    "qa6": "Answer the question based on the context. Answer yes or no.",
    "qa7": "Answer the question based on the context. The answer is a number word.",
    "qa8": "Answer the question based on the context. List the objects.",
    "qa9": "Answer the question based on the context. Answer yes or no.",
    "qa10": "Answer the question based on the context. Answer yes, no, or maybe.",
}

# Few-shot examples per task (from 0k split)
TASK_EXAMPLES = {
    "qa1": [
        {
            "context": "Mary moved to the bathroom. John went to the hallway.",
            "question": "Where is Mary?",
            "answer": "bathroom",
        },
        {
            "context": "Daniel went back to the hallway. Sandra moved to the garden.",
            "question": "Where is Sandra?",
            "answer": "garden",
        },
    ],
    "qa2": [
        {
            "context": "John moved to the bedroom. Mary grabbed the football. Mary went to the hallway.",
            "question": "Where is the football?",
            "answer": "hallway",
        },
    ],
    "qa3": [
        {
            "context": "Sandra picked up the milk. Sandra went to the office. Sandra put down the milk. Sandra moved to the bathroom.",
            "question": "Where was the milk before the bathroom?",
            "answer": "office",
        },
    ],
    "qa4": [
        {
            "context": "The hallway is north of the bathroom. The bedroom is west of the bathroom.",
            "question": "What is north of the bathroom?",
            "answer": "hallway",
        },
    ],
    "qa5": [
        {
            "context": "Fred gave the apple to Jeff. Jeff gave the apple to Bill.",
            "question": "Who did Jeff give the apple to?",
            "answer": "bill",
        },
    ],
    "qa6": [
        {
            "context": "John is in the kitchen. Mary is in the garden.",
            "question": "Is John in the kitchen?",
            "answer": "yes",
        },
    ],
    "qa7": [
        {
            "context": "Sandra grabbed the football. Sandra dropped the football. Sandra picked up the milk.",
            "question": "How many objects is Sandra carrying?",
            "answer": "one",
        },
    ],
    "qa8": [
        {
            "context": "Daniel took the apple. Daniel got the milk.",
            "question": "What is Daniel carrying?",
            "answer": "apple,milk",
        },
    ],
    "qa9": [
        {
            "context": "Sandra travelled to the office. Fred is no longer in the office.",
            "question": "Is Fred in the office?",
            "answer": "no",
        },
    ],
    "qa10": [
        {
            "context": "John is either in the kitchen or the bathroom. John moved to the garden.",
            "question": "Is John in the garden?",
            "answer": "yes",
        },
    ],
}


def build_prompt(task, context, question):
    """Build a few-shot prompt for a BABILong example."""
    instruction = TASK_INSTRUCTIONS.get(task, TASK_INSTRUCTIONS["qa1"])
    examples = TASK_EXAMPLES.get(task, TASK_EXAMPLES["qa1"])

    parts = [instruction, ""]

    for ex in examples:
        parts.append(f"<context>\n{ex['context']}\n</context>")
        parts.append(f"Question: {ex['question']}")
        parts.append(f"Answer: {ex['answer']}")
        parts.append("")

    parts.append("Now answer the following:")
    parts.append("")
    parts.append(f"<context>\n{context}\n</context>")
    parts.append(f"Question: {question}")
    parts.append("Answer:")

    return "\n".join(parts)
