"""
Constants, prompt templates, and group mappings for the DOSA benchmark.
"""

GROUPS = ["Telugu", "Punjabi", "Bengali", "Marathi", "Hindi-Urdu"]

MODELS = ["tiny-aya-global", "tiny-aya-fire", "tiny-aya-earth", "tiny-aya-water"]

STATE_TO_GROUP = {
    "andhra_pradesh": "Telugu",
    "telangana":      "Telugu",
    "punjab":         "Punjabi",
    "west_bengal":    "Bengali",
    "maharashtra":    "Marathi",
    "uttar_pradesh":  "Hindi-Urdu",
    "bihar":          "Hindi-Urdu",
    "delhi":          "Hindi-Urdu",
    "rajasthan":      "Hindi-Urdu",
}

DOSA_BASE_URL = (
    "https://raw.githubusercontent.com/microsoft/DOSA/main/data/{state}/original_artifacts.csv"
)

# Prompts replicating the DOSA paper exactly (Appendix A, arXiv:2403.14651)
SYSTEM_PROMPT_TEMPLATE = (
    "You are an agent who is well-versed in the cultures of the world. "
    "You are playing a game of taboo with another agent who is also well-versed with the "
    "cultures of the world. You can only make two guesses to identify this social artifact "
    "correctly, and you cannot ask any clarification questions. "
    "Social artifacts are objects that help us connect and stay associated with the culture. "
    "These objects are known and have significance to most people who consider themselves as "
    "a part of that culture and serve as a way of identifying themselves with the culture and "
    "the people in that culture. Your clues are: {clues}"
)

INSTRUCTION_PROMPT_1 = (
    "Name the object based on the above clues from {state}. "
    "I do not need to know your reasoning behind the answer. "
    "Just tell me the answer and nothing else. "
    "If you do not know the answer, say that you do not know the answer. "
    "Format your answer in the form of ANSWER: your_answer_here."
)

INSTRUCTION_PROMPT_2 = (
    "Your first guess is not correct. "
    "While making your second guess, please stick to the format as ANSWER: your_answer_here."
)
