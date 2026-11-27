import re

HEROES = {
    "spider-man": (
        "Spider-Man",
        ("spider man", "spiderman", "spider verse", "peter parker", "miles morales"),
    ),
    "batman": ("Batman", ("batman", "bruce wayne", "dark knight")),
    "superman": ("Superman", ("superman", "clark kent", "man of steel")),
    "wonder-woman": ("Wonder Woman", ("wonder woman", "diana prince")),
    "x-men": ("X-Men", ("x men", "wolverine", "professor x", "magneto")),
    "fantastic-four": (
        "Fantastic Four",
        ("fantastic four", "fantastic 4", "reed richards", "sue storm"),
    ),
}


def name_pattern(names: tuple[str, ...]) -> re.Pattern:
    separator = r"[\s\-\u2010-\u2015]+"
    aliases = [
        separator.join(re.escape(word) for word in name.split()) for name in names
    ]
    return re.compile(r"(?<!\w)(?:" + "|".join(aliases) + r")(?!\w)", re.IGNORECASE)


HERO_PATTERNS = {key: name_pattern(names) for key, (_, names) in HEROES.items()}
SUPERHERO_PATTERN = name_pattern(
    (
        *(name for _, names in HEROES.values() for name in names),
        "superhero",
        "superheroes",
        "super hero",
        "supervillain",
        "superhuman powers",
        "captain america",
        "iron man",
        "black panther",
        "deadpool",
        "hulk",
        "black adam",
        "shazam",
        "green lantern",
        "blue beetle",
    )
)


def superhero_evidence(title: str, plot: str, hero: str) -> tuple[str, bool] | None:
    pattern = HERO_PATTERNS.get(hero, SUPERHERO_PATTERN)
    for label, text in (("Title", title), ("Plot", plot)):
        if match := pattern.search(text):
            return f"{label} mentions {match.group()}", label == "Title"
    return None
