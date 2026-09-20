"""Reading a street off a reply the way a person reads one off a sign.

The courier used to choose its street by a number the observation assigned:
``walk_to(3)``. The numbers are stable within a junction -- clockwise from
north -- and mean nothing across junctions, so a policy could not carry any
fact about a street from one corner to the next. Rue de Grenelle is street 3
here, street 1 there and absent at the corner after that; "I already tried
that one" is not expressible.

A street name is the same everywhere, and so is a compass bearing. Together
they say exactly one thing at 100% of the map's junctions, which numbers also
did, while remaining true a hundred metres later, which numbers did not. That
is the whole reason for the change: not realism for its own sake, but a name
the policy can remember.

What this module does is the unglamorous half -- letting the model write the
name the way a person would. The map has accents (Rue de Sévigné),
disambiguating suffixes (Avenue Bonaparte (2)), and eight compass bearings
that can be written six ways each. A reply that names the right street and is
refused on spelling measures typing, not navigation.
"""

from __future__ import annotations

import re
import unicodedata

# The eight bearings the map uses, and what a model might write for each. The
# aliases are not a guess: they are the forms that appear in the route
# instructions the phone gives ("head north-east"), plus the two abbreviation
# styles anyone writes without thinking.
COMPASS: dict[str, tuple[str, ...]] = {
    "north": ("n", "north", "northward", "northwards", "up"),
    "north-east": ("ne", "northeast", "north east", "north-east"),
    "east": ("e", "east", "eastward", "eastwards"),
    "south-east": ("se", "southeast", "south east", "south-east"),
    "south": ("s", "south", "southward", "southwards", "down"),
    "south-west": ("sw", "southwest", "south west", "south-west"),
    "west": ("w", "west", "westward", "westwards"),
    "north-west": ("nw", "northwest", "north west", "north-west"),
}

_ORDERED = ("north", "north-east", "east", "south-east",
            "south", "south-west", "west", "north-west")
# "left"/"right" are relative to facing, not points of the compass.
_RELATIVE_TURN = {"left": -90.0, "right": 90.0}


def resolve_relative(heading: str | None, facing: float | None) -> str | None:
    """Turn "left"/"right" into a compass heading against ``facing``; anything
    else passes through unchanged."""
    if not heading:
        return heading
    turn = _RELATIVE_TURN.get(heading.strip().lower())
    if turn is None or facing is None:
        return heading
    return _ORDERED[int(((facing + turn) % 360.0) / 45.0 + 0.5) % 8]

_ALIAS = {alias: heading for heading, aliases in COMPASS.items() for alias in aliases}

# Words that decorate a bearing in ordinary speech and carry no information:
# "head west", "west overall", "going west", "to the west".
_FILLER = ("head", "heading", "go", "going", "toward", "towards", "to", "the",
           "overall", "roughly", "about", "continue", "continuing", "then")


def fold(text: str) -> str:
    """A street name reduced to what distinguishes it from another street.

    Accents go, case goes, punctuation goes, runs of space collapse. What
    survives is enough to tell Rue de Sévigné from Rue de Seine and not enough
    to tell it from "rue de sevigne", which is the point -- an agent that read
    the sign correctly and typed it without the accent has not made a
    navigational error.

    The "(2)" suffix is kept, because it is the map's own way of saying these
    are two different streets that share a name. Dropping it would merge them.
    """
    text = unicodedata.normalize("NFKD", str(text))
    text = "".join(c for c in text if not unicodedata.combining(c))
    text = text.lower()
    text = re.sub(r"[^a-z0-9()]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def base_name(text: str) -> str:
    """The name without its disambiguating suffix: "rue du bac (2)" -> "rue du bac"."""
    return re.sub(r"\s*\(\d+\)\s*$", "", fold(text)).strip()


def read_heading(text: str | None) -> str | None:
    """The compass bearing a phrase names, or None if it names none.

    Accepts what the phone's own route instructions say ("head north-east"),
    what the observation prints ("north-east"), and the abbreviations a model
    reaches for ("NE", "north east"). Returns the canonical form the map uses,
    so callers compare like with like.
    """
    if not text:
        return None
    words = [w for w in fold(text).replace("-", " ").split() if w not in _FILLER]
    if not words:
        return None
    joined = " ".join(words)
    for candidate in (joined, joined.replace(" ", "-"), joined.replace(" ", "")):
        if candidate in _ALIAS:
            return _ALIAS[candidate]
    # A two-word compound written apart, with something else in the phrase.
    for i in range(len(words) - 1):
        pair = f"{words[i]}-{words[i + 1]}"
        if pair in _ALIAS:
            return _ALIAS[pair]
    for word in words:
        if word in _ALIAS:
            return _ALIAS[word]
    return None


class StreetNotHere(Exception):
    """No street of that name leaves this junction."""


class StreetAmbiguous(Exception):
    """The name is here more than once and the bearing did not separate them."""

    def __init__(self, message: str, headings: list[str]):
        super().__init__(message)
        self.headings = headings


def match_street(rows: list[dict], street: str, heading: str | None = None) -> dict:
    """Which of this junction's streets the courier meant.

    ``rows`` are the junction's candidates, each with ``street`` and
    ``heading``. Raises rather than returning None, because every failure here
    has a different thing to tell the courier and a bare None would flatten
    them into one unhelpful refusal.

    A bearing is only required when the name alone is ambiguous, which on this
    map is 73% of junctions -- most streets arrive at a corner and leave it
    again, so "Rue Mouffetard" names two ways to walk. Where the name is
    unique the bearing is optional and, if given and wrong, is corrected
    rather than refused: the street is the decision, the bearing was how the
    courier described it.
    """
    wanted = fold(street)
    named = [row for row in rows if fold(row["street"]) == wanted]
    if not named:
        # The suffix is the map's business, not the courier's. Someone reading
        # "Rue du Bac" off a sign cannot know the compiler called this one
        # "Rue du Bac (2)", so a base-name match counts.
        base = base_name(street)
        named = [row for row in rows if base_name(row["street"]) == base]
    if not named:
        raise StreetNotHere(street)

    if len(named) == 1:
        return named[0]

    asked = read_heading(heading)
    if asked is None:
        raise StreetAmbiguous(street, sorted(row["heading"] for row in named))
    exact = [row for row in named if row["heading"] == asked]
    if len(exact) == 1:
        return exact[0]
    if not exact:
        raise StreetAmbiguous(street, sorted(row["heading"] for row in named))
    # Two ways down the same street with the same bearing is a property of the
    # graph, not of the reply; take the nearer, which is what "walk down it"
    # means from where the courier stands.
    return min(exact, key=lambda row: row.get("distance_m", 0.0))
